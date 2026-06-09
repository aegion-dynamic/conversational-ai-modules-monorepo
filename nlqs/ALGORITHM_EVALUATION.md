# NLQS Algorithm — Current-State Evaluation & Fix Plan

This document evaluates the current state of the Natural Language Query System
(NLQS) algorithm, enumerates the gaps that prevent it from working as designed,
and proposes a concrete fix for each gap.

The reference design is described in [`nlqs/README.md`](./README.md). The live
implementation lives in:

- `nlqs/nlqs.py` — the orchestrating workflow (`NLQS.execute_nlqs_query_workflow`)
- `nlqs/summarization.py` — LLM summarization / intent / column classification
- `nlqs/query_construction.py` — SQL fragment construction
- `nlqs/search_field.py` — runs the fragments and aggregates results
- `nlqs/vectordb_driver.py` / `nlqs/neondb_driver.py` — vector backends

---

## TL;DR

The pipeline *runs end-to-end for a single-field query* (e.g. a pure descriptive
search), but it does **not** implement the core algorithm described in the README,
and several robustness/wiring issues make it brittle or wrong for anything more
complex. The most important problem is that **the search does a UNION (OR) of all
constraints instead of the INTERSECTION (AND) the README specifies**, so
multi-constraint queries return far too many (wrong) rows. There is also a set of
crash-risk bugs, vector-backend/config mismatches, dead code, and out-of-date
tests.

Gaps are grouped by severity:

- **A. Algorithm-correctness gaps** (results are wrong even when nothing crashes)
- **B. Robustness / crash gaps** (the workflow fails outright on valid input)
- **C. Vector-backend & configuration gaps**
- **D. Dead / duplicated code**
- **E. Test-suite gaps**

---

## A. Algorithm-correctness gaps

### A1 — Search performs a UNION of constraints, not the INTERSECTION the design requires

**Where:** `nlqs/search_field.py` `run_queries()` (the four `for` loops that call
`results.extend(result)`), consumed by `nlqs/nlqs.py` (the
`search_results["default"]` loop).

**What's wrong:** Each query fragment is turned into its own
`SELECT * FROM <table> WHERE <fragment>` and every result row is appended into one
flat list. That is an OR across *all* fragments and *all* field types
(identifier / quantitative / categorical / descriptive). The README §4 explicitly
requires the opposite:

> Both qualitative and quantitative primary keys are compared to find common
> entries … If no intersection is found, a union … is created [as a fallback].

So a query like *"high-CBD products that taste like s'mores"* should return rows
matching **both** the CBD constraint **and** the descriptive match. Today it
returns everything matching **either**, which is effectively unfiltered.

**Fix:**
1. In `SearchField`, stop merging into a single `"default"` bucket. Run the
   fragments *per field type* and collect a **set of primary keys per field type**.
   Within a single field type, combine fragments appropriately (descriptive PK
   sets are unioned; quantitative/categorical/identifier fragments AND together).
2. Compute the **intersection** of the per-field-type PK sets → "exact match".
3. If the intersection is empty, fall back to the **union** → "related match",
   and record which path was taken (see A2).
4. Return structured results (`{field_type: set_of_pks}`) from `get_results()`
   rather than a single concatenated list.

### A2 — `NLQSResult` cannot express "exact" vs "related"

**Where:** `nlqs/nlqs.py` `@dataclass NLQSResult` (only `records`, `uris`,
`is_input_irrelevant`).

**What's wrong:** README §5 requires telling the user whether the data is an exact
match or merely related. The result object has no field for this, so the
fallback semantics from A1 cannot be surfaced.

**Fix:** Add `is_exact_match: bool = True` (and optionally a human-readable
`message`) to `NLQSResult`, and set it based on whether the intersection or the
union path produced the rows.

### A3 — Primary key is assumed to be `row[0]`

**Where:** `nlqs/nlqs.py` (`# Assuming first column is primary key`,
`all_primary_keys.append(row[0])`) combined with
`nlqs/query_construction.py` `construct_final_search_query()` which emits
`SELECT * FROM ...`.

**What's wrong:** `SELECT *` returns columns in physical table order. The code
assumes the first column is the primary key. This only works by luck when the PK
happens to be declared first (as in the test fixture's `id SERIAL PRIMARY KEY`).
On any table whose PK is not the first column, the wrong column is harvested as
the "primary key" and the final `WHERE pk IN (...)` query is garbage.

**Fix:** Thread the already-known `primary_key` (from
`driver.get_primary_key(...)`) into the search layer and have the field queries
`SELECT <primary_key> FROM ...` explicitly, or build a column-name→index map and
read the PK by name instead of position.

### A4 — `summarize()` processes `quantitative_data` twice

**Where:** `nlqs/summarization.py` — there are two loops over
`summarized_input_dict.get("quantitative_data", {})` (one near the top that
unconditionally drops existing columns into `numerical_data`, and a second,
type-checked, loop further down).

**What's wrong:** The same dictionary is consumed twice with different rules. The
first pass adds any existing column straight to `numerical_data` without checking
its real type; the second pass re-derives the type. This is redundant, can
double-insert, and can place a column in the wrong bucket depending on which pass
"wins".

**Fix:** Delete the first quantitative loop and keep a single, type-checked pass.
Mirror the structure of the `qualitative_data` loop so quantitative and
qualitative handling are symmetric and each input dict is consumed exactly once.

### A5 — Validated user-requested columns are computed and then thrown away

**Where:** `nlqs/summarization.py` — `get_validated_user_requested_columns(...)`
is called but its return value is not assigned; the `SummarizedInput` is then
built from the **raw** `summarized_input_dict.get("user_requested_columns", [])`.

**What's wrong:** The validation/closest-column-substitution logic is dead. Invalid
or fuzzy column names from the LLM are never corrected, so the downstream
`len(user_requested_columns) > 0` gate and the column existence assumptions are
based on unvalidated data.

**Fix:** Assign the return value
(`validated = get_validated_user_requested_columns(...)`) and use it when
constructing `SummarizedInput`.

### A6 — Identifier columns are never shown to the summarizer

**Where:** `nlqs/nlqs.py` calls `summarize(...)` without passing
`column_descriptions_dict["identifier_columns"]`; `summarize()`'s signature has
no `identifier_columns` parameter, even though `ColumnType.IDENTIFIER` and
`identifier_data` exist throughout the pipeline.

**What's wrong:** The LLM prompt lists numerical/categorical/descriptive columns
but never identifier columns, so the model has no basis to populate identifier
constraints. Identifier handling is therefore effectively unreachable from the
summary step.

**Fix:** Add an `identifier_columns` parameter to `summarize()`, include it in the
classification prompt, and pass it from `execute_nlqs_query_workflow`.

### A7 — Vector store db/table names are hard-coded to defaults

**Where:** `nlqs/summarization.py` and `nlqs/query_construction.py` use
`DEFAULT_DB_NAME` / `DEFAULT_TABLE_NAME` (`"default_db"` / `"default_table"`) for
all vector lookups, while the SQL side uses the real configured table name
(e.g. `new_dataset` in the demo).

**What's wrong:** The vector metadata filter (`db_name`/`table_name`) is decoupled
from the actual SQL table being queried. If the vector store was populated with
anything other than the literal defaults, the descriptive/closest-column searches
silently return nothing.

**Fix:** Carry the real database/table identifiers (from the connection config)
through `summarize()` and the `construct_*` helpers instead of the hard-coded
defaults, and ensure population uses the same identifiers.

---

## B. Robustness / crash gaps

### B1 — Empty-summary retry can crash legitimate short/phatic inputs

**Where:** `nlqs/nlqs.py` — the `while not summarized_input.summary and count < 5`
loop that ends with `raise ValueError("Unable to summarize the data.")`.

**What's wrong:** This retry runs **before** the intent checks
(`sql_injection` / `phatic_communication`). If the intent LLM returns an empty
`summary` (common for greetings or one-word inputs), the workflow burns five LLM
calls and then raises, never reaching the phatic-communication early return that
was meant to handle exactly these inputs.

**Fix:** Evaluate intent first and short-circuit phatic/irrelevant inputs before
retrying. Treat an empty summary as a soft condition (return an empty/irrelevant
`NLQSResult`) rather than a hard `ValueError`.

### B2 — Descriptive `IN (...)` fragments don't quote non-numeric keys

**Where:** `nlqs/query_construction.py` `construct_descriptive_search_query_fragments()`
builds `f"{pk_column_name} IN ({values_list})"` with raw, unquoted values.

**What's wrong:** This only produces valid SQL when the lookup key is numeric. If a
table's primary/lookup key is a string (e.g. a `PackageID`), the generated SQL is
syntactically invalid and the query errors out.

**Fix:** Detect whether each lookup value is numeric (as
`construct_identifier_search_query_fragments` already does) and wrap non-numeric
values in single quotes.

### B3 — `VectorDBDriver.get_closest_data_from_description` is broken

**Where:** `nlqs/vectordb_driver.py` `get_closest_data_from_description()`.

**What's wrong:** It guards with `if not isinstance(lookup_key, int): continue`, but
`lookup_key` is the lookup **column name** (a string), so the guard always
`continue`s and the method returns `[]`. It also iterates `results["documents"]`
while indexing `results["metadatas"][index][0]`, which does not match Chroma's
nested result shape. The Neon backend's equivalent does not share this bug, so the
two backends disagree.

**Fix:** Correct the type check (validate `lookup_value`, not the key name) and the
result indexing to match Chroma's `QueryResult` structure — or remove the method
if it is confirmed unused on the live path and rely on `qualitative_dataset_search`.

### B4 — Library configures global logging and prints data at import/run time

**Where:** `nlqs/nlqs.py` calls `logging.basicConfig(level=logging.DEBUG)` at
import, and the workflow contains numerous `print(...)` / `print(json.dumps(...))`
debug statements (banner of `*`, full `column_descriptions_dict`, summaries,
search results, the final `result`, etc.). `summarization.py` and `query.py` also
`print` raw LLM output.

**What's wrong:** A library should not reconfigure root logging for the whole host
application, and dumping full records/LLM output to stdout is noisy and leaks
data. It also masks real logging levels.

**Fix:** Remove `basicConfig` from the module (let the application configure it),
delete the `print` debugging, and route any needed diagnostics through
`logger.debug(...)`.

---

## C. Vector-backend & configuration gaps

### C1 — `NeonDBConfig.embedding_dim` defaults to 1536 while NLQS uses 384-dim BGE

**Where:** `nlqs/neondb_driver.py` (`embedding_dim: int = 1536`) vs `nlqs/nlqs.py`
which builds embeddings via `get_default_embedding_function(use_local=True)`
(BAAI `bge-small-en-v1.5`, 384 dims). The test fixture works only because it
explicitly passes `embedding_dim=384`.

**What's wrong:** Anyone using the default `NeonDBConfig` with NLQS gets a
pgvector dimension mismatch on insert/query (the `populate_*` methods even assert
the dimension).

**Fix:** Default `embedding_dim` to 384 to match the default embedding function, or
derive it from the embedding function at construction time and validate
consistency.

### C2 — LLM/embedding backends are hard-coded

**Where:** `nlqs/nlqs.py` — `get_default_llm(use_azure=True)` and
`get_default_embedding_function(use_local=True)` are hard-wired.

**What's wrong:** There is no way to choose the LLM/embedding backend per
deployment; the algorithm silently requires Azure credentials for the LLM and a
local BGE model for embeddings simultaneously, with no configuration surface.

**Fix:** Make the LLM and embedding backends configurable (constructor args or the
connection/vector config), defaulting to the current behavior, and document the
required environment variables in one place.

---

## D. Dead / duplicated code

### D1 — A second, divergent `summarize`/`SummarizedInput` in `nlqs/query.py`

**Where:** `nlqs/query.py` defines its own `SummarizedInput` (no `identifier_data`)
and a cannabis-specific `summarize()` that is not used by the live workflow
(`nlqs/nlqs.py` imports from `nlqs/summarization.py`).

**What's wrong:** Two classes named `SummarizedInput` with different shapes invite
import mistakes and confusion about which is authoritative.

**Fix:** Remove `nlqs/query.py` (or fold any still-wanted domain heuristics into
`summarization.py`) so there is a single source of truth.

### D2 — Unused imports / leftover scaffolding

**Where:** e.g. `from unittest.mock import DEFAULT` in `query_construction.py`,
`from sqlalchemy import column` and the large commented-out `qualitaive_search`
block in `summarization.py`, unused imports in `vectordb_driver.py`
(`collections`, `Option`, `DataFrame` paths, etc.).

**What's wrong:** Noise; some imports are accidental and misleading.

**Fix:** Prune unused imports and remove the dead commented blocks.

---

## E. Test-suite gaps

### E1 — `test_nlqs_basic.py` constructs `ChromaDBConfig` with non-existent fields

**Where:** `nlqs/tests/test_nlqs_basic.py` uses
`ChromaDBConfig(host=..., port=..., chroma_db_impl=..., persist_directory=...)`,
but the dataclass only accepts `persist_path` (plus host/port/collection names).
It also patches `nlqs.nlqs.OpenAIEmbeddings`, which the module no longer imports.

**Fix:** Update the tests to the current `ChromaDBConfig(persist_path=...)` API and
patch the embedding/LLM entry points that `nlqs.py` actually uses
(`get_default_llm`, `get_default_embedding_function`).

### E2 — E2E test does not populate the vector collections

**Where:** `nlqs/tests/e2e/test_nlqs.py::test_nlsq_api` depends on
`chroma_config, pg_config, setup_postgres_database` but **not** the
`vectordb_driver` fixture that loads the TSVs into Chroma. `NLQS.__init__` raises
if the collections don't exist.

**Fix:** Add the population fixture (or an equivalent setup) as a dependency of the
e2e test so the collections exist before `NLQS` is constructed.

### E3 — Chroma `vectordb_driver` fixture mixes a 1536-dim embedder with 384-dim data

**Where:** `nlqs/tests/conftest.py` — the `vectordb_driver` fixture populates the
384-dim BGE TSVs but builds the driver with the Azure (`embedding_function`,
1536-dim) fixture. Any *query* through that fixture mismatches dimensions.

**Fix:** Build the Chroma `vectordb_driver` fixture with `bge_embedding_function`
so the query-time embedder matches the stored 384-dim vectors.

### E4 — `nlqs/tests/test_nlqs.py` is empty

**Where:** `nlqs/tests/test_nlqs.py` (0 bytes).

**Fix:** Either remove the placeholder or add real unit tests for
`execute_nlqs_query_workflow` (mocking the LLM/vector layer), especially once A1–A3
change the result contract.

---

## Suggested fix sequencing

A dependency-aware order that keeps the suite green at each step:

1. **B-series robustness fixes** (B1–B4) and **D-series cleanup** — low-risk,
   make the pipeline observable and stop spurious crashes/noise.
2. **A3 (explicit primary key)** — prerequisite for trustworthy PK harvesting.
3. **A1 + A2 (intersection/union + exact-vs-related result)** — the core
   algorithm correction; this changes the `SearchField` contract and `NLQSResult`.
4. **A4–A7 (summarization correctness, identifier wiring, real db/table names)**.
5. **C-series (config)** — make backends/dimensions configurable and consistent.
6. **E-series (tests)** — update fixtures/tests to the corrected contracts and add
   coverage for the intersection/union behavior.

## Risk notes

- A1/A2 change the public result contract (`NLQSResult`) and the `SearchField`
  return shape; downstream consumers (e.g. `examples/nlqs_demo/...`) must be
  updated together.
- Several fixes (A6, A7) require re-populating the vector store with the correct
  `db_name`/`table_name` to be observable end-to-end.
- Full end-to-end verification needs the heavy dependency stack (torch /
  sentence-transformers / chromadb), a Postgres instance, and Azure OpenAI
  credentials; these are not currently provisioned in this environment.
