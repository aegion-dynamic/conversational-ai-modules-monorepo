# NL2GQL - Natural Language to GraphQL

A Python package that converts natural language queries into GraphQL queries using Large Language Models (LLMs).

## 🚀 Current State of Development

This package is in **early development** and provides basic functionality for natural language to GraphQL conversion.

### ✅ Implemented Features

- **LLM-powered Query Generation**: Uses Azure OpenAI to convert natural language to GraphQL
- **Schema Management**: Automatic schema fetching and loading from GraphQL endpoints
- **Command Line Interface**: CLI tool for quick conversions
- **Modern LangChain Integration**: Updated to use latest LangChain patterns (v0.2+)
- **Flexible Schema Support**: Works with any GraphQL schema

### 📁 Package Structure

```
nl2gql/
├── __init__.py                 # Package initialization
├── main.py                     # Main entry point
├── chains.py                   # LangChain chain implementation
├── cli.py                      # Command line interface
├── prompts.py                  # LLM prompt templates
├── schema_utils.py             # GraphQL schema utilities
├── README.md                   # This documentation
└── nl2gql-prompt/             # Legacy subdirectory (empty)
```

### 🔧 Core Components

#### 1. **chains.py**
- Implements the LLM chain using modern LangChain patterns
- Uses `prompt | llm | output_parser` instead of deprecated `LLMChain`
- Configured for Azure OpenAI integration

#### 2. **prompts.py**
- Contains the prompt template for GraphQL generation
- Simple template that includes schema and user query context

#### 3. **cli.py**
- Command-line interface for the package
- Supports schema file loading and natural language query processing
- Includes example integration with countries GraphQL API

#### 4. **schema_utils.py**
- GraphQL schema introspection and management
- Automatic schema fetching from endpoints
- Schema file creation and loading utilities

## 🛠️ Usage

### CLI Usage

```bash
python -m nl2gql.cli --schema path/to/schema.graphql --query "What are all the countries in Europe?"
```

### Programmatic Usage

```python
from nl2gql.chains import create_graphql_chain, run_chain
from nl2gql.schema_utils import load_schema

# Create the chain
chain = create_graphql_chain()

# Load your GraphQL schema
schema = load_schema("path/to/schema.graphql")

# Convert natural language to GraphQL
query = "Find all countries in Europe"
gql_query = run_chain(chain, schema, query)
print(gql_query)
```

## 📋 Requirements

- Python 3.8+
- LangChain (latest version)
- Azure OpenAI API access
- Environment variables:
  - `AZURE_OPENAI_DEPLOYMENT`
  - `AZURE_OPENAI_API_VERSION`

## 🚧 Known Limitations

1. **Limited Error Handling**: Basic error handling for failed API calls
2. **Schema Complexity**: Works best with simpler GraphQL schemas
3. **No Query Validation**: Generated queries are not validated against the schema
4. **Single LLM Provider**: Currently only supports Azure OpenAI

## 📝 Notes

- The package currently uses a hardcoded example with the countries GraphQL API
- Schema fetching is implemented but may need refinement
- The CLI interface provides a good starting point for testing and development
