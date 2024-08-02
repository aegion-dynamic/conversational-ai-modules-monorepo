# Pipeline Structure for Building Neo4j Knowledge Graph from CSV File

## Overview

This document outlines the pipeline structure for creating a knowledge graph in Neo4j from a CSV file. The process involves extracting entities and relationships from the CSV file, loading the data into Neo4j, and visualizing the knowledge graph.


### 1. Data Preparation

- **Objective**: Prepare the CSV file for processing.
- **Input**: CSV file containing citation data with columns such as authors, published date, title, and abstract.
- **File Path**: `C:\Users\Satwik\Desktop\AegionDynamic\pubmed\pubmed_articles.csv`
- **Tools**: Pandas (for data manipulation and cleaning)

**Tasks**:
1. Load the CSV file using Pandas.
2. Clean the data (e.g., handle missing values, standardize formats).

### 2. Entity Extraction

- **Objective**: Extract entities (e.g., authors, titles) from the cleaned data.
- **Tools**: Custom scripts using Python (e.g., `pandas` for data manipulation)

**Tasks**:
1. Identify and extract entities from the cleaned data.
2. Create a list of unique entities and their attributes.

### 3. Relationship Extraction

- **Objective**: Extract relationships between entities.
- **Tools**: Custom scripts using Python (e.g., `pandas` for data manipulation)

**Tasks**:
1. Define relationships based on the data (e.g., authorship, publication).
2. Create a list of relationships with their source and target entities.

### 4. Data Transformation

- **Objective**: Transform the extracted entities and relationships into a format suitable for Neo4j.
- **Tools**: Python (e.g., `neo4j` Python driver)

**Tasks**:
1. Convert the entities and relationships into Cypher queries or CSV files compatible with Neo4j's import tools.

### 5. Data Loading into Neo4j

- **Objective**: Load the transformed data into Neo4j.
- **Tools**: Neo4j Desktop or Neo4j Aura (cloud-based), Neo4j import tools

**Tasks**:
1. Create a new Neo4j database or use an existing one.
2. Import the data using Neo4j's import functionality or Cypher queries.

### 6. Obtaining Neo4j Credentials

- **Objective**: Obtain credentials for a free Neo4j instance.
- **Tools**: Neo4j Aura (cloud-based)

**Tasks**:
1. Go to [Neo4j Aura](https://neo4j.com/cloud/aura/) and sign up for a free account.
2. Create a new instance of Neo4j Aura.
3. Obtain your connection credentials (e.g., URI, username, password) from the Neo4j Aura dashboard.
4. Use these credentials to connect to your Neo4j instance from your scripts or Neo4j Desktop.

### 7. Visualization and Validation

- **Objective**: Visualize and validate the knowledge graph in Neo4j.
- **Tools**: Neo4j Browser 

**Tasks**:
1. Use Neo4j Browser to explore and visualize the graph.
2. Validate the accuracy of the entities and relationships.


## References

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [Neo4j Aura](https://neo4j.com/cloud/aura/)

## Notes
- Ensure all tools and libraries are up-to-date.
- Follow best practices for data cleaning and transformation.
- Regularly backup your Neo4j database to avoid data loss.
