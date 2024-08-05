from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq

class KnowledgeGraph:

    def __init__(self, graphdb_uri: str, graphdb_username:str, graphdb_password: str, groq_api_key:str, model_name:str= "Gemma2-9b-It" ) -> None:
        
        self.graphdb_uri = graphdb_uri
        self.graphdb_username=graphdb_username
        self.graphdb_password=graphdb_password

        self.llm=ChatGroq(groq_api_key=groq_api_key,model_name=model_name)
        # Store all the conneciton info

        self._connect()



    def _connect(self):
        # Connect to the database
        self.graph_connection=Neo4jGraph(
            url=self.graphdb_uri,
            username=self.graphdb_username,
            password=self.graphdb_password,
        )
    
    def add_node(self):
        ##
        pass


    def add_relationship(self):
        pass

    
    def query(self):
        pass

