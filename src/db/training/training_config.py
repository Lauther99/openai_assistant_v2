import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\open_ai_assistant_v2")
from src.settings.settings import Settings
from src.components.llms.llms import LLMs
from langchain_community.vectorstores.chroma import (
    Chroma as Langchain_Chroma_Collection,
)
from src.settings.env_config import Config
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb


class TrainingAssistant:
    def __init__(self, chromadb_directory: str = "../chroma_db") -> None:
        # Configuraciones básicas (keys, modelos, etc)
        self.settings = Settings()
        self.llms = LLMs(self.settings)

        # Configuracion cliente chromadb
        self.chromadb_directory = chromadb_directory
        self.chromadb_client = chromadb.PersistentClient(path=self.chromadb_directory)

        # Configuracion embeddings (chromadb/langchain)
        self.openai_native_llm = self.llms.get_native_llm(llm_type="native-openai")
        self.chromadb_embeddings_function = OpenAIEmbeddingFunction(
            api_key=self.settings.openai.api_key,
            model_name=self.settings.openai.embeddings_model,
        )
        self.langchain_embeddings_model = self.llms.get_embeddings_llm(
            llm_type=self.settings.openai.embeddings_model
        )

        # Configuracion nombre de colecciones
        self.collection_names = {
            "KEYWORDS_COLLECTION": Config.get_chromadb_config()["KEYWORDS_COLLECTION"],
            "CONTEXT_COLLECTION": Config.get_chromadb_config()["CONTEXT_COLLECTION"],
            "SUMMARY_COLLECTION": Config.get_chromadb_config()["SUMMARY_COLLECTION"],
            "CLASSIFIER_COLLECTION": Config.get_chromadb_config()[
                "CLASSIFIER_COLLECTION"
            ],
        }

    def get_embeddings(self, input_string: str):
        response = self.openai_native_llm.embeddings.create(
            input=input_string, model=self.settings.openai.embeddings_model
        )
        return response.data[0].embedding

    def sum_embeddings(self, list_embeddings: list[list[float]]):
        sums = [sum(x) for x in zip(*list_embeddings)]
        average = [total / len(list_embeddings) for total in sums]
        return average

    def _init_collections(self) -> None:
        """For creating collections if not exists, USE IT ON TRAINING NOTEBOOK!"""

        self.chromadb_client.get_or_create_collection(
            name=self.collection_names["KEYWORDS_COLLECTION"],
            embedding_function=self.chromadb_embeddings_function,
        )
        self.chromadb_client.get_or_create_collection(
            name=self.collection_names["CONTEXT_COLLECTION"],
            embedding_function=self.chromadb_embeddings_function,
        )
        self.chromadb_client.get_or_create_collection(
            name=self.collection_names["SUMMARY_COLLECTION"],
            embedding_function=self.chromadb_embeddings_function,
        )
        self.chromadb_client.get_or_create_collection(
            name=self.collection_names["CLASSIFIER_COLLECTION"],
            embedding_function=self.chromadb_embeddings_function,
        )

    def train_with_chroma(
        self,
        collection_name: str,
        embeddings: list[list[float]],
        ids,
        metadatas: list[dict] = [{"source": ""}],
    ):
        collection = self.chromadb_client.get_collection(
            name=collection_name, embedding_function=self.chromadb_embeddings_function
        )
        collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

    def train_collection_with_langchain(
        self,
        collection_name: str,
        texts: str,
        metadatas: list[dict] = [{"source": ""}],
    ):
        Langchain_Chroma_Collection.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.langchain_embeddings_model,
            persist_directory=self.chromadb_directory,
            collection_name=collection_name,
        )
