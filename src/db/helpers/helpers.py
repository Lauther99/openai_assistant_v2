import sys

sys.path.append("C:\\Users\\lauth\\OneDrive\\Desktop\\open_ai_assistant_v2")
from langchain_core.documents import Document
from langchain_community.vectorstores.chroma import Chroma


def find_db_examples(
    query: str, collection: Chroma, k: int = 3, score_threshold: float = 0.7
) -> list[tuple[Document, float]]:
    examples = collection.similarity_search_with_relevance_scores(
        query=query, k=k, score_threshold=score_threshold
    )
    return examples


