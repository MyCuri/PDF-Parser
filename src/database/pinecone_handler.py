from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI


import config
from src.database.schema import DocumentType


class PineconeHandler:
    def __init__(self):
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=config.OPENAI_API_KEY
        )
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            api_key=config.OPENAI_API_KEY,
        )

        self._initialize_index()
        self._setup_vectorstore()

    def _initialize_index(self):
        """Initialize Pinecone index if it doesn't exist."""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = self.pc.Index(self.index_name)

    def _setup_vectorstore(self):
        """Set up Langchain Pinecone vector_store and retriever."""
        self.vector_store = PineconeVectorStore(
            index=self.index, embedding=self.embeddings, text_key="page_content"
        )

    async def add_documents(self, documents):
        """Add documents to the vector_store."""
        await self.vector_store.aadd_documents(documents)

    def get_retriever(self, document_type: DocumentType, pdf_file_names: List[str]):
        """Query the vector_store"""

        num_vectors = 3
        if document_type == DocumentType.SUMMARY:
            # Increase the number of vectors to retrieve for summary documents
            num_vectors = 10

        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(
                search_kwargs={
                    "k": num_vectors,
                    "filter": {
                        "document_type": document_type.value,
                        "filename": {"$in": pdf_file_names},
                    },
                }
            ),
            llm=self.llm,
        )

        return self.multi_query_retriever

    def delete_vectors_by_source(self, filename: str):
        """Delete all vectors from the index that match the given source."""

        for ids in self.index.list(prefix=filename):
            # Delete the vectors
            self.index.delete(ids=ids)
