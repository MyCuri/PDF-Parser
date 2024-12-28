import os
import shutil
import uuid
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from src.database.schema import DocumentType


class PDFProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            api_key=config.OPENAI_API_KEY,
        )

    def convert_to_documents(
        self,
        pdf_path: str,
        pdf_name: str,
        summary_prompt: str,
        hard_max_chunk_size: int = 4000,
        overlap: int = 200,
    ):
        """Chunks a pdf, extract the various elements, and generates summarizes for each extract"""

        loader = PyPDFLoader(pdf_path)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=hard_max_chunk_size,
            chunk_overlap=overlap,
        )
        chunked_documents = text_splitter.split_documents(document)

        summary_documents = []
        raw_documents = []

        for doc in chunked_documents:
            # Create the raw document
            raw_doc = doc.model_copy(deep=True)
            # Create new metadata and id
            raw_doc.metadata["document_type"] = DocumentType.RAW_TEXT.value
            raw_doc.metadata["filename"] = pdf_name
            raw_doc.id = f"{pdf_name}_{DocumentType.RAW_TEXT.value}_{str(uuid.uuid4())}"
            raw_documents.append(raw_doc)

            # Create a summary document
            summary_doc = Document(
                page_content=self._generate_summaries(summary_prompt, doc.page_content),
                metadata=doc.metadata.copy(),
            )
            # Create new metadata and id
            summary_doc.metadata["document_type"] = DocumentType.SUMMARY.value
            summary_doc.metadata["filename"] = pdf_name
            summary_doc.id = (
                f"{pdf_name}_{DocumentType.SUMMARY.value}_{str(uuid.uuid4())}"
            )
            summary_documents.append(summary_doc)

        return raw_documents + summary_documents

    def _generate_summaries(self, summary_prompt: str, text):
        """Generate summaries for the input text"""

        prompt = ChatPromptTemplate.from_template(
            """{summary_prompt}
            
            {text}"""
        )

        # Text summarization chain
        chain = prompt | self.llm | StrOutputParser()
        # Return the summary
        return chain.invoke({"summary_prompt": summary_prompt, "text": text})

    def reset_temp_folder(self):
        """Remove and recreate the temporary folder"""
        if os.path.exists(config.TEMP_FOLDER):
            shutil.rmtree(config.TEMP_FOLDER)
        os.makedirs(config.TEMP_FOLDER)
