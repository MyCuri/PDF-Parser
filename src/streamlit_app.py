import os
import streamlit as st
import asyncio

import config
from src.database.mongodb_handler import MongoDBHandler
from src.database.pinecone_handler import PineconeHandler
from src.pdf_processor import PDFProcessor
from src.chain.qa_chain import QAChain
from src.database.schema import DocumentType


class StreamlitApp:
    def __init__(self):
        # Initialize handlers
        self.pinecone_handler = PineconeHandler()
        self.mongo_handler = MongoDBHandler()
        self.pdf_processor = PDFProcessor()

        # Get a list of pdf filenames
        self.pdf_filenames = self.mongo_handler.get_pdf_filenames()

        # Get a list of system prompt names
        self.prompts = self.mongo_handler.get_prompts()
        self.prompt_names = [p for p in self.prompts.keys()]

    def setup_deletion(self):
        """Setup the Streamlit sidebar with all necessary inputs."""

        item_type_to_delete = st.radio(
            "Item",
            ["PDF", "Prompt"],
            index=0,
            label_visibility="hidden",
            horizontal=True,
        )

        if item_type_to_delete == "PDF":
            selected_name = st.selectbox(
                "Select a PDF", options=self.pdf_filenames, index=None
            )
        elif item_type_to_delete == "Prompt":
            selected_name = st.selectbox(
                "Select a Prompt", options=self.prompt_names, index=None
            )

        if st.button("Delete"):
            self.delete_item(selected_name, item_type_to_delete)

    def setup_prompt_selection(self):

        st.subheader("Load Prompt")
        selected_prompt_name = st.selectbox(
            "Select a Prompt", options=self.prompt_names, label_visibility="collapsed"
        )

        self.system_prompt = st.text_area(
            "Prompt",
            placeholder="LLM's system prompt",
            height=200,
            label_visibility="collapsed",
            value=self.prompts.get(selected_prompt_name),
        )

        st.subheader("Save Prompt")
        prompt_name = st.text_input(
            "Prompt Name",
            placeholder="Enter a name for the Prompt",
            label_visibility="collapsed",
        )
        save_prompt = st.button("Save Prompt")

        if self.system_prompt and prompt_name and save_prompt:
            pass
            self.mongo_handler.insert_prompt(
                name=prompt_name, prompt=self.system_prompt
            )
            st.success("Prompt Saved")

    def setup_pdf_query(self):

        col1, col2 = st.columns([2, 1])
        selected_pdf_filename = col1.multiselect(
            "Select PDFs", options=self.pdf_filenames, default=[self.pdf_filenames[0]]
        )
        selected_document_type = col2.selectbox(
            "Document Type", options=DocumentType, index=0
        )

        user_query = st.text_area(
            "Question", placeholder="What are the types of conflict?"
        )

        if st.button("Query") and user_query and self.system_prompt:
            with st.spinner("Processing your query..."):
                # Get a pinecone retriever
                pc_retriever = self.pinecone_handler.get_retriever(
                    document_type=selected_document_type,
                    pdf_file_names=selected_pdf_filename,
                )
                qa_chain = QAChain(
                    system_prompt=self.system_prompt, retriever=pc_retriever
                )
                response = qa_chain.run(user_query)
                st.write(response)

    def setup_pdf_indexing(self):

        uploaded_files = st.file_uploader(
            "Upload PDF files", type="pdf", accept_multiple_files=True
        )

        col1, col2 = st.columns(2)
        chunk_size = col1.number_input("Chunk Size", min_value=100, value=4000)
        chunk_overlap = col2.number_input("Chunk Overlap", min_value=0, value=200)
        summary_prompt = st.text_area(
            "Summarization Prompt",
            value="Summarize all meaningful context from the text",
        )
        if uploaded_files and st.button("Run"):
            # Process PDFs
            self.process_pdfs(uploaded_files, summary_prompt, chunk_size, chunk_overlap)

    def run(self):
        """Main application logic."""

        st.title("PDF Parser")
        query_tab, index_tab = st.tabs(["Query", "Index"])

        with st.sidebar:
            with st.expander("System Prompt", expanded=True):
                self.setup_prompt_selection()
            with st.expander("Delete PDF and Prompt", expanded=True):
                self.setup_deletion()

        with query_tab:
            if self.pdf_filenames:
                self.setup_pdf_query()
            else:
                st.info("No PDFs indexed. Please upload and index PDFs first.")

        with index_tab:
            self.setup_pdf_indexing()

    def process_pdfs(
        self, uploaded_files, summary_prompt, hard_max_chunk_size, overlap
    ):
        """Process uploaded PDF files."""
        self.pdf_processor.reset_temp_folder()

        for uploaded_file in uploaded_files:
            # Construct the pdf file path
            pdf_path = os.path.join(config.TEMP_FOLDER, uploaded_file.name)

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Extracting elements from {uploaded_file.name}..."):

                # Process PDF
                pdf_documents = self.pdf_processor.convert_to_documents(
                    pdf_path=pdf_path,
                    pdf_name=uploaded_file.name,
                    summary_prompt=summary_prompt,
                    hard_max_chunk_size=hard_max_chunk_size,
                    overlap=overlap,
                )

                # Create async function to add documents
                async def add_docs():
                    await self.pinecone_handler.add_documents(pdf_documents)

                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(add_docs())
                finally:
                    loop.close()

                # Store the pdf name in in MongoDB
                self.mongo_handler.insert_pdf(uploaded_file.name)

                self.pdf_processor.reset_temp_folder()

        st.success("PDF processing and summarization completed!")

    def delete_item(self, item_name: str, item_type: str):
        """Delete a PDF or Prompt from the system."""
        if not item_name:
            st.error("Please select an item to delete")
            return

        if item_type == "PDF":
            # Delete vectors from Pinecone
            # self.pinecone_handler.delete_vectors_by_source(item_name)
            # Delete from MongoDB
            self.mongo_handler.delete_pdf_filename(item_name)
            # Refresh the PDF filenames list
            self.pdf_filenames = self.mongo_handler.get_pdf_filenames()
        else:  # Prompt
            # Delete from MongoDB
            self.mongo_handler.delete_prompt(item_name)
            # Refresh the prompts
            self.prompts = self.mongo_handler.get_prompts()
            self.prompt_names = [p for p in self.prompts.keys()]

        st.success(f"'{item_name}' deleted successfully")
