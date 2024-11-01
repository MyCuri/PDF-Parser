import os
import streamlit as st
import pinecone
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import base64
import pandas as pd
from unstructured.partition.pdf import partition_pdf
import shutil
import io
import numpy as np
from PIL import Image
from dotenv import load_dotenv


load_dotenv()

# Set OpenAI and Pinecone API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

import os
from pinecone import Pinecone, ServerlessSpec
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# Initialize Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "pdfs"

# Check if the index exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Correct dimension for OpenAI embeddings
        metric="cosine",  # Can be 'euclidean', 'dotproduct', or 'cosine'
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Connect to the created or existing index
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)

# The storage layer for parent documents
store = InMemoryStore()
id_key = "doc_id"

# Create Pinecone vector store with Langchain integration
vectorstore = LangchainPinecone(
    index=index, embedding=embeddings, text_key="page_content"
)

# Create a retriever with the vectorstore and document store
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Initialize session state variables if not already initialized
if "pdf_processed" not in st.session_state:
    st.session_state["pdf_processed"] = False

if "pdf_elements" not in st.session_state:
    st.session_state["pdf_elements"] = {
        "texts": [],
        "tables": [],
        "images": [],
        "text_summaries": [],
        "table_summaries": [],
        "image_summaries": [],
    }


# Function to reset session variables
def reset_session():
    """Reset session state variables and delete previous figures folder."""
    # Remove the figures directory if it exists
    figures_dir = "figures"
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)

    # Reset session state variables
    st.session_state["pdf_processed"] = False
    st.session_state["pdf_elements"] = {
        "texts": [],
        "tables": [],
        "images": [],
        "text_summaries": [],
        "table_summaries": [],
        "image_summaries": [],
    }

    # Delete the uploaded PDF if it exists
    pdf_file = "uploaded_file.pdf"
    if os.path.exists(pdf_file):
        os.remove(pdf_file)


# Sidebar for settings and PDF upload
with st.sidebar:
    st.title("Settings")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type="pdf", key="pdf_uploader"
    )

# Main section of the app
st.title("PDF Extractor with Summarization and Question Answering")

# Check if a new file is uploaded
if uploaded_file:
    # If a new file is uploaded, reset the session and start fresh
    reset_session()

    process_pdf_button = st.button("Process PDF")

    if process_pdf_button:
        # Save uploaded PDF
        pdf_path = "uploaded_file.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract elements from the PDF (text, tables, images)
        with st.spinner("Extracting elements from the PDF..."):
            raw_pdf_elements = partition_pdf(
                filename=pdf_path,
                extract_images_in_pdf=True,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=4000,
                new_after_n_chars=3800,
                combine_text_under_n_chars=2000,
                image_output_dir_path="figures",
            )

        # Separate text, tables, and images
        texts, tables = [], []
        for element in raw_pdf_elements:
            if "Table" in str(type(element)):
                tables.append(str(element))
            elif "CompositeElement" in str(type(element)):
                texts.append(str(element))

        # Store extracted elements in session state
        st.session_state["pdf_elements"]["texts"] = texts
        st.session_state["pdf_elements"]["tables"] = tables
        st.session_state["pdf_processed"] = True

        st.write(f"Extracted {len(tables)} tables, {len(texts)} text blocks.")

        # Delete the uploaded PDF after processing
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            st.write("Uploaded PDF deleted successfully.")

# Summarization Chain for text, tables, and images
if st.session_state.get("pdf_processed", False):
    model = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_API_KEY)

    # Define summarization prompt for text and tables
    prompt_text = """You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summarize text and tables
    if (
        st.session_state["pdf_elements"]["texts"]
        and not st.session_state["pdf_elements"]["text_summaries"]
    ):
        st.write("Summarizing text and tables...")
        text_summaries = []
        table_summaries = []

        with st.spinner("Summarizing text..."):
            for text in st.session_state["pdf_elements"]["texts"]:
                summary = (prompt | model | StrOutputParser()).invoke({"element": text})
                text_summaries.append(summary)

        with st.spinner("Summarizing tables..."):
            for table in st.session_state["pdf_elements"]["tables"]:
                summary = (prompt | model | StrOutputParser()).invoke(
                    {"element": table}
                )
                table_summaries.append(summary)

        # Store summaries in session state
        st.session_state["pdf_elements"]["text_summaries"] = text_summaries
        st.session_state["pdf_elements"]["table_summaries"] = table_summaries

    st.write("Summaries completed!")


# Function to encode image to base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


from base64 import b64decode


def split_image_text_types(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        # Ensure that doc is a string
        if isinstance(doc, str):
            try:
                # Try decoding to check if it's a base64-encoded string (image)
                b64decode(doc)
                b64.append(doc)
            except Exception:
                # If it's not base64, assume it's regular text
                text.append(doc)
        else:
            # Non-string documents (like objects) should be ignored or handled
            text.append(str(doc))  # Optionally convert non-string objects to strings
    return {"images": b64, "texts": text}


def prompt_func(data):
    """Format the text and image content for the question"""
    format_texts = "\n".join(data["context"]["texts"])

    # Check if there are any images in the context
    if data["context"]["images"]:
        return [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"""Answer the question based only on the following context, which can include text, tables, and the below image:
                    Question: {data["question"]}

                    Text and tables:
                    {format_texts}
                    """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{data['context']['images'][0]}"
                        },
                    },
                ]
            )
        ]
    else:
        return [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"""Answer the question based only on the following context, which can include text and tables:
                    Question: {data["question"]}

                    Text and tables:
                    {format_texts}
                    """,
                    },
                ]
            )
        ]


# Define the model (updated to use gpt-4o-mini)
model = ChatOpenAI(
    temperature=0, model="gpt-4o", max_tokens=1024, api_key=OPENAI_API_KEY
)

# Update the chain to ensure "context" key is properly included in the output
chain = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(
        lambda x: {
            "context": {
                "texts": x["context"]["texts"],
                "images": x["context"]["images"],
            },
            "question": x["question"],
        }
    )
    | RunnableLambda(prompt_func)
    | model
    | StrOutputParser()
)

# Ask a question based on extracted elements
query = st.text_input("Ask a question about the PDF content:")
submit_question = st.button("Submit Question")

if submit_question and query:
    with st.spinner("Processing your question..."):
        # Invoke the chain and display the result
        response = chain.invoke(query)
    st.write("Answer:")
    st.write(response)


# Download CSV with text, table, and image summaries
def download_summary_csv():
    texts = st.session_state["pdf_elements"]["texts"]
    tables = st.session_state["pdf_elements"]["tables"]
    text_summaries = st.session_state["pdf_elements"]["text_summaries"]
    table_summaries = st.session_state["pdf_elements"]["table_summaries"]
    image_summaries = st.session_state["pdf_elements"]["image_summaries"]

    # Create a DataFrame with text, tables, and their summaries
    df_text = pd.DataFrame({"Text": texts, "Text Summary": text_summaries})
    df_tables = pd.DataFrame({"Table": tables, "Table Summary": table_summaries})
    df_images = pd.DataFrame({"Image Summary": image_summaries})

    # Save DataFrame to CSV
    csv_buffer = io.StringIO()
    df_text.to_csv(csv_buffer, index=False)
    df_tables.to_csv(csv_buffer, index=False)
    df_images.to_csv(csv_buffer, index=False)

    return csv_buffer.getvalue()


# Make sure pdf_summaries exists before trying to download the file
if (
    "text_summaries" in st.session_state["pdf_elements"]
    and st.session_state["pdf_elements"]["text_summaries"]
):
    st.download_button(
        label="Download Summaries CSV",
        data=download_summary_csv(),
        file_name="summaries.csv",
        mime="text/csv",
    )
