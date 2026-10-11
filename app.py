
import streamlit as st
from io import BytesIO
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# Configure the webpage
st.set_page_config(page_title="AI PDF Chatbot")

st.title("AI PDF Question Answering Chatbot")
st.write("Upload a PDF and ask questions about its content.")


# Load the embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Store the PDF database and conversation
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# Upload PDF
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# Process PDF
if uploaded_file is not None:
    if st.button("Process PDF"):
        try:
            with st.spinner("Reading and processing PDF..."):

                pdf = PdfReader(
                    BytesIO(uploaded_file.getvalue())
                )

                documents = []
                full_pdf_text = ""

                for page_number, page in enumerate(
                    pdf.pages, start=1
                ):
                    page_text = page.extract_text() or ""

                    if page_text.strip():
                        full_pdf_text += (
                            "\n\nPage "
                            + str(page_number)
                            + "\n"
                            + page_text
                        )

                        documents.append(
                            Document(
                                page_content=page_text,
                                metadata={"page": page_number}
                            )
                        )

                if not documents:
                    st.error(
                        "No readable text found in this PDF. "
                        "It may be a scanned document."
                    )
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=700,
                        chunk_overlap=100
                    )

                    chunks = splitter.split_documents(documents)

                    embeddings = load_embeddings()

                    vectorstore = FAISS.from_documents(
                        chunks,
                        embeddings
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.chat_history = []
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.full_pdf_text = full_pdf_text
                    st.session_state.pdf_chunks = chunks

                    st.success(
                        "PDF processed successfully. "
                        + str(len(chunks))
                        + " text chunks created."
                    )

                    # Show extracted PDF text
                    with st.expander("View all extracted PDF text"):
                        st.text(full_pdf_text)

        except Exception as e:
            st.error("PDF processing error: " + str(e))


# Ask questions after processing
if st.session_state.vectorstore is not None:

    st.subheader("Ask a question")

    question = st.chat_input(
        "Type your question about the PDF"
    )

    if question:
        try:
            with st.spinner("Finding the answer..."):

                # Retrieve five relevant chunks
                relevant_docs = (
                    st.session_state.vectorstore.similarity_search(
                        question,
                        k=5
                    )
                )

                context = "\n\n".join(
                    [
                        "Page "
                        + str(doc.metadata.get("page", "?"))
                        + ":\n"
                        + doc.page_content
                        for doc in relevant_docs
                    ]
                )

                # Show retrieved text for debugging
                st.session_state.last_context = context

                # Send the retrieved text to the local model
                llm = ChatOllama(
                    model="llama3.2:1b",
                    temperature=0
                )

                prompt = """
You are a question answering assistant.

Answer the user's question using the PDF text below.
Read all the supplied text carefully before answering.

If the answer is present, explain it in simple English.
If the answer is truly absent, say that it was not found.

PDF TEXT:
""" + context + """

QUESTION:
""" + question + """

ANSWER:
"""

                response = llm.invoke(prompt)
                answer = response.content

                pages = sorted(
                    {
                        doc.metadata.get("page", 0)
                        for doc in relevant_docs
                    },
                    key=str
                )

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "pages": pages,
                        "context": context
                    }
                )

        except Exception as e:
            st.error("Question answering error: " + str(e))


# Display conversation
for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

        st.caption(
            "Source pages: "
            + ", ".join(map(str, chat["pages"]))
        )

        with st.expander("View retrieved PDF text"):
            st.text(chat["context"])
