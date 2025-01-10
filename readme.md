# Multimodal Q&A Chatbot

This project is a chatbot designed to answer questions about documents using a multimodal approach. It processes both text and images, allowing users to chat with their documents (Complex PDFs, Images). The chatbot leverages Langchain for document processing, OpenAI for natural language understanding, and Langfuse for observability and analytics.

## Features

- **Multimodal Input**: Accepts both text and image inputs.
- **Contextual Understanding**: Uses embeddings to retrieve relevant information from documents.
- **User Feedback Integration**: Collects user feedback to improve response quality. [Yet to implement]
- **Analytics**: Integrates with Langfuse for detailed observability of user interactions.

## Technologies Used

- Python
- Streamlit
- Langchain
- OpenAI API
- Langfuse
- Unstructured (for PDF processing)

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.6 or higher
- pip (Python package installer)

## Setup Instructions

1. **Clone the Repository**


2. **Install Required Packages**
Create a virtual environment (optional but recommended) and install the required packages:


3. **Set Up Environment Variables**
Create a `.env` file in the root directory of the project with the following content:

    EMBEDDINGS_MODEL_NAME=""
    PERSIST_DIRECTORY=""
    TARGET_SOURCE_CHUNKS=""
    SOURCE_DIRECTORY="db"
    OPENAI_API_KEY=""
    SOURCE_DIRECTORY=""
    GROQ_API_KEY=""
    LANGFUSE_SECRET_KEY=""
    NEXT_PUBLIC_LANGFUSE_PUBLIC_KEY=""

4. **Upload Data**
Place your PDF and image files in the `data` folder. The chatbot currently supports .pdf and image files.

5. **Run Data Ingestion**
Execute the `ingest.py` script to parse the PDFs and images, summarize them, and store the embeddings in Chroma:

6. **Run the Application**
Start the Streamlit application:

    'streamlit run app.py'


7. **Access the Chatbot**
Open your web browser and navigate to `http://localhost:8501` to interact with the chatbot.

## Usage

1. **Ask Questions**: Type your questions about research papers in the chat input box.
2. **Upload Images**: You can upload images related to your questions for contextual understanding. [Yet to implement]
3. **Provide Feedback**: After receiving responses, you can provide feedback on their helpfulness. [Yet to implement]


## Acknowledgments

- [Langchain](https://langchain.com/)
- [OpenAI](https://openai.com/)
- [Langfuse](https://langfuse.com/)
- [Unstructured](https://unstructured.io/)

---


