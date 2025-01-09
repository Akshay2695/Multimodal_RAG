import streamlit as st
import os
import io
import base64
from PIL import Image
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler(
    public_key=os.getenv("NEXT_PUBLIC_LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com"
)


def initialize_chain():
    vectorstore = Chroma(
        persist_directory=os.environ.get('PERSIST_DIRECTORY', "./db"),
        collection_name="multi_modal_rag",
        embedding_function=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def convert_image_format(b64_string):
        try:
            # Decode base64 string to image
            image_data = base64.b64decode(b64_string)
            img = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Save as JPEG in memory
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            print(f"Image conversion error: {e}")
            return None

    def parse_docs(docs):
        b64_images = []
        texts = []
        for doc in docs:
            try:
                # Try to decode as base64
                base64.b64decode(doc.page_content)
                converted_image = convert_image_format(doc.page_content)
                if converted_image:
                    b64_images.append(converted_image)
            except:
                texts.append(doc)
        return {"images": b64_images, "texts": texts}

    def build_prompt(kwargs):
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]

        context_text = ""
        if docs_by_type["texts"]:
            context_text = "\n".join(doc.page_content for doc in docs_by_type["texts"])

        prompt_template = f"""
        Answer based on this context (including text, tables, and images below).
        Context: {context_text}
        Question: {user_question}
        """

        prompt_content = [{"type": "text", "text": prompt_template}]

        for image in docs_by_type["images"]:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            })

        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough()
        }
        | RunnableLambda(build_prompt)
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )

    return chain

def main():
    st.title("Multimodal Q&A")
    
    if 'chain' not in st.session_state:
        st.session_state.chain = initialize_chain()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if question := st.chat_input("Ask about your papers"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = st.session_state.chain.invoke(question, config={"callbacks": [langfuse_handler]})
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

