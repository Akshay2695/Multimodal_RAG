import os
import glob
import uuid
import shutil
import logging
import base64
from typing import List
from unstructured.partition.pdf import partition_pdf
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import List
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def process_image_file(file_path: str) -> str:
    """Process a single image file and return a summary."""
    logger.info(f"Processing image file: {file_path}")
    try:
        # Load image and convert to base64
        with open(file_path, "rb") as img_file:
            image_data = img_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Summarize the image
        summary = summarize_images([image_base64])
        return summary[0] if summary else ""
    except Exception as e:
        logger.error(f"Error processing image file {file_path}: {e}")
        return ""

def process_pdf_document(file_path: str) -> (List[Document], List[Document]):
    """Process a single PDF file and extract tables and texts."""
    try:
        logger.info(f"Processing PDF file: {file_path}")
        
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )
        
        logger.info(f"Total chunks extracted: {len(chunks)}")
        
        tables = []
        texts = []
        
        for chunk in chunks:
            chunk_type = str(type(chunk))
            logger.info(f"Processing chunk type: {chunk_type}")
            
            is_table = False
            
            if "Table" in chunk_type:
                logger.info("Found table via type check")
                is_table = True
            elif hasattr(chunk, 'metadata'):
                text_as_html = getattr(chunk.metadata, 'text_as_html', None)
                if text_as_html and '<table' in text_as_html:
                    logger.info("Found table via HTML content")
                    is_table = True
                
                element_type = getattr(chunk.metadata, 'element_type', '')
                if element_type and 'table' in str(element_type).lower():
                    logger.info("Found table via element type")
                    is_table = True
            
            if is_table:
                tables.append(chunk)
            elif "CompositeElement" in chunk_type:
                texts.append(chunk)

        logger.info(f"Extracted {len(tables)} tables and {len(texts)} text chunks")
        return tables, texts
    
    except Exception as e:
        logger.error(f"Error processing PDF document {file_path}: {e}")
        logger.exception("Full exception details:")
        return [], []

def get_images_base64(chunks):
    """Extract base64 images from chunk elements."""
    images_b64 = []
    try:
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                    chunk_els = chunk.metadata.orig_elements
                    for el in chunk_els:
                        if "Image" in str(type(el)) and hasattr(el, 'metadata'):
                            image_base64 = getattr(el.metadata, 'image_base64', None)
                            if image_base64:
                                logger.info("Found and extracted base64 image")
                                images_b64.append(image_base64)
        
        logger.info(f"Extracted {len(images_b64)} images from chunks")
        return images_b64
    except Exception as e:
        logger.error(f"Error extracting images: {e}")
        logger.exception("Full exception details:")
        return []

def summarize_texts_and_tables(texts: List[str], tables: List[str]) -> (List[str], List[str]):
    """Summarize text and table chunks."""
    if not texts and not tables:
        logger.warning("No content to summarize")
        return [], []
        
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.
    
    Respond only with the summary, no additional comment.
    Just give the summary as it is.
    
    Table or text chunk: {element}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    
    try:
        text_contents = [chunk.text if hasattr(chunk, 'text') else str(chunk) for chunk in texts]
        table_contents = [table.metadata.text_as_html if hasattr(table.metadata, 'text_as_html') else str(table) for table in tables]
        
        text_summaries = summarize_chain.batch(text_contents) if text_contents else []
        table_summaries = summarize_chain.batch(table_contents) if table_contents else []
        
        return text_summaries, table_summaries
    except Exception as e:
        logger.error(f"Error summarizing content: {e}")
        return [], []

def summarize_images(images: List[str]) -> List[str]:
    """Summarize images based on their base64 data.
    
    Args:
        images: List of base64-encoded image strings
        
    Returns:
        List of image descriptions/summaries
    """
    if not images:
        logger.warning("No images to summarize")
        return []

    try:
        model = ChatOpenAI(model="gpt-4o-mini")
        
        # Process each image with proper message format
        image_summaries = []
        for image in images:
            try:
                # Create message with image content directly
                message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Describe the image in detail. For context, the image is part of a research paper explaining the transformers architecture. Be specific about graphs, such as bar plots."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"
                            }
                        }
                    ]
                )
                
                # Invoke model directly with the message
                response = model.invoke([message])
                summary = response.content
                image_summaries.append(summary)
                logger.info("Successfully summarized an image")
                
            except Exception as e:
                logger.error(f"Error processing individual image: {e}")
                continue
        
        logger.info(f"Successfully summarized {len(image_summaries)} images")
        return image_summaries
        
    except Exception as e:
        logger.error(f"Error summarizing images: {e}")
        logger.exception("Full exception details:")
        return []

def main():
    # Configuration variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY', "./db")
    source_directory = os.environ.get('SOURCE_DIRECTORY', "./data")

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    logger.info(f"Loading documents from {source_directory}")
    
    # Correctly find PDF and image files
    pdf_files = glob.glob(os.path.join(source_directory, '**/*.pdf'), recursive=True)
    image_files = glob.glob(os.path.join(source_directory, '**/*.[pjPJ][npNP]*[gG]'), recursive=True)

    all_texts = []
    all_tables = []
    
    # Process PDF files
    for pdf_file in pdf_files:
        tables, texts = process_pdf_document(pdf_file)
        if tables or texts:
            all_tables.extend(tables)
            all_texts.extend(texts)

    # Process Image files
    all_image_summaries = []
    for image_file in image_files:
        summary = process_image_file(image_file)
        if summary:
            all_image_summaries.append(summary)

    if not all_texts and not all_tables and not all_image_summaries:
        logger.error("No content extracted from PDFs or images")
        return

    # Get summaries
    text_summaries, table_summaries = summarize_texts_and_tables(all_texts, all_tables)

    try:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            collection_name="multi_modal_rag",
            embedding_function=OpenAIEmbeddings()
        )
        
        # Add text summaries
        if text_summaries:
            doc_ids = [str(uuid.uuid4()) for _ in text_summaries]
            summary_texts = [Document(page_content=summary, metadata={"doc_id": doc_id}) 
                           for summary, doc_id in zip(text_summaries, doc_ids)]
            vectorstore.add_documents(summary_texts)

        # Add table summaries
        if table_summaries:
            table_ids = [str(uuid.uuid4()) for _ in table_summaries]
            summary_tables = [Document(page_content=summary, metadata={"doc_id": table_id})
                            for summary, table_id in zip(table_summaries, table_ids)]
            vectorstore.add_documents(summary_tables)

        # Add image summaries
        if all_image_summaries:
            img_ids = [str(uuid.uuid4()) for _ in all_image_summaries]
            summary_images = [Document(page_content=summary, metadata={"doc_id": img_id})
                            for summary, img_id in zip(all_image_summaries, img_ids)]
            vectorstore.add_documents(summary_images)

        vectorstore.persist()
        logger.info("Ingestion complete!")
        
    except Exception as e:
        logger.error(f"Error during ingestion process: {e}")
        logger.exception("Full exception details:")

if __name__ == "__main__":
    main()

