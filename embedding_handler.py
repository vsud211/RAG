import os
import hashlib
import pickle
import faiss  # FAISS for GPU support
import torch  # PyTorch for device checking
import pytesseract
from PIL import Image  # For image processing with OCR
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from logger_util import get_logger


# Get a logger for this module
logging = get_logger(__name__)

# Check CUDA availability and log the device
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.get_device_name(0))  # Should return the GPU name, e.g., NVIDIA GeForce RTX 3050 Ti
print(faiss.__version__)

pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Device Configuration: Use GPU if available, else fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Selected device: {device}")

# Initialize FAISS GPU resources if GPU is available
gpu_resources = None
if device == "cuda":
    try:
        gpu_resources = faiss.StandardGpuResources()
        logging.info("FAISS GPU resources initialized successfully.")
    except AttributeError as e:
        logging.error("Failed to initialize FAISS GPU resources: FAISS version does not support StandardGpuResources. Falling back to CPU.")
        gpu_resources = None
        device = "cpu"  # Fallback to CPU if GPU support fails
    except Exception as e:
        logging.error(f"Failed to initialize FAISS GPU resources: {e}")
        gpu_resources = None
        device = "cpu"

# Function to hash file content
def hash_file(file_path):
    hash_obj = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


# EmbeddingHandler class for managing embeddings and vector stores
class EmbeddingHandler:
    def __init__(self, cache_folder="/app/Repo_cache", cache_file_name="embeddings_cache.pkl", vector_store_folder="/app/Repo_cache/vector_store"):
        logging.debug("Initializing EmbeddingHandler...")

        self.cache_folder = cache_folder
        self.cache_file_name = cache_file_name
        self.cache_file = os.path.join(self.cache_folder, self.cache_file_name)
        self.vector_store_folder = vector_store_folder

        # Ensure cache and vector store folders exist
        os.makedirs(self.cache_folder, exist_ok=True)
        os.makedirs(self.vector_store_folder, exist_ok=True)

        # Load cached embeddings and initialize
        self.embeddings_cache = self._load_cache()
        self.embeddings = OllamaEmbeddings(model="llama2:latest")
        self.db = None

    # Load cached embeddings
    def _load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    logging.info("Loaded cached embeddings.")
                    embeddings_cache = pickle.load(f)

                    # Print each document hash and its embedding
                    for doc_hash, embedding in embeddings_cache.items():
                        print(f"Document Hash: {doc_hash}")
                        print(f"Embedding: {embedding}\n")

                    return embeddings_cache
                
            logging.info("No cache found. Starting fresh.")
            return {}
        except Exception as e:
            logging.error(f"Error loading cache: {e}")
            return {}


    # Save cache to disk
    def _save_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.embeddings_cache, f)
                logging.info("Embeddings cache saved.")
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

    # Create or load the vector store, and update it with new documents if available
    def initialize_vector_store(self, updated_documents):
        try:
            index_path = os.path.join(self.vector_store_folder, "index.faiss")  # Path to index file
            if os.path.exists(index_path):
                logging.info("Loading existing vector store.")
                self.db = FAISS.load_local(self.vector_store_folder, self.embeddings, allow_dangerous_deserialization=True)
                
                if updated_documents:
                    logging.info("Updating vector store with new documents.")
                    self.db.add_documents(updated_documents)
                    self.db.save_local(self.vector_store_folder)
                    logging.info("Vector store updated successfully.")
                else:
                    logging.info("No new documents to update vector store.")
            else:
                logging.info("No cached vector store found. Creating a new one.")
                if updated_documents:
                    logging.info("Creating vector store with new documents.")
                    self.db = FAISS.from_documents(updated_documents, self.embeddings)
                    self.db.save_local(self.vector_store_folder)
                    logging.info("Vector store cached successfully.")
                else:
                    logging.warning("No documents available to create vector store.")
        except Exception as e:
            logging.error(f"Error initializing or updating vector store: {e}")
            self.db = None

    # Process documents and update embeddings cache
    def process_documents(self, documents):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            updated_documents = []
            failed_files = []  # Track failed files
            file_count = 0

            for doc in documents:
                source = doc.metadata.get("source", "Unknown")
                filename = source.split("/")[-1] if source != "Unknown" else "Unknown File"

                # Check if the file exists
                if source == "Unknown" or not os.path.exists(source):
                    logging.warning(f"File does not exist: {filename}")
                    continue  # Skip non-existent files

                # Check if the file has changed (based on hash or timestamp)
                try:
                    doc_hash = hash_file(source)  # Compute the hash of the file
                    if doc_hash in self.embeddings_cache:
                        logging.info(f"Skipped unchanged file: {filename}")
                        continue  # Skip unchanged files
                except Exception as hash_error:
                    logging.error(f"Failed to compute hash for {filename}: {hash_error}")
                    failed_files.append(filename)
                    continue

                logging.info(f"Processing document: {filename}")

                # Skip empty documents early
                if not doc.page_content and not source:
                    logging.warning(f"Skipping document with no content and no source metadata: {filename}")
                    continue

                # Attempt OCR if no text content is found
                if not doc.page_content:
                    logging.warning(f"No text found in document: {filename}. Attempting OCR...")
                    try:
                        if source.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):  # Image file
                            image = Image.open(source)
                            ocr_text = pytesseract.image_to_string(image)
                        elif source.lower().endswith('.pdf'):  # PDF file
                            from pdf2image import convert_from_path
                            pages = convert_from_path(source)
                            ocr_text = ""
                            for page_number, page in enumerate(pages, start=1):
                                ocr_text += f"\n--- Page {page_number} ---\n"
                                ocr_text += pytesseract.image_to_string(page)
                        else:
                            logging.error(f"Unsupported file type for OCR: {filename}")
                            failed_files.append(filename)
                            continue

                        doc.page_content = ocr_text  # Update the document content with OCR-extracted text
                    except Exception as ocr_error:
                        logging.error(f"OCR failed for document {filename}: {ocr_error}")
                        failed_files.append(filename)
                        continue

                # Split the text into chunks
                text_chunks = text_splitter.split_text(doc.page_content)
                if not text_chunks:
                    logging.warning(f"No content found to split in document: {filename}")
                    failed_files.append(filename)
                    continue

                # Process each chunk and attach metadata
                for chunk in text_chunks:
                    # Skip chunk if its embedding already exists
                    chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
                    if chunk_hash in self.embeddings_cache:
                        logging.info(f"Skipped chunk in {filename}, already in cache.")
                        continue

                    updated_documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": source,
                                "type": doc.metadata.get("type", "Unknown"),
                                "date": doc.metadata.get("date", "Unknown"),
                                "author": doc.metadata.get("author", "Unknown"),
                            }
                        )
                    )

                    # Update embeddings cache
                    try:
                        embedding = self.embeddings.embed_query(chunk)
                        self.embeddings_cache[chunk_hash] = embedding
                        logging.info(f"Embedding added to cache for chunk in document: {filename}")
                    except Exception as embedding_error:
                        logging.error(f"Failed to generate embedding for {filename}: {embedding_error}")
                        failed_files.append(filename)
                        continue

                file_count += 1

                # Save cache periodically to prevent data loss
                if file_count % 10 == 0:
                    self._save_cache()

            # Save embeddings cache to disk after all processing
            self._save_cache()
            logging.info(f"Total files processed: {file_count}")
            logging.warning(f"Failed to process {len(failed_files)} documents: {failed_files}")
            return updated_documents
        except Exception as e:
            logging.error(f"Failed to process documents: {e}")
            return []



    # Create retriever and RetrievalQA chain
    def create_retriever_and_chain(self):
        try:
            if self.db:
                retriever = self.db.as_retriever()
                llm = OllamaLLM(model="llama2:latest")
                qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                logging.info("Retriever and QA chain created successfully.")
                return qa_chain
            else:
                raise ValueError("Vector store is not initialized.")
        except Exception as e:
            logging.error(f"Failed to create retriever and chain: {e}")
            return None