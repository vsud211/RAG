from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from embedding_handler import EmbeddingHandler
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from logger_util import get_logger

# Get a logger for this module
logging = get_logger(__name__)

# Initialize Flask and SocketIO
app = Flask(__name__, template_folder="templates")
socketio = SocketIO(app, async_mode='eventlet')

# Initialize EmbeddingHandler
embedding_handler = EmbeddingHandler()
qa_chain = None


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
async def query_api():
    logging.debug("Received a POST request on /query endpoint.")
    data = request.get_json()
    if not data:
        logging.warning("No JSON payload received in the POST request.")
        return jsonify({"error": "No data received in request."}), 400

    if "query" not in data:
        logging.warning("Query key missing in the JSON payload.")
        return jsonify({"error": "Please provide a query"}), 400

    query = data["query"]
    logging.info(f"Processing query: {query}")
    try:
        # Run the chain to get the main response
        result = qa_chain.invoke(query)  # Await the asynchronous call
        main_response = result["result"]

        # Fetch relevant documents from the retriever
        retriever = embedding_handler.db.as_retriever()  # Access the retriever from the embedding handler
        retrieved_docs = retriever.get_relevant_documents(query)

        # Format the retrieved documents' metadata
        metadata_list = []
        for doc in retrieved_docs:
            metadata_list.append({
                "source": doc.metadata.get("source", "N/A"),
                "type": doc.metadata.get("type", "N/A"),
                "author": doc.metadata.get("author", "N/A"),
                "date": doc.metadata.get("date", "N/A"),
                "content_preview": doc.page_content[:200] + "..."  # Provide a preview of the content
            })

        logging.info(f"Query processed successfully. Result: {main_response}, Documents retrieved: {len(metadata_list)}")
        
        # Return the response with metadata
        return jsonify({
            "query": query,
            "answer": main_response,
            "documents_used": metadata_list
        })
    except Exception as e:
        logging.error(f"Error while processing query: {query}. Exception: {e}")
        return jsonify({"error": str(e)}), 500




'''
# WebSocket support
@socketio.on("query")
def handle_query(data):
    query = data.get("query", "")
    if not query:
        emit("response", {"error": "No query provided"})
        return

    try:
        result = qa_chain.invoke(query)
        emit("response", {"query": query, "answer": result["result"]})
    except Exception as e:
        emit("response", {"error": str(e)})
'''
if __name__ == "__main__":
    logging.info("Initializing documents and QA chain before starting Flask app.")
    try:
        loader = DirectoryLoader('/app/Doc_repo/', glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        # Process documents and update embeddings cache
        processed_documents = embedding_handler.process_documents(documents)
        logging.info(f"Processed {len(processed_documents)} documents for embedding.")

        # Initialize vector store
        embedding_handler.initialize_vector_store(processed_documents)
        qa_chain = embedding_handler.create_retriever_and_chain()
    except Exception as e:
        logging.error(f"Error during initialization: {e}")
    
    logging.info("Starting Flask application.")
    socketio.run(app, host="0.0.0.0", port=5000)

