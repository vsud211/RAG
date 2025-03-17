# LangChain Q&A Application with Document Retrieval with Docker support

This repository contains a Flask-based application that leverages LangChain to perform question-answering over a collection of PDF documents. It utilizes embeddings to retrieve relevant document snippets and generate coherent responses.

## Features

* **Document Loading:** Loads PDF documents from a specified directory.
* **Embedding Generation:** Generates embeddings for the document content using a specified embedding model (configured within `embedding_handler.py`).
* **Vector Storage:** Stores document embeddings in a vector database (configured within `embedding_handler.py`).
* **Question Answering:** Uses LangChain's question-answering chain to generate responses based on user queries and retrieved document snippets.
* **Document Metadata Retrieval:** Retrieves and returns metadata (source, type, author, date) of the documents used to generate the response.
* **REST API:** Provides a REST API endpoint for querying the application.
* **Logging:** Implements logging for debugging and monitoring.

## Prerequisites

* Python 3.7+
* Poetry (recommended) or pip for dependency management
* A vector database (e.g., Chroma, FAISS) configured according to your `embedding_handler.py` setup.
* PDF documents placed into the `/app/Doc_repo/` directory.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install dependencies using Poetry (recommended):**

    ```bash
    poetry install
    ```

    Or, install dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Place your PDF documents in the `/app/Doc_repo/` directory.**

4.  **Configure `embedding_handler.py`:**
    * Adjust the embedding model, vector store, and other settings as needed.
    * Ensure that the vector database is running.

## Running the Application

1.  **Start the Flask application:**

    ```bash
    poetry run python app.py
    ```

    or

    ```bash
    python app.py
    ```

2.  **Access the application:**

    * The application will be accessible at `http://0.0.0.0:5000`.
    * The index.html file can be accessed by navigating to the base url.
    * Send POST requests to `http://0.0.0.0:5000/query` with a JSON payload like this:

        ```json
        {
          "query": "What is the main topic of this document?"
        }
        ```

3.  **Example response:**

    ```json
    {
      "query": "What is the main topic of this document?",
      "answer": "The main topic of this document is...",
      "documents_used": [
        {
          "source": "/app/Doc_repo/document1.pdf",
          "type": "pdf",
          "author": "John Doe",
          "date": "2023-10-26",
          "content_preview": "A preview of the relevant document content..."
        },
        {
          "source": "/app/Doc_repo/document2.pdf",
          "type": "pdf",
          "author": "Jane Smith",
          "date": "2023-10-25",
          "content_preview": "Another preview of relevant document content..."
        }
      ]
    }
    ```

## Code Structure

* `app.py`: Main Flask application file.
* `embedding_handler.py`: Handles document loading, embedding generation, vector store management, and QA chain creation.
* `logger_util.py`: Configures logging for the application.
* `requirements.txt`: Lists Python dependencies.
* `templates/index.html`: a simple html file for testing.
* `/app/Doc_repo/`: Directory for storing PDF documents.

## Configuration

* Modify `embedding_handler.py` to configure the embedding model, vector store, and other settings.
* Adjust logging levels in `logger_util.py` as needed.
* The document directory `/app/Doc_repo/` can be changed in `app.py`.

## Future Improvements

* Implement user authentication and authorization.
* Add support for more document types.
* Enhance the front-end user interface.
* Improve error handling and robustness.
* Add web socket support.
* Add support for environment variable configuration.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.
