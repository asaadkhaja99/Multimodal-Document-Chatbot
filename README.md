# Multimodal-Document-Chatbot

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Haystack](https://img.shields.io/badge/powered%20by-Haystack-orange.svg)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Multimodal-Document-Chatbot.git
cd Multimodal-Document-Chatbot
```

2. Install dependencies:
```bash
uv sync
```

3. Set up Ollama and pull required models:
```bash
# Install Ollama from https://ollama.com/
# Pull required models
ollama pull nomic-embed-text
ollama pull llama3.2:1b
ollama pull llama-guard3:1b
ollama pull moondream:1.8b
ollama pull qwen3:30b (for evaluation -- only if you have ~20GB memory to spare )
```

4. Add your documents to the `data/` directory

5. Run the application:
```bash
# Activate virtual environment
source .venv/bin/activate

# Run backend
python src/main.py

# After indexing is done and backend is running, run frontend
streamlit run src/user_interface/chat_interface.py           
```

6. Access the Chat Interface: http://localhost:8501

## Overview
While cloud-hosted LLMs like Claude Sonnet or Gemini might provide allow one to easily 'chat' with their documents or get insights about multiple documents at once, there are situations where the use of these services is not appropriate. Think proprietary or sensitive documents that cannot leave your laptop. Having faced similar issues in the past, but unwilling to sacrifice the productivity that comes with using LLMs for text aggregation and quick look up, I thought that I would roll a simple RAG application that uses purely locally hosted LLMs.

However, simple RAG is often deficient in the quality of responses. Particularly, when dealing with documents where information is represented in equations, graphs and other media. In such cases, a slightly more sophisticated mechanism is necessary to enable all the information present in the documents to be represented in text format, such that it can be indexed and queried by the retrieval phase in the RAG application.

#### RAG Pipeline and Technical Approach
To guarantee data security and prevent potential leakage, the entire pipeline, including the language models, the vector database and the observability tool (Langfuse) were hosted locally. The system is currently configured to handle file inputs in PDF, PowerPoint (`.pptx`) and plain text (`.txt`) formats.

In the case of PDFs, in addition to text extraction, the pipeline identifies and extracts embedded images (such as charts, graphs, and diagrams) from PDF documents. These images are then processed by a vision model, `moondream:1.8b`, which generates detailed textual descriptions. These descriptions are indexed alongside the standard text, making visual information searchable and accessible to the chatbot.

To maintain ethical standards, an input validation layer was integrated using the `llama-guard3:1b` model. This "guardrail" model screens user queries for inappropriate or unsafe content before they are processed by the main RAG pipeline, ensuring responsible system behavior.

#### Model Choices
The selection of models was guided by the need to balance high performance with the constraints of local system memory. After experimentation, the following open-source models were chosen:

* Embedding Model (nomic-embed-text): This model surpasses the OpenAI text-embedding-ada-002 and text-embedding-3-small on long and short context tasks, allowing for high quality embeddings to be generated.
* Generative LLM (llama3.2:1b): Serves as the primary 'brain' of the chatbot, responsible for synthesizing retrieved information and generating coherent, context-aware answers.
* Guardrail Model (llama-guard3:1b): A specialized, lightweight model used for input moderation to ensure user queries are safe and on-topic.
* Vision Model (moondream:1.8b): A light weight multimodal model used to generate descriptions of visual data extracted from documents.

### Pipeline Structure
The end-to-end forecasting pipeline is orchestrated by the main script src/main.py and is divided into two distinct stages:
1. Indexing:
    * This stage handles the ingestion of raw source documents to build a searchable knowledge base. The pipeline begins by scanning the specified data directory for supported file types (.pdf, .pptx, .txt).
    * For PDF documents, it performs multimodal processing: text is extracted directly, while embedded images are passed to the moondream:1.8b model to generate textual descriptions.
    * All extracted content is then cleaned and segmented into smaller, overlapping chunks to ensure semantic continuity. The chunking strategy in use is to chunk documents into chunks of 100 words maximum, with 10 word overlaps. This is a simple strategy that serves as a starting point, and will likely have to be reviewed after evaluation.
    * Each chunk is subsequently converted into a vector embedding by the nomic-embed-text model and stored in a ChromaDB vector database. To ensure efficiency, the pipeline automatically skips any documents that have already been indexed.
2. Querying:
    * This is the user-facing stage where the chatbot generates answers based on the indexed knowledge. The process begins when a user submits a question.
    * The query first passes through a safety check powered by the llama-guard3:1b model. Valid queries proceed, while inappropriate ones are deflected with a static response.
    * The user's question is then converted into a vector embedding, which is used to retrieve the most relevant context chunks from the ChromaDB database.
    * Finally, the original question and the retrieved context are combined into a comprehensive prompt for the llama3.2:1b model, which synthesizes the information to generate a coherent, accurate answer.

### Directory
The directory is organised as such:
-   config.yaml: Contains configuration that control all pipeline parameters.
-   data/: Directory containing the corpus that information is retrieved from
-   src/: All the main source code for the project.
    -   src/api/: Script for serving the query pipeline as a FastAPI endpoint
    -   src/pipelines/: Scripts for orchestrating the query and indexing pipelines using Haystack
    -   src/user_interface/: Script for running the chat interface that users can submit queries to the endpoint serving the query pipeline.
    - src/evaluate/: Script for running deepeval on RAG Pipeline

### Hardware
Development and deployment of the RAG pipeline was tested on a Surface Laptop 5 with 16GB RAM and a 12th Gen Intel(R) Core(TM) i5-1235U.


## Running the application

### Set up locally hosted services
Before the application can be run, the following services need to be set up and running on your local machine.

#### 1. Ollama
* Install Ollama, if not already done: 
    * Download and run the official installer from the [Ollama website](https://ollama.com/). The application will run as a background service after installation.
* Pull the required models:
    ```bash
    # Embedding model
    ollama pull nomic-embed-text

    # Generative LLM for answering questions
    ollama pull llama3.2:1b

    # Guardrail model for input validation
    ollama pull llama-guard3:1b

    # Vision model for describing images
    ollama pull moondream:1.8b
    ```
* Verify Installation:
    ```bash
    ollama list
    ```
#### 2. ChromaDB (vector database)
ChromaDB stores the vector embeddings of your documents, enabling efficient semantic search. ChromaDB is embedded within the application and requires no additional setup. The database files are automatically created in the `chroma_db/` directory when you run the application for the first time.


#### 3. Langfuse (Observability) - Optional
Langfuse is used for tracing and evaluating the pipeline's performance. It also runs on Docker.
* Clone the Langfuse repository
git clone https://github.com/langfuse/langfuse.git
    cd langfuse
* Start the application
docker compose up
* Open http://localhost:3000 in your browser to access the Langfuse UI. Add the following variables to your .env file
```bash
    LANGFUSE_SECRET_KEY=
    LANGFUSE_PUBLIC_KEY=
    LANGFUSE_HOST=
    HAYSTACK_CONTENT_TRACING_ENABLED=True
```
Despite this setup procedure being the one officially recommmended in the docs, I have noticed issues reliably running Langfuse locally on different machines. Therefore, this is optional. Simply set `enabled: False` in the config.yaml file to disable Langfuse.

### Running the application

Before running the pipeline:

1. Adjust variables in `config.yaml` to desired settings
2. Add retrieval docs to the `data/` directory
3. Ensure Ollama is running with the required models pulled

Usage:
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the main application
python src/main.py


# After indexing is done and backend is running, run frontend
streamlit run src/user_interface/chat_interface.py 
```

## Evaluation

Evaluation of the RAG pipeline is conducted to quantitatively measure its performance and accuracy. The assessment is performed using the DeepEval framework, which leverages a local language model to grade the pipeline's responses, ensuring that the evaluation process adheres to the same privacy-first principles as the application itself.

### Test Dataset

The evaluation suite includes 10 comprehensive questions based on the "Attention Is All You Need" paper (Vaswani et al., 2017), covering:

- **Architecture fundamentals**: Main contributions and core concepts of the Transformer
- **Technical details**: Multi-head attention mechanism, positional encoding, and mathematical formulations
- **Model specifications**: Network dimensions, hyperparameters, and architectural choices
- **Performance metrics**: BLEU scores and computational complexity comparisons
- **Implementation details**: Activation functions, layer structures, and training datasets

### Methodology

The pipeline is tested against this curated set of questions covering key concepts from the Transformer architecture paper. For each question, the RAG system's generated answer is compared against a pre-defined, factually correct "expected answer."

The evaluation employs DeepEval's GEval metric, configured to perform a binary (pass/fail) correctness assessment. The locally hosted `qwen3:30b` is used as the evaluator to grade the pipeline's output. A score of 1 is awarded if the actual output is factually correct and semantically aligned with the expected answer, and a score of 0 otherwise.

### Running the Evaluation

To evaluate your RAG pipeline:

1. Ensure your FastAPI server is running:
   ```bash
   # In one terminal, start the API server
   python src/main.py
   ```

2. Run the evaluation suite:
   ```bash
   # In another terminal, run the tests
   pytest src/evaluate/deepeval_test.py -v
   ```

The evaluation will test all 10 questions and provide a comprehensive report of the pipeline's performance on Transformer-related queries.

## Future Improvements

- Support for additional document formats (Word, Excel, etc.)
- Advanced chunking strategies beyond simple word-based splitting
- Integration with additional vision models for better image analysis
- Performance optimization for larger document collections

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
