# Documentation Agent

- RAG based application to download documentation from a website, create a vector store, and answer user queries with the help of the vector store. It employs advanced RAG techniques like Contextual RAG and Graph RAG.
- It is built on open source libraries and free APIs, to be easily accessible to anyone at no cost.
- It is built using llamaindex as the core framework. 
- Vector database - Chromadb for local, Pinecone for deployment. Can be easily changed.
- Graph Databse - Neo4j is used as the graph database.
- LLM - It is configured to use free llms from gemini / cerebras / groq etc. Also Ollama can be used for local usage.
- Embeddings - It can be used with google embeddings (free and recommended) or any open source embedding from huggingface using fastembed library (local)

## Installation

### Prerequisites
- Python 3.11 or higher
- Node.js and npm (for frontend)

### Option 1: Using uv (Recommended)
[uv](https://docs.astral.sh/uv/) is a fast Python package manager and project manager. It's much faster than pip and handles virtual environments automatically.

#### 1. Install uv
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: using pip
pip install uv
```

#### 2. Clone the repository
```bash
git clone https://github.com/Sparshsing/documentation_agent.git
cd documentation_agent
```

#### 3. Install dependencies
```bash
# This creates a virtual environment and installs all dependencies
uv venv --python 3.11  # or whichever python version above 3.11 you prefer
uv sync
# optional - create detailed requirements file
uv pip compile pyproject.toml -o requirements.txt
```

#### 4. Activate the virtual environment
```bash
# On Linux/macOS
source .venv/bin/activate

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

# On Windows (Command Prompt)
.venv\Scripts\activate.bat
```

### Option 2: Using pip (Traditional method)

#### 1. Clone the repository
```bash
git clone https://github.com/Sparshsing/documentation_agent.git
cd documentation_agent
```

#### 2. Create and activate a virtual environment
```bash
# Create virtual environment
python -m venv .venv

# Activate the environment
# On Linux/macOS
source .venv/bin/activate

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

# On Windows (Command Prompt)
.venv\Scripts\activate.bat
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### Create .env file in project directory
- Add API Keys, and other env variables. refer env_example.txt

### Create an index
- Download the data in data directory. Use download_documentation.py to download from website or github.
eg. run this in a .py file (may not work in jupyter notebook)
```python
from download_documentation import download_docs_from_website, save_all_content_to_file

url = 'https://ai.google.dev/api?lang=python'
download_path = 'data/google_genai/api'

asyncio.run(download_docs_from_website(url, download_path))
save_all_content_to_file(download_path, os.path.join(download_path, '__all_docs__.md'))
```

- Edit core/create_vector_store.py to  input the index name, data path, choose metadata extractor, change settings etc
```python
python core/create_vector_store.py
```
- The index data is saved in processed_data/chromadb, other index files saved in processed_data/index_name directory.
- Now you can Query the index by running backend and frontend.

### Backend
```bash
# From the project root directory
python backend/run_server.py
```

### Frontend
```bash
# From the project root directory
cd frontend
npm install  # First time only
npm run dev
```






