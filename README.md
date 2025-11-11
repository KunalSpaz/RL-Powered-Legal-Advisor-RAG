# RL-Powered-Legal-Advisor-RAG
# ⚖️ Legal RAG with Reinforcement Learning

An intelligent legal advisory chatbot that combines Retrieval-Augmented Generation (RAG) with Reinforcement Learning to provide accurate legal advice based on Indian criminal laws (IPC, CrPC, BNS, BNSS). The system implements a Markov Decision Process (MDP) where states include user queries, retrieved documents, and system parameters (retrieval_k, chunk_size, temperature, search_type), actions involve adjusting these parameters, and rewards combine response quality (40%), relevance score (30%), and retrieval accuracy (30%) weighted with 70% user feedback. The multi-agent architecture features a DocumentRetrievalAgent using hybrid search (semantic similarity + pattern matching, retrieving 50 docs and scoring them to return top 15), RelevanceCheckAgent validating section presence via regex, and ResponseGenerationAgent using GPT-4o-mini for grounded generation. The RL engine continuously learns through online policy gradients with ε-greedy exploration (10% after 10 interactions), adaptive weight learning that adjusts reward components based on human-AI alignment, and experience replay storing successful (reward > 0.7) and failed (reward < 0.3) patterns. Built with LangChain, ChromaDB (HNSW vector indexing), OpenAI embeddings (text-embedding-ada-002, 1536D), and Streamlit UI, the system features auto-healing for database corruption, pagewise PDF indexing (1500 char chunks, 300 overlap), query expansion for legal abbreviations, and real-time performance tracking with trend analysis (comparing last 10 vs. previous 10 rewards). Install via `pip install streamlit openai langchain langchain-openai langchain-chroma chromadb PyMuPDF numpy`, configure API key and PDF folder path, run `streamlit run Legal_RAG.py`, and the system learns optimal configurations automatically through user interactions without requiring batch training or pre-labeled data, converging to stable performance after 100-200 queries while maintaining continuous improvement indefinitely.

## Quick Start

1. **Install**: `pip install streamlit openai langchain langchain-chroma chromadb PyMuPDF numpy`
2. **Configure**: Set `OPENAI_API_KEY` and `PDF_FOLDER_PATH` in `Legal_RAG.py`
3. **Run**: `streamlit run Legal_RAG.py`
4. **Use**: Ask legal questions, rate responses (0.0-1.0 slider), and watch the RL system automatically optimize retrieval strategies and generation parameters based on your feedback

## Key Features
- **Hybrid Retrieval**: Semantic search + regex pattern scoring for section-specific queries
- **Multi-Agent Pipeline**: Retrieval → Relevance Check → Generation with state validation
- **Online RL**: Learns from every interaction using MDP formulation with ε-greedy policy
- **Adaptive Rewards**: Multi-objective function (quality + relevance + accuracy) + human feedback
- **Auto-Optimization**: Adjusts retrieval_k (15→20), search_type (similarity↔MMR), temperature (0.0↔0.2), chunk_size (800/1000/1200)
- **Experience Replay**: Stores success/failure patterns for query-type-specific learning
- **Fault Tolerance**: Auto-healing ChromaDB corruption recovery

## Technologies
LangChain, ChromaDB, OpenAI GPT-4o-mini, text-embedding-ada-002, Streamlit, PyMuPDF, NumPy, Custom MDP/RL Implementation

