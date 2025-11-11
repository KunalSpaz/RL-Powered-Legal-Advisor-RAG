########## ========== CONFIGURATION ========== ##########
import os, shutil, time, hashlib, numpy as np, re
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict
import fitz, openai, streamlit as st
from collections import deque, defaultdict
from dataclasses import dataclass

OPENAI_API_KEY = 
CHROMA_PERSIST_DIR = ".rl_rag_dir"

########## ========== QUERY EXPANSION ========== ##########
def expand_legal_query(query: str) -> str:
    expansions = {
        r'\bIPC\b': 'IPC Indian Penal Code',
        r'\bCrPC\b': 'CrPC Criminal Procedure Code',
        r'\bCPC\b': 'CPC Civil Procedure Code',
        r'\bSC\b': 'SC Supreme Court',
        r'\bHC\b': 'HC High Court',
        r'\bFIR\b': 'FIR First Information Report',
        r'\bBNS\b': 'BNS Bharatiya Nyaya Sanhita',
        r'\bBNSS\b': 'BNSS Bharatiya Nagarik Suraksha Sanhita',
        r'\bSection\s+(\d+)': r'Section \1 Sec \1',
    }
    expanded = query
    for pattern, replacement in expansions.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded

########## ========== EMBEDDING COMPONENT ========== ##########
class SimpleOpenAIEmbeddings:
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = []
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(input=batch, model=self.model)
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                if len(texts) > batch_size:
                    time.sleep(0.1)
            return embeddings
        except Exception as e:
            return [[0.0]*1536 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding
        except Exception as e:
            return [0.0]*1536

########## ========== DOCUMENT LOADING (PAGEWISE) ========== ##########
def pagewise_pdf_loader(file_path: str) -> List['Document']:
    from langchain.schema import Document
    documents = []
    try:
        pdf = fitz.open(file_path)
        for page_num in range(pdf.page_count):
            try:
                page = pdf.load_page(page_num)
                text = page.get_text("text")
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={"source": Path(file_path).name, "page": page_num + 1, "filepath": str(file_path)}
                    )
                    documents.append(doc)
            except Exception:
                continue
        pdf.close()
    except Exception:
        pass
    return documents

########## ========== RL-RAG STATE TYPEDEFS ========== ##########
class DocumentState(TypedDict):
    query: str
    expanded_query: str
    documents: list
    context: str
    retrieval_method: str
    num_documents_retrieved: int
    error: Optional[str]

class RelevanceState(TypedDict):
    query: str
    context: str
    is_relevant: bool
    relevance_score: float
    reasoning: str
    error: Optional[str]

class GenerationState(TypedDict):
    query: str
    context: str
    response: str
    confidence_score: float
    sources_used: list
    error: Optional[str]

@dataclass
class RLMetrics:
    query_id: str
    response_quality: float
    retrieval_accuracy: float
    relevance_score: float
    user_feedback: float
    reward: float
    timestamp: float

########## ========== RAG AGENTS ========== ##########
class DocumentRetrievalAgent:
    def __init__(self, vectorstore, text_splitter, llm):
        self.vectorstore = vectorstore
        self.text_splitter = text_splitter
        self.llm = llm

    def run(self, state: DocumentState) -> DocumentState:
        try:
            expanded_query = expand_legal_query(state["query"])
            state["expanded_query"] = expanded_query
            
            section_match = re.search(r'section\s+(\d+)', state["query"].lower())
            
            if section_match:
                section_num = section_match.group(1)
                all_docs = self.vectorstore.similarity_search(expanded_query, k=50)
                
                section_patterns = [
                    rf'\bSection\s*{section_num}\b',  
                    rf'\bsection\s*{section_num}\b',  
                    rf'\bSec\.\s*{section_num}\b',    
                    rf'\bsec\.\s*{section_num}\b',    
                    rf'\bSec\s*{section_num}\b',      
                    rf'\bsec\s*{section_num}\b',     
                    rf'\b{section_num}\.\s+',         
                    rf'^{section_num}\.\s+',          
                ]
                
                scored_docs = []
                for doc in all_docs:
                    content = doc.page_content
                    score = 0
                    
                    for pattern in section_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                        score += len(matches) * 10
                    
                    if re.search(rf'^.{{0,200}}Section\s*{section_num}\b', content, re.IGNORECASE | re.DOTALL):
                        score += 50
                    
                    scored_docs.append((doc, score))
                
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                
                if scored_docs[0][1] > 0:
                    final_docs = [doc for doc, score in scored_docs[:15] if score > 0]
                    state["retrieval_method"] = f"hybrid_section_found_{len(final_docs)}_docs"
                else:
                    final_docs = all_docs[:15]
                    state["retrieval_method"] = f"hybrid_section_notfound_broadSearch"
                
            else:
                final_docs = self.vectorstore.similarity_search(expanded_query, k=10)
                state["retrieval_method"] = "similarity"
            
            context = "\n\n=====DOCUMENT SEPARATOR=====\n\n".join([doc.page_content for doc in final_docs])
            state["documents"] = final_docs
            state["context"] = context
            state["num_documents_retrieved"] = len(final_docs)
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
            state["context"] = ""
            state["num_documents_retrieved"] = 0
            state["expanded_query"] = state["query"]
            state["retrieval_method"] = "error"
        return state

class RelevanceCheckAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: RelevanceState) -> RelevanceState:
        try:
            context = state["context"]
            query = state["query"]
            
            if len(context.strip()) < 50:
                state["is_relevant"] = False
                state["relevance_score"] = 0.0
                state["reasoning"] = "No sufficient context retrieved"
                state["error"] = None
                return state
            
            query_lower = query.lower()
            context_lower = context.lower()
            
            section_match = re.search(r'section\s+(\d+)', query_lower)
            
            if section_match:
                section_num = section_match.group(1)
                
                section_patterns = [
                    rf'\bSection\s*{section_num}\b',
                    rf'\bsection\s*{section_num}\b',
                    rf'\bSec\.\s*{section_num}\b',
                    rf'\bsec\.\s*{section_num}\b',
                    rf'\bSec\s*{section_num}\b',
                    rf'\bsec\s*{section_num}\b',
                    rf'\b{section_num}\.\s+',
                    rf'^{section_num}\.\s+',
                ]
                
                found = any(re.search(pattern, context, re.IGNORECASE | re.MULTILINE) for pattern in section_patterns)
                
                if found:
                    state["is_relevant"] = True
                    state["relevance_score"] = 1.0
                    state["reasoning"] = f"✓ Section {section_num} found in retrieved context"
                else:
                    state["is_relevant"] = False
                    state["relevance_score"] = 0.0
                    state["reasoning"] = f"✗ Section {section_num} not found despite broad search. May need larger chunks or different PDF."
            else:
                stopwords = {'what', 'is', 'the', 'of', 'in', 'and', 'or', 'for', 'to', 'a', 'an', 'explain', 'describe', 'tell'}
                query_words = [w for w in query_lower.split() if len(w) > 3 and w not in stopwords]
                
                if not query_words:
                    overlap_ratio = 0.5
                else:
                    matches = sum(1 for word in query_words if word in context_lower)
                    overlap_ratio = matches / len(query_words)
                
                if overlap_ratio >= 0.3:
                    state["is_relevant"] = True
                    state["relevance_score"] = 1.0
                    state["reasoning"] = f"Found {int(overlap_ratio*100)}% query term overlap in context"
                else:
                    state["is_relevant"] = False
                    state["relevance_score"] = 0.0
                    state["reasoning"] = f"Only {int(overlap_ratio*100)}% query term overlap - insufficient relevance"
            
            state["error"] = None
            
        except Exception as e:
            state["error"] = str(e)
            state["relevance_score"] = 0.0
            state["is_relevant"] = False
            state["reasoning"] = f"Error: {str(e)}"
        
        return state

class ResponseGenerationAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: GenerationState) -> GenerationState:
        try:
            prompt = f"""You are a legal advisor expert in Indian Penal Code. Use ONLY the information provided in the context below to answer the question.

Context (multiple document sections):
{state["context"]}

Question: {state["query"]}

Instructions:
- Search through ALL the document sections provided above
- Look for the specific section number or topic mentioned in the question
- Provide detailed explanation including section number, provisions, punishments, and any relevant notes
- If you find the information, give a comprehensive answer
- Be specific and cite the section details

Answer:"""
            output = self.llm.invoke(prompt)
            state["response"] = output.content.strip()
            state["confidence_score"] = 1.0 if "unable to answer" not in state["response"].lower() else 0.2
            state["sources_used"] = []
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
            state["response"] = "Unable to generate response due to error."
            state["confidence_score"] = 0.0
        return state

########## ========== RL FEEDBACK, OPTIMIZER, MEMORY ========== ##########
class RewardModel:
    def __init__(self):
        self.feedback_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        self.learning_rate = 0.01
        self.weights = {"response_quality": 0.4, "relevance_score": 0.3, "retrieval_accuracy": 0.3}

    def calculate_reward(self, response_quality, relevance_score, retrieval_accuracy, user_feedback=None):
        base_reward = (response_quality * self.weights["response_quality"] + relevance_score * self.weights["relevance_score"] + retrieval_accuracy * self.weights["retrieval_accuracy"])
        if user_feedback is not None:
            final_reward = 0.7 * user_feedback + 0.3 * base_reward
            self.update_weights(base_reward, user_feedback)
        else:
            final_reward = base_reward
        return max(0.0, min(1.0, final_reward))

    def update_weights(self, automated_score, user_feedback):
        alignment = abs(automated_score - user_feedback)
        if alignment < 0.2:
            self.weights["response_quality"] = min(0.5, self.weights["response_quality"] + self.learning_rate * 0.1)
        elif alignment > 0.5:
            self.weights["response_quality"] = max(0.2, self.weights["response_quality"] - self.learning_rate * 0.1)

    def update_performance(self, metrics):
        self.feedback_history.append(metrics)
        self.performance_metrics["rewards"].append(metrics.reward)
        self.performance_metrics["quality"].append(metrics.response_quality)
        self.performance_metrics["relevance"].append(metrics.relevance_score)

    def get_performance_trend(self, window_size=10):
        if len(self.performance_metrics["rewards"]) < window_size:
            return {"trend": "insufficient_data", "improvement": 0.0, "total_interactions": len(self.feedback_history)}
        recent = np.mean(self.performance_metrics["rewards"][-window_size:])
        older = np.mean(self.performance_metrics["rewards"][-2*window_size:-window_size]) if len(self.performance_metrics["rewards"]) >= 2*window_size else recent
        delta = recent - older
        trend = "improving" if delta > 0.02 else ("stable" if abs(delta) <= 0.02 else "declining")
        return {"trend": trend, "improvement": delta, "recent_avg": recent, "total_interactions": len(self.feedback_history)}

class ParameterOptimizer:
    def __init__(self):
        self.parameter_history = []
        self.current_parameters = {"retrieval_k": 15, "chunk_size": 1000, "temperature": 0.0, "search_type": "similarity"}
        self.best_parameters = self.current_parameters.copy()
        self.best_reward = 0.0
        self.exploration_rate = 0.1
        self.performance_threshold = 0.7

    def suggest_parameters(self, current_reward, performance_trend):
        s = {}
        total_interactions = performance_trend.get("total_interactions", 0)
        if performance_trend["trend"] == "declining" or current_reward < self.performance_threshold:
            if current_reward < 0.5:
                s["retrieval_k"] = min(20, self.current_parameters["retrieval_k"] + 5)
                s["search_type"] = "mmr" if self.current_parameters["search_type"] == "similarity" else "similarity"
            if current_reward < 0.6:
                s["temperature"] = 0.2 if self.current_parameters["temperature"] == 0.0 else 0.0
        elif np.random.random() < self.exploration_rate and total_interactions > 10:
            s["chunk_size"] = np.random.choice([800, 1000, 1200])
        return s

    def apply_suggestions(self, s):
        if s:
            self.current_parameters.update(s)

    def update_best_parameters(self, reward):
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_parameters = self.current_parameters.copy()

class RLMemory:
    def __init__(self, max_size=500):
        self.successful_patterns = deque(maxlen=max_size)
        self.failed_patterns = deque(maxlen=max_size)

    def store_interaction(self, query, context, response, reward):
        pattern = {"query_type": self.classify_query(query), "context_length": len(context), "response_length": len(response), "reward": reward, "timestamp": time.time()}
        if reward > 0.7:
            self.successful_patterns.append(pattern)
        elif reward < 0.3:
            self.failed_patterns.append(pattern)

    def classify_query(self, query):
        q = query.lower()
        if any(x in q for x in ["what", "define", "definition"]):
            return "definition"
        if any(x in q for x in ["how", "process", "procedure"]):
            return "process"
        if any(x in q for x in ["why", "reason", "because"]):
            return "explanation"
        return "general"

    def get_insights(self):
        if not self.successful_patterns:
            return {"message": "No patterns learned yet"}
        suc = [p["query_type"] for p in self.successful_patterns]
        fail = [p["query_type"] for p in self.failed_patterns]
        return {"most_successful_query_type": max(set(suc), key=suc.count) if suc else "none", "most_failed_query_type": max(set(fail), key=fail.count) if fail else "none", "avg_successful_reward": np.mean([p["reward"] for p in self.successful_patterns]) if self.successful_patterns else 0.0, "pattern_count": len(self.successful_patterns)}

########## ========== RL-ENHANCED RAG CHATBOT ORCHESTRATOR ========== ##########
class RLEnhancedLegalChatbot:
    def __init__(self, pdf_folder_path: str, openai_api_key: str, temperature: float = 0.0, chroma_persist_directory: str = ".rl_rag_dir", chunk_size: int = 1500, chunk_overlap: int = 300, max_retries: int = 2):
        self.pdf_folder_path = Path(pdf_folder_path)
        self.chroma_persist_directory = chroma_persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_retries = max_retries
        self.temperature = temperature
        self.openai_api_key = openai_api_key
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_openai import ChatOpenAI
        self.embeddings = SimpleOpenAIEmbeddings(api_key=openai_api_key)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=self.temperature, max_tokens=1500, api_key=self.openai_api_key, streaming=False)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len, separators=["\n\n", "\n", " ", ""])
        self.vectorstore = Chroma(persist_directory=self.chroma_persist_directory, collection_name="rl-rag-chroma", embedding_function=self.embeddings)
        self.reward_model = RewardModel()
        self.parameter_optimizer = ParameterOptimizer()
        self.rl_memory = RLMemory()
        self.interaction_count = 0
        self.document_agent = DocumentRetrievalAgent(self.vectorstore, self.text_splitter, self.llm)
        self.relevance_agent = RelevanceCheckAgent(self.llm)
        self.generation_agent = ResponseGenerationAgent(self.llm)

    def clear_vectorstore(self):
        try:
            if hasattr(self.vectorstore, '_client') and self.vectorstore._client:
                collections = self.vectorstore._client.list_collections()
                for collection in collections:
                    if collection.name == "rl-rag-chroma":
                        self.vectorstore._client.delete_collection(name="rl-rag-chroma")
                        break
            print("✓ Vectorstore cleared")
        except Exception:
            pass

    def load_and_process_documents_pagewise(self, pdf_files: List[str], progress_callback=None) -> int:
        total_chunks = 0
        total_pages = 0
        try:
            if progress_callback:
                progress_callback(0, "Starting pagewise document loading...")
            page_counts = {}
            for pdf_file in pdf_files:
                try:
                    pdf = fitz.open(pdf_file)
                    page_counts[pdf_file] = pdf.page_count
                    total_pages += pdf.page_count
                    pdf.close()
                except:
                    page_counts[pdf_file] = 0
            if progress_callback:
                progress_callback(5, f"Found {len(pdf_files)} PDFs, {total_pages} total pages - pagewise indexing started")
            processed_pages = 0
            page_batch_size = 10
            for pdf_idx, pdf_file in enumerate(pdf_files):
                filename = Path(pdf_file).name
                if progress_callback:
                    progress_callback(int(5 + (pdf_idx / len(pdf_files)) * 90), f"Processing {filename} ({pdf_idx + 1}/{len(pdf_files)})")
                pages = pagewise_pdf_loader(pdf_file)
                if not pages:
                    continue
                for batch_start in range(0, len(pages), page_batch_size):
                    batch_pages = pages[batch_start:batch_start + page_batch_size]
                    batch_chunks = self.text_splitter.split_documents(batch_pages)
                    if batch_chunks:
                        self.vectorstore.add_documents(batch_chunks)
                        total_chunks += len(batch_chunks)
                    processed_pages += len(batch_pages)
                    msg = f"Pages indexed: {processed_pages}/{total_pages} • Chunks created: {total_chunks}"
                    if progress_callback:
                        progress = int(5 + (processed_pages / total_pages) * 90)
                        progress_callback(progress, f"{msg} • Latest PDF: {filename} pages {batch_start+1}-{batch_start+len(batch_pages)}")
            if progress_callback:
                progress_callback(100, f"✓ Finished! {processed_pages} pages indexed, {total_chunks} chunks created.")
            else:
                print(f"✓ Finished! {processed_pages} pages indexed, {total_chunks} chunks created from {len(pdf_files)} PDFs.")
            return total_chunks
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Error: {e}")
            print(f"Error during pagewise loading: {e}")
            raise e

    def chat(self, query: str) -> dict:
        doc_state = self.document_agent.run({"query": query, "expanded_query": query, "documents": [], "context": "", "retrieval_method": "similarity", "num_documents_retrieved": 0, "error": None})
        rel_state = self.relevance_agent.run({"query": query, "context": doc_state.get("context", ""), "is_relevant": False, "relevance_score": 0.0, "reasoning": "", "error": None})
        gen_state = self.generation_agent.run({"query": query, "context": doc_state.get("context", ""), "response": "", "confidence_score": 0.0, "sources_used": [], "error": None})
        response = gen_state.get("response", "")
        return {"query": query, "expanded_query": doc_state.get("expanded_query", query), "response": response, "metadata": {"num_documents_retrieved": doc_state.get("num_documents_retrieved", 0), "relevance_score": rel_state.get("relevance_score", 0.0), "confidence_score": gen_state.get("confidence_score", 0.0), "context": doc_state.get("context", ""), "relevance_reasoning": rel_state.get("reasoning", ""), "retrieval_method": doc_state.get("retrieval_method", ""), "error": gen_state.get("error")}}

    def chat_with_learning(self, query: str, user_feedback: float = None) -> dict:
        self.interaction_count += 1
        query_id = hashlib.md5(f"{query}{time.time()}".encode()).hexdigest()[:8]
        try:
            current_params = self.parameter_optimizer.current_parameters
            if current_params["chunk_size"] != self.chunk_size:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=current_params["chunk_size"], chunk_overlap=self.chunk_overlap, length_function=len, separators=["\n\n", "\n", " ", ""])
            base_response = self.chat(query)
            metadata = base_response.get("metadata", {})
            num_docs = metadata.get("num_documents_retrieved", 0)
            rel_score = metadata.get("relevance_score", 0.0)
            resp_qual = metadata.get("confidence_score", 0.0)
            ret_acc = min(1.0, num_docs / current_params["retrieval_k"]) if current_params["retrieval_k"] > 0 else 0.0
            reward = self.reward_model.calculate_reward(resp_qual, rel_score, ret_acc, user_feedback)
            metrics = RLMetrics(query_id=query_id, response_quality=resp_qual, retrieval_accuracy=ret_acc, relevance_score=rel_score, user_feedback=user_feedback if user_feedback is not None else 0.0, reward=reward, timestamp=time.time())
            self.reward_model.update_performance(metrics)
            self.parameter_optimizer.update_best_parameters(reward)
            self.rl_memory.store_interaction(query, metadata.get("context", ""), base_response.get("response", ""), reward)
            perf_trend = self.reward_model.get_performance_trend()
            param_suggestion = self.parameter_optimizer.suggest_parameters(reward, perf_trend)
            if param_suggestion:
                self.parameter_optimizer.apply_suggestions(param_suggestion)
            enhanced_response = base_response.copy()
            enhanced_response["rl_data"] = {"query_id": query_id, "reward": reward, "user_feedback": user_feedback, "performance_trend": perf_trend, "parameter_suggestions": param_suggestion, "interaction_count": self.interaction_count, "current_parameters": current_params.copy(), "learning_insights": self.rl_memory.get_insights()}
            return enhanced_response
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Exception in chat_with_learning: {e}")
            print(error_msg)
            return {"query": query, "expanded_query": query, "response": f"An error occurred: {str(e)}", "error": str(e), "metadata": {"num_documents_retrieved": 0, "relevance_score": 0.0, "confidence_score": 0.0, "context": "", "relevance_reasoning": "", "retrieval_method": ""}, "rl_data": {"error": str(e), "query_id": query_id, "reward": 0.0, "performance_trend": {"trend": "error"}}}

    def get_learning_summary(self) -> dict:
        return {"interaction_count": self.interaction_count, "performance_trend": self.reward_model.get_performance_trend(), "current_parameters": self.parameter_optimizer.current_parameters, "best_parameters": self.parameter_optimizer.best_parameters, "best_reward": self.parameter_optimizer.best_reward, "learning_insights": self.rl_memory.get_insights(), "feedback_count": len([m for m in self.reward_model.feedback_history if m.user_feedback is not None]), "reward_weights": self.reward_model.weights}

########## ========== AUTO-HEALING CHROMADB INITIALIZATION ========== ##########
def safe_chroma_init(chatbot_cls, pdf_folder_path, openai_api_key, temperature, chroma_persist_directory, pdf_files, update_progress):
    try:
        chatbot = chatbot_cls(pdf_folder_path=pdf_folder_path, openai_api_key=openai_api_key, temperature=temperature, chroma_persist_directory=chroma_persist_directory)
        num_chunks = chatbot.load_and_process_documents_pagewise([str(f) for f in pdf_files], update_progress)
        return chatbot, num_chunks, False
    except Exception as e:
        err_str = str(e).lower()
        if "compaction" in err_str or "hnsw segment writer" in err_str or "chroma" in err_str:
            if os.path.exists(chroma_persist_directory):
                shutil.rmtree(chroma_persist_directory, ignore_errors=True)
            st.warning("⚠️ Detected ChromaDB corruption. Auto-deleting DB folder and retrying once.")
            chatbot = chatbot_cls(pdf_folder_path=pdf_folder_path, openai_api_key=openai_api_key, temperature=temperature, chroma_persist_directory=chroma_persist_directory)
            num_chunks = chatbot.load_and_process_documents_pagewise([str(f) for f in pdf_files], update_progress)
            return chatbot, num_chunks, True
        else:
            raise

########## ========== STREAMLIT UI/APP MAIN LOOP ========== ##########
st.set_page_config(page_title="Legal RL-RAG Chatbot", page_icon="⚖️", layout="wide")
st.title("⚖️ RL-Enhanced Legal RAG Chatbot")
st.markdown("Ask legal questions and rate the answers. The chatbot learns and improves automatically using reinforcement learning.")

with st.sidebar:
    st.header("System Information")
    st.info(f"📁 Document Folder:\n{PDF_FOLDER_PATH}")
    temperature = st.slider("Model Temperature", 0.0, 1.0, 0.0, 0.05)
    st.markdown("---")
    st.subheader("Vector Database Settings")
    auto_clear = st.checkbox("Clear vector DB after each query", value=False, help="If enabled, reloads documents fresh for each question. Slower but ensures clean state.")
    if st.button("🗑️ Clear Vector DB Now", use_container_width=True):
        if "chatbot" in st.session_state:
            st.session_state["chatbot"].clear_vectorstore()
            st.success("✓ Vector database cleared!")
            st.rerun()
    st.caption("✨ ULTRA MODE: Retrieves 50 docs, scores them, returns top 15! Chunk size: 1500 | Performance trend: 10 interactions")

if "show_question_form" not in st.session_state:
    st.session_state["show_question_form"] = True

if os.path.exists(PDF_FOLDER_PATH):
    if "chatbot" not in st.session_state or st.session_state.get("last_temp") != temperature:
        try:
            pdf_files = list(Path(PDF_FOLDER_PATH).glob("*.pdf"))
            progress_bar = st.progress(0)
            status_text = st.empty()
            def update_progress(percent, message):
                progress_bar.progress(percent)
                status_text.info(message)
            chatbot, num_chunks, was_recovered = safe_chroma_init(RLEnhancedLegalChatbot, PDF_FOLDER_PATH, OPENAI_API_KEY, temperature, CHROMA_PERSIST_DIR, pdf_files, update_progress)
            st.session_state["chatbot"] = chatbot
            st.session_state["num_chunks"] = num_chunks
            st.session_state["last_temp"] = temperature
            st.session_state["history"] = []
            st.session_state["pdf_files"] = pdf_files
            progress_bar.empty()
            status_text.empty()
            if was_recovered:
                st.success("✅ ChromaDB was corrupted, but was automatically reset and rebuilt!")
            else:
                st.success(f"✅ Chatbot ready! Indexed {num_chunks} chunks from {len(pdf_files)} PDFs (chunk_size=1500, overlap=300).")
        except Exception as e:
            st.error(f"Initialization failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
else:
    st.error(f"PDF folder not found: {PDF_FOLDER_PATH}")
    st.stop()

chatbot = st.session_state.get("chatbot", None)

if chatbot:
    if st.session_state["show_question_form"]:
        st.subheader("❓ Ask Your Legal Question")
        with st.form("chat_form", clear_on_submit=False):
            user_query = st.text_area("Your legal question or scenario:", key="q_area", height=100, placeholder="Example: What is Section 304 of IPC?")
            feedback_score = st.slider("Rate the last answer (0.0=poor, 1.0=excellent)", 0.0, 1.0, 1.0, 0.05, key="fb_area")
            submitted = st.form_submit_button("Get Answer", use_container_width=True)
            if submitted and user_query:
                with st.spinner("🔍 Using ULTRA search: 50 docs retrieved, scored, top 15 returned..."):
                    response = chatbot.chat_with_learning(user_query, user_feedback=feedback_score)
                    answer = response.get("response", "No response generated")
                    metadata = response.get("metadata", {})
                    rl_data = response.get("rl_data", {})
                    expanded_query = response.get("expanded_query", user_query)
                    rl_metrics = {"Docs Retrieved": metadata.get("num_documents_retrieved", 0), "Relevance Score": metadata.get("relevance_score", 0.0), "Confidence Score": metadata.get("confidence_score", 0.0), "RL Reward": rl_data.get("reward", 0.0), "Performance Trend": rl_data.get("performance_trend", {}).get("trend", "NA")}
                    st.session_state["history"].append({"question": user_query, "expanded_query": expanded_query, "answer": answer, "metrics": rl_metrics, "relevance_reasoning": metadata.get("relevance_reasoning", ""), "retrieval_method": metadata.get("retrieval_method", "")})
                    st.session_state["show_question_form"] = False
                    if auto_clear:
                        with st.spinner("🔄 Clearing and reloading vector database..."):
                            chatbot.clear_vectorstore()
                            from langchain_chroma import Chroma
                            chatbot.vectorstore = Chroma(persist_directory=chatbot.chroma_persist_directory, collection_name="rl-rag-chroma", embedding_function=chatbot.embeddings)
                            pdf_files = st.session_state.get("pdf_files", [])
                            progress_bar_reload = st.progress(0)
                            status_reload = st.empty()
                            def reload_progress(p, m):
                                progress_bar_reload.progress(p)
                                status_reload.info(m)
                            chatbot.load_and_process_documents_pagewise([str(f) for f in pdf_files], reload_progress)
                            chatbot.document_agent = DocumentRetrievalAgent(chatbot.vectorstore, chatbot.text_splitter, chatbot.llm)
                            progress_bar_reload.empty()
                            status_reload.empty()
                            st.success("✓ Vector database refreshed for next query")
                    st.rerun()

    if not st.session_state["show_question_form"] and st.session_state.get("history"):
        latest = st.session_state["history"][-1]
        st.success("✅ Answer Generated!")
        st.markdown("### Your Question")
        st.info(latest["question"])
        if latest.get("expanded_query") != latest["question"]:
            with st.expander("🔍 Query Expansion Details"):
                st.markdown(f"**Original Query:** {latest['question']}")
                st.markdown(f"**Expanded Query:** {latest['expanded_query']}")
                st.caption("Abbreviations were expanded for better document retrieval")
        if latest.get("retrieval_method"):
            st.caption(f"🔎 Retrieval: **{latest['retrieval_method']}**")
        st.markdown("### Legal Advisor Response")
        st.success(latest["answer"])
        st.markdown("### Response Metrics")
        cols = st.columns(len(latest["metrics"]))
        for col, (k, v) in zip(cols, latest["metrics"].items()):
            col.metric(k, v)
        if latest.get("relevance_reasoning"):
            with st.expander("🧠 Relevance Check Details"):
                st.markdown(latest["relevance_reasoning"])
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        if col1.button("➕ Ask New Question", type="primary", use_container_width=True):
            st.session_state["show_question_form"] = True
            st.rerun()
        if col2.button("📊 View Learning Summary", use_container_width=True):
            with st.expander("📈 RL Learning Summary", expanded=True):
                summary = chatbot.get_learning_summary()
                st.json(summary)
        if col3.button("🧠 View Memory Insights", use_container_width=True):
            with st.expander("🔍 RL Memory Insights", expanded=True):
                insights = chatbot.rl_memory.get_insights()
                st.json(insights)

    st.markdown("---")
    st.subheader("📜 Conversation History")
    if st.session_state.get("history"):
        for idx, item in enumerate(st.session_state["history"]):
            with st.expander(f"Q{idx+1}: {item['question'][:80]}...", expanded=False):
                st.markdown(f"**Question:** {item['question']}")
                if item.get("expanded_query") != item["question"]:
                    st.markdown(f"**Expanded Query:** {item['expanded_query']}")
                if item.get("retrieval_method"):
                    st.markdown(f"**Retrieval Method:** {item['retrieval_method']}")
                st.markdown(f"**Answer:** {item['answer']}")
                if item.get("relevance_reasoning"):
                    st.markdown(f"**Relevance Reasoning:** {item['relevance_reasoning']}")
                st.markdown("**Metrics:**")
                metric_cols = st.columns(len(item["metrics"]))
                for col, (k, v) in zip(metric_cols, item["metrics"].items()):
                    col.metric(k, v)
    else:
        st.info("No questions asked yet. Start a conversation above!")

    st.markdown("---")
    footer_col1, footer_col2 = st.columns(2)
    if footer_col1.button("🔄 Reset Entire Session", use_container_width=True):
        if "chatbot" in st.session_state:
            st.session_state["chatbot"].clear_vectorstore()
        st.session_state.clear()
        st.success("✓ Session reset! Reloading...")
        st.rerun()
    if footer_col2.button("💾 Export Conversation History", use_container_width=True):
        if st.session_state.get("history"):
            history_text = "Legal RAG Chatbot Conversation History\n\n"
            for idx, item in enumerate(st.session_state["history"]):
                history_text += f"Q{idx+1}: {item['question']}\n"
                if item.get("expanded_query") != item["question"]:
                    history_text += f"Expanded: {item['expanded_query']}\n"
                if item.get("retrieval_method"):
                    history_text += f"Retrieval: {item['retrieval_method']}\n"
                history_text += f"Answer: {item['answer']}\n"
                history_text += f"Metrics: {item['metrics']}\n"
                if item.get("relevance_reasoning"):
                    history_text += f"Relevance: {item['relevance_reasoning']}\n"
                history_text += "---\n\n"
            st.download_button(label="Download as Text File", data=history_text, file_name="legal_chatbot_history.txt", mime="text/plain", use_container_width=True)
        else:
            st.warning("No conversation history to export!")







