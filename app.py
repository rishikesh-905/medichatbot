# app.py (with enhanced source filtering)
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec
import logging
import threading
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("pinecone_plugin_interface").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Global variable for the RAG chain
rag_chain = None
retriever = None
app_initialized = False
init_lock = threading.Lock()

# Primary medical encyclopedia name
MEDICAL_ENCYCLOPEDIA = "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"

# System prompt for the AI with citation requirement
system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Provide accurate, concise information in a professional manner. "
    "For medical questions, always include a disclaimer that you are an AI assistant and not a substitute for professional medical advice. "
    "At the end of your response, include citations mentioning ONLY the source documents that are actually relevant to your answer. "
    "Format citations as: [Source: filename.pdf, page X]"
    "\n\n"
    "{context}"
)

def is_relevant_to_question(source_name, content, question, answer):
    """Check if a source is actually relevant to both the question and the final answer"""
    if not source_name or not content:
        return False
    
    # Always prioritize the main medical encyclopedia for medical questions
    question_lower = question.lower()
    if MEDICAL_ENCYCLOPEDIA.lower() in source_name.lower():
        medical_terms = ["medical", "health", "disease", "symptom", "treatment", 
                        "patient", "doctor", "hospital", "medicine", "drug", "infection",
                        "bacteria", "virus", "condition", "disorder", "illness"]
        if any(term in question_lower for term in medical_terms):
            return True
    
    # Check if the source content contains key terms from both question and answer
    content_lower = content.lower()
    answer_lower = answer.lower()
    
    # Extract key terms from question and answer
    question_terms = set(re.findall(r'\b\w{4,}\b', question_lower))
    answer_terms = set(re.findall(r'\b\w{4,}\b', answer_lower))
    
    # Combine relevant terms
    relevant_terms = question_terms.union(answer_terms)
    
    # Remove common stop words
    stop_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", 
                 "how", "the", "and", "or", "but", "for", "nor", "so", "yet", "with",
                 "about", "above", "across", "after", "against", "along", "among",
                 "around", "at", "before", "behind", "below", "beneath", "beside",
                 "between", "beyond", "by", "down", "during", "except", "for", "from",
                 "in", "inside", "into", "like", "near", "of", "off", "on", "onto",
                 "out", "outside", "over", "past", "since", "through", "throughout",
                 "to", "toward", "under", "until", "up", "upon", "with", "within",
                 "without", "this", "that", "these", "those", "is", "are", "was",
                 "were", "be", "being", "been", "have", "has", "had", "do", "does",
                 "did", "will", "would", "shall", "should", "may", "might", "must",
                 "can", "could", "its", "your", "our", "their", "mine", "yours",
                 "ours", "theirs"}
    
    relevant_terms = relevant_terms - stop_words
    
    # Check if content contains any relevant terms
    if not any(term in content_lower for term in relevant_terms):
        return False
    
    # Additional checks for specific content types
    if "bee" in source_name.lower() and any(term in question_lower for term in ["ohm", "electr", "circuit", "current", "voltage", "resistance"]):
        return True
    
    if "satellite" in source_name.lower() and any(term in question_lower for term in ["satellite", "orbit", "space", "communication"]):
        return True
    
    if "ai" in source_name.lower() or "artificial" in source_name.lower() and any(term in question_lower for term in ["ai", "artificial", "intelligence", "machine", "learning"]):
        return True
    
    if "botany" in source_name.lower() and any(term in question_lower for term in ["plant", "botany", "flower", "leaf", "root"]):
        return True
    
    return True

def initialize_pinecone_index():
    """Initialize Pinecone index if it doesn't exist"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "medical-encyclopedia"
        
        # Check if index exists, create if not
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,  # matches sentence-transformers/all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            time.sleep(10)
            return True, pc
        logger.info(f"Pinecone index {index_name} already exists")
        return False, pc
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        return False, None

def load_and_process_pdfs(pc):
    """Load and process PDF documents"""
    pdf_folder = "research/Data/"
    all_documents = []
    
    # Check if PDF folder exists
    if not os.path.exists(pdf_folder):
        logger.error(f"PDF folder not found: {pdf_folder}")
        return False
    
    # Load all PDFs in the folder
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            logger.info(f"Loading PDF: {pdf_path}")
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                
                # Add metadata to identify source
                for doc in docs:
                    doc.metadata["source"] = file_name
                    # Ensure page metadata is preserved
                    if "page" not in doc.metadata:
                        doc.metadata["page"] = "unknown"
                
                all_documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {file_name}")
            except Exception as e:
                logger.error(f"Error loading {pdf_path}: {str(e)}")
    
    if not all_documents:
        logger.error("No documents were loaded")
        return False   
    
    # Split documents into chunks with better parameters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks for medical concepts
        chunk_overlap=400,  # More overlap for complete concepts
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    text_chunks = splitter.split_documents(all_documents)
    logger.info(f"Split into {len(text_chunks)} text chunks")
    
    # Use the updated HuggingFaceEmbeddings import if available
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        # Fallback to deprecated import
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Upsert to Pinecone in smaller batches to avoid timeouts
    logger.info("Uploading documents to Pinecone...")
    try:
        # Process in smaller batches
        batch_size = 50  # Even smaller batches for reliability
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(text_chunks)-1)//batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name="medical-encyclopedia"
            )
            
            # Small delay between batches
            time.sleep(2)
            
        logger.info("Documents uploaded to Pinecone successfully")
        return True
    except Exception as e:
        logger.error(f"Error uploading to Pinecone: {str(e)}")
        return False

def expand_query(query):
    """Add related terms to improve retrieval"""
    query_expansions = {
        "campylobacteriosis": "campylobacter infection bacterial gastroenteritis food poisoning",
        "epilogue": "conclusion summary ending final remarks outcome results",
        "what is": "define definition explain describe",
        "symptoms": "signs indications manifestations clinical features",
        "treatment": "therapy management intervention medication drugs",
        "causes": "etiology risk factors pathogenesis",
        "diagnosis": "diagnostic tests identification detection",
        "nutrition": "diet food nutrients vitamins minerals health eating",
        "medical": "health disease treatment hospital doctor patient medicine",
        "disease": "infection illness condition disorder syndrome",
        "ohm": "electrical electricity circuit current voltage resistance law",
        "law": "principle rule formula equation",
        "electr": "current voltage power energy circuit",
        "physics": "force motion energy matter",
        "engineering": "design build construct technical"
    }
    
    expanded = query.lower()
    for term, expansion in query_expansions.items():
        if term in expanded:
            expanded += " " + expansion
    
    return expanded

def setup_rag_chain():
    """Set up the RAG chain for question answering"""
    try:
        # Use the updated HuggingFaceEmbeddings import if available
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            # Fallback to deprecated import
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize vector store
        docsearch = PineconeVectorStore(
            index_name="medical-encyclopedia",
            embedding=embeddings
        )
        
        # Create retriever with IMPROVED parameters
        retriever = docsearch.as_retriever(
        search_type="similarity",  # Use basic similarity for focused retrieval
        search_kwargs={
            "k": 1  # Only retrieve the single most relevant document
        }
    )
        
        # Initialize LLM
        llm = Ollama(model="gemma:2b", temperature=0.4)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Create chains (using the standard approach)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain, retriever
    except Exception as e:
        logger.error(f"Error setting up RAG chain: {str(e)}")
        return None, None

def initialize_app():
    """Initialize the application"""
    global rag_chain, app_initialized, retriever
    
    with init_lock:
        if app_initialized:
            return
            
        logger.info("Initializing application...")
        
        # Initialize Pinecone index
        index_created, pc = initialize_pinecone_index()
        
        if pc is None:
            logger.error("Failed to initialize Pinecone")
            return
            
        # If index was just created or we need to process PDFs
        if index_created:
            logger.info("Processing PDF documents...")
            success = load_and_process_pdfs(pc)
            if not success:
                logger.error("Failed to process PDF documents")
        
        # Set up RAG chain
        rag_chain, retriever = setup_rag_chain()
        if rag_chain is None:
            logger.error("Failed to set up RAG chain")
        else:
            logger.info("RAG chain setup completed successfully")
        
        app_initialized = True

@app.before_request
def before_request():
    """Initialize the app before the first request"""
    if not app_initialized:
        # Run initialization in a separate thread to avoid blocking
        thread = threading.Thread(target=initialize_app)
        thread.daemon = True
        thread.start()

@app.route("/")
def home():
    """Render the main chat interface"""
    return render_template("chat.html")

@app.route("/status")
def status():
    """Check application status"""
    return jsonify({
        "initialized": app_initialized,
        "rag_chain_ready": rag_chain is not None
    })

@app.route("/ask", methods=["POST"])
def ask_question():
    """Handle question asking with accurate citations"""
    if not app_initialized:
        return jsonify({"answer": "System is still initializing. Please try again in a moment."})
    
    if rag_chain is None:
        return jsonify({"answer": "System is not ready yet. Please try again in a moment."})
    
    try:
        data = request.get_json()
        question = data.get("question", "")
        
        if not question:
            return jsonify({"answer": "Please provide a question."})
        
        # Expand query with related terms for better retrieval
        expanded_question = expand_query(question)
        logger.info(f"Original question: '{question}', Expanded: '{expanded_question}'")
        
        # Get response from RAG chain
        response = rag_chain.invoke({"input": expanded_question})
        answer = response["answer"]
        
        # Extract and FILTER sources from the retrieved documents
        retrieved_docs = response.get("context", [])
        sources = set()
        
        if retrieved_docs:
            top_doc = retrieved_docs[0]
            source = top_doc.metadata.get("source", "unknown")
            page = top_doc.metadata.get("page", "unknown")
            sources = {f"{source}, page {page}"}

        
        # Add source information to the answer (only if relevant sources found)
        if sources:
            answer += f"\n\n**Sources:** {', '.join(sorted(sources))}"
        else:
            # If no relevant sources found, be honest about it
            answer += f"\n\n*No specific sources identified for this information*"
        
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({"answer": "Sorry, I encountered an error while processing your question."})

@app.route("/debug_retrieval", methods=["POST"])
def debug_retrieval():
    """Debug what documents are being retrieved"""
    if not app_initialized or retriever is None:
        return jsonify({"error": "System not initialized"})
    
    try:
        data = request.get_json()
        question = data.get("question", "")
        
        if not question:
            return jsonify({"error": "Please provide a question."})
        
        # Expand query
        expanded_question = expand_query(question)
        
        # Get the raw retrieved documents
        retrieved_docs = retriever.invoke(expanded_question)
        
        results = []
        for i, doc in enumerate(retrieved_docs):
            results.append({
                "rank": i + 1,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "relevance_score": "N/A"
            })
        
        return jsonify({
            "question": question,
            "expanded_question": expanded_question,
            "retrieved_docs": results
        })
    except Exception as e:
        logger.error(f"Error in debug retrieval: {str(e)}")
        return jsonify({"error": str(e)})

@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Handle PDF uploads"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and file.filename.endswith('.pdf'):
            # Save the file
            upload_folder = "uploads"
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            
            # Process the PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Add source metadata
            for doc in docs:
                doc.metadata["source"] = file.filename
                if "page" not in doc.metadata:
                    doc.metadata["page"] = "unknown"
            
            # Split into chunks with better parameters
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=400,
                separators=["\n\n", "\n", " ", ""],
                length_function=len
            )
            text_chunks = splitter.split_documents(docs)
            
            # Use the updated HuggingFaceEmbeddings import if available
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            except ImportError:
                # Fallback to deprecated import
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Add to Pinecone in smaller batches
            batch_size = 50
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i+batch_size]
                PineconeVectorStore.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    index_name="medical-encyclopedia"
                )
                time.sleep(1)
            
            return jsonify({"message": "PDF uploaded and processed successfully"})
        else:
            return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        return jsonify({"error": "Failed to process PDF"}), 500

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=8080, debug=True)