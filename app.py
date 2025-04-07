import os
import sys
import streamlit as st
import time
import importlib.util
import threading
import logging
import numpy as np
from io import BytesIO
import tempfile
import av
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure global error handling
error_messages = []

# Import utility modules
try:
    from utils.document_processor import DocumentProcessor
except ImportError as e:
    error_messages.append(f"Error importing document processor: {str(e)}")

try:
    from utils.vector_store import VectorStore
except ImportError as e:
    error_messages.append(f"Error importing vector store: {str(e)}")

try:
    from utils.llm_service import get_llm_response
except ImportError as e:
    error_messages.append(f"Error importing LLM service: {str(e)}")

try:
    from utils.auto_update import start_file_watcher
except ImportError as e:
    error_messages.append(f"Error importing auto update: {str(e)}")

try:
    from utils.auth import AuthManager
except ImportError as e:
    error_messages.append(f"Error importing authentication manager: {str(e)}")

# Set page config
st.set_page_config(
    page_title="Banking Procedures Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database connection if available
db_manager = None
try:
    from utils.database import DatabaseManager
    db_manager = DatabaseManager()
    db_available = True
    st.session_state.db_available = True
    logger.info("Database connection initialized")
except Exception as e:
    logger.error(f"Error initializing database connection: {str(e)}")
    db_available = False
    st.session_state.db_available = False

# Initialize authentication manager
auth_manager = AuthManager()

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "last_update" not in st.session_state:
    st.session_state.last_update = None
if "file_watcher_started" not in st.session_state:
    st.session_state.file_watcher_started = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "recent_queries" not in st.session_state:
    st.session_state.recent_queries = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None
if "presentation_mode" not in st.session_state:
    st.session_state.presentation_mode = False
if "key_points" not in st.session_state:
    st.session_state.key_points = []

# Function to extract key points from documents
def extract_key_points(query="important concepts", num_points=5):
    """
    Extract key points from the documents for presentation.
    
    Args:
        query: The query to use for finding relevant sections (default: "important concepts")
        num_points: Number of key points to extract (default: 5)
        
    Returns:
        List of key points with sources
    """
    try:
        if not st.session_state.vector_store:
            return []
            
        # Find relevant document chunks
        relevant_docs = st.session_state.vector_store.search(query, k=num_points)
        
        key_points = []
        for i, doc in enumerate(relevant_docs):
            # Extract document name from filepath
            doc_name = os.path.basename(doc.get('filepath', 'Unknown'))
            
            # Create a key point with source
            key_point = {
                'content': doc.get('content', '').strip()[:200] + "...",  # Limit length
                'source': f"{doc_name} (p. {i+1})"
            }
            key_points.append(key_point)
            
        return key_points
    except Exception as e:
        logger.error(f"Error extracting key points: {str(e)}")
        return []

# Function to process documents and update vector store
def process_documents():
    with st.spinner("Processing documents..."):
        st.session_state.processing = True
        try:
            # Initialize document processor
            doc_processor = DocumentProcessor()
            
            # Process documents from data directory and store in database
            documents = doc_processor.process_documents("data")
            
            if not documents:
                # Try loading from database if no documents in directory
                if db_available and db_manager:
                    try:
                        logger.info("No documents in data directory. Trying to load from database...")
                        db_documents = doc_processor.get_documents_from_database()
                        if db_documents:
                            documents = db_documents
                            logger.info(f"Loaded {len(documents)} documents from database")
                        else:
                            st.warning("No documents found in the data directory or database.")
                            st.session_state.processing = False
                            return
                    except Exception as e:
                        logger.error(f"Error loading documents from database: {str(e)}")
                        st.warning("No documents found in the data directory.")
                        st.session_state.processing = False
                        return
                else:
                    st.warning("No documents found in the data directory.")
                    st.session_state.processing = False
                    return
            
            # Initialize vector store with BERT-inspired embeddings
            vector_store = VectorStore(embedding_dim=100, use_bert=True)
            
            # Try to load from database first if available
            if db_available and hasattr(vector_store, 'db_available') and vector_store.db_available:
                loaded_from_db = vector_store.load_from_database()
                if loaded_from_db:
                    logger.info("Successfully loaded vector store from database")
                else:
                    # If loading from database fails, add documents to vector store
                    vector_store.add_documents(documents)
                    logger.info("Added documents to vector store with BERT-inspired embeddings")
            else:
                # Add documents to vector store if database not available
                vector_store.add_documents(documents)
                logger.info("Added documents to vector store with BERT-inspired embeddings")
            
            # Update session state
            st.session_state.vector_store = vector_store
            st.session_state.documents_loaded = True
            st.session_state.last_update = time.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.processing = False
            
            # Success message
            st.success(f"Successfully processed {len(documents)} documents.")
        except Exception as e:
            logger.error(f"Error in process_documents: {str(e)}")
            st.error(f"Error processing documents: {str(e)}")
            st.session_state.processing = False

# Function to handle document updates
def handle_update():
    st.info("New documents detected. Updating knowledge base...")
    process_documents()
    st.rerun()

# Function to handle login
def login(username, password):
    success, role = auth_manager.authenticate(username, password)
    if success:
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.role = role
        return True
    return False

# Function to handle logout
def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.chat_history = []

# Audio recorder class for capturing voice input
class AudioRecorder(AudioProcessorBase):
    """
    Records audio from the user's microphone using WebRTC.
    Formats the audio data for use with Google Speech Recognition.
    """
    def __init__(self):
        self.audio_chunks = []
        self.recording = True
        self.sample_rate = 16000  # Sample rate for speech recognition
        self.sample_width = 2     # Sample width in bytes (16-bit)

    def recv(self, frame):
        """Process a frame of audio."""
        if self.recording:
            # Add audio chunk to list
            self.audio_chunks.append(frame)
        return frame

    def get_audio_data(self):
        """Get the recorded audio data as bytes in WAV format for speech recognition."""
        if not self.audio_chunks:
            return None
        
        try:
            # Combine all audio chunks into a single numpy array
            import numpy as np
            import wave
            from io import BytesIO
            
            # Concatenate audio data from chunks
            audio_data = np.concatenate([chunk.to_ndarray() for chunk in self.audio_chunks])
            
            # Convert to int16 (required for speech recognition)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Create a WAV file in memory
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Get the WAV bytes
            wav_buffer.seek(0)
            return wav_buffer.read()
                
        except Exception as e:
            logger.error(f"Error converting audio data: {e}")
            return None

# Function to transcribe audio to text
def audio_to_text(audio_bytes):
    """
    Process recorded audio and convert it to text using Google Web Speech API.
    
    Args:
        audio_bytes: The audio data as bytes
        
    Returns:
        Transcribed text
    """
    try:
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Use speech recognition to transcribe audio
        recognizer = sr.Recognizer()
        
        try:
            with sr.AudioFile(tmp_path) as source:
                # Adjust for ambient noise and record
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.record(source)
                
                # Try Google Web Speech API (doesn't require API key)
                text = recognizer.recognize_google(audio_data)
                logger.info("Successfully transcribed audio using Google Web Speech API")
                return text
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand audio")
            return "Could not understand audio. Please try speaking more clearly."
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            return "Error connecting to speech recognition service. Please try again later."
    except Exception as e:
        logger.error(f"Error in audio transcription: {str(e)}")
        return "Error transcribing audio. Please try again."
    finally:
        # Clean up the temporary file
        try:
            os.remove(tmp_path)
        except:
            pass

# Function to process a query and display the response
def process_query(query):
    """
    Process a user query and display the AI response.
    
    Args:
        query: The user query text
    """
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.vector_store:
                # Retrieve relevant documents
                relevant_docs = st.session_state.vector_store.search(query, k=4)
                
                # Get response from LLM
                response, sources = get_llm_response(query, relevant_docs)
                
                # Format response with citations
                full_response = f"{response}\n\n"
                
                if sources:
                    full_response += "**Sources:**\n"
                    for i, source in enumerate(sources):
                        # Make sources more readable
                        if isinstance(source, str):
                            # Clean up file paths for display
                            source_display = source.replace("data/", "").replace(".txt", "").replace(".pdf", "").replace(".docx", "")
                            # Capitalize and format for better readability
                            source_display = source_display.replace("_", " ").title()
                            full_response += f"{i+1}. {source_display}\n"
                        else:
                            full_response += f"{i+1}. Document {i+1}\n"
                
                st.markdown(full_response)
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            else:
                st.error("Vector store not initialized. Please process documents first.")

# Start the file watcher if not already started
watcher_available = 'start_file_watcher' in globals() and not any(err.startswith("Error importing auto update") for err in error_messages)
if watcher_available and not st.session_state.file_watcher_started:
    try:
        watcher_thread = threading.Thread(
            target=start_file_watcher, 
            args=("data", handle_update),
            daemon=True
        )
        watcher_thread.start()
        st.session_state.file_watcher_started = True
    except Exception as e:
        error_messages.append(f"Error starting file watcher: {str(e)}")

# Process documents automatically on app startup
if not st.session_state.documents_loaded and not st.session_state.processing:
    # Automatically process documents on startup for all users
    process_documents()

# Main app layout
st.title("Banking Procedures Assistant 🏦")

# Display error messages if any
if error_messages:
    with st.expander("❌ Error Messages", expanded=True):
        st.error(
            "The following errors occurred during application startup:\n\n" +
            "\n".join([f"• {err}" for err in error_messages])
        )

# User authentication UI
if not st.session_state.authenticated:
    st.markdown("### Login")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if login(username, password):
                    st.success(f"Welcome, {username}! You are logged in as {st.session_state.role}.")
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")
    
    with col2:
        st.info("""
        ## Welcome to the Banking Procedures Assistant
        
        This system helps retrieve information from banking procedure documents. 
        
        ### Default credentials:
        - Username: admin
        - Password: admin123
        
        Regular users can only query the database, while admins can also add and manage documents.
        """)
    
else:
    # Sidebar
    with st.sidebar:
        st.markdown(f"### Logged in as: {st.session_state.username}")
        st.markdown(f"**Role**: {st.session_state.role}")
        
        if st.button("Logout"):
            logout()
            st.rerun()
        
        st.markdown("---")
        
        # Admin-only controls
        if st.session_state.role == "admin":
            st.header("Admin Controls")
            
            with st.expander("User Management", expanded=False):
                # User management form
                with st.form("add_user_form"):
                    st.subheader("Add New User")
                    new_username = st.text_input("Username")
                    new_password = st.text_input("Password", type="password")
                    new_role = st.selectbox("Role", ["user", "admin"])
                    add_user_button = st.form_submit_button("Add User")
                    
                    if add_user_button:
                        if new_username and new_password:
                            success = auth_manager.add_user(new_username, new_password, new_role)
                            if success:
                                st.success(f"Added user: {new_username} with role: {new_role}")
                            else:
                                st.error(f"User {new_username} already exists")
                        else:
                            st.warning("Please provide both username and password")
            
            st.header("Document Management")
            
            # File upload section
            with st.expander("Upload Documents", expanded=True):
                st.write("Upload PDF or DOCX files containing banking procedures:")
                uploaded_files = st.file_uploader("Choose files", 
                                                 type=["pdf", "docx"], 
                                                 accept_multiple_files=True)
                
                if uploaded_files:
                    files_saved = False
                    for uploaded_file in uploaded_files:
                        # Create data directory if it doesn't exist
                        if not os.path.exists("data"):
                            os.makedirs("data")
                        
                        # Save uploaded file to data directory
                        file_path = os.path.join("data", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.success(f"Saved file: {uploaded_file.name}")
                        files_saved = True
                    
                    # Automatically process documents after upload
                    if files_saved:
                        st.info("Processing uploaded documents...")
                        process_documents()
            
            if st.button("Process Documents", disabled=st.session_state.processing):
                process_documents()
            
            if st.session_state.documents_loaded:
                st.success("Documents loaded successfully!")
                st.info(f"Last update: {st.session_state.last_update}")
                
                # Key points presentation feature
                with st.expander("Extract Key Points for Presentation", expanded=False):
                    st.subheader("Generate Key Points from Documents")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        topic = st.text_input("Topic or Concept", value="important concepts")
                    with col2:
                        num_points = st.slider("Number of Points", min_value=3, max_value=10, value=5)
                    
                    if st.button("Extract Key Points"):
                        with st.spinner("Extracting key points..."):
                            # Extract key points
                            st.session_state.key_points = extract_key_points(topic, num_points)
                            if st.session_state.key_points:
                                st.success(f"Successfully extracted {len(st.session_state.key_points)} key points.")
                                st.session_state.presentation_mode = True
                            else:
                                st.error("Could not extract key points. Please process documents first.")
        
        # Both admin and user can see recent queries
        if db_available and db_manager:
            with st.expander("Recent Queries", expanded=False):
                try:
                    recent_queries = db_manager.get_recent_queries(limit=5)
                    if recent_queries:
                        for query in recent_queries:
                            st.markdown(f"**Q:** {query['query_text']}")
                            st.markdown(f"**A:** {query['response_text'][:100]}...")
                            st.markdown("---")
                    else:
                        st.info("No recent queries found.")
                except Exception as e:
                    logger.error(f"Error loading recent queries: {str(e)}")
                    st.error("Could not load recent queries from database.")
        
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("""
        1. Admins: Upload banking documents (PDF, DOCX) using the upload feature
        2. Documents are processed automatically upon startup and when new files are added
        3. All users: Ask questions about banking procedures in the main panel
        4. Use voice input to speak your questions directly
        5. Admins: Use 'Extract Key Points' to create presentations
        """)
        
        st.markdown("---")
        st.markdown("### Example Questions:")
        st.markdown("""
        - What is the KYC process?
        - How do I handle a customer dispute?
        - What's the procedure for account closure?
        - Explain the loan approval workflow
        """)

    # Main content area
    if st.session_state.role == "admin":
        # Admin can see document status and query interface
        if not st.session_state.documents_loaded:
            with st.spinner("Loading document database..."):
                process_documents()
        else:
            # Check if in presentation mode
            if st.session_state.presentation_mode and st.session_state.key_points:
                # Display presentation mode
                st.markdown("## 🎯 Key Points for Presentation")
                
                # Display a button to exit presentation mode
                if st.button("Exit Presentation Mode"):
                    st.session_state.presentation_mode = False
                    st.rerun()
                
                # Display key points in a clean format
                st.markdown("---")
                
                for i, point in enumerate(st.session_state.key_points):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"### Point {i+1}")
                        st.markdown(point['content'])
                    with col2:
                        st.markdown(f"**Source:**  \n{point['source']}")
                    st.markdown("---")
                
                # Add export options
                st.subheader("Presentation Options")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Copy to Clipboard"):
                        # Create clipboard text
                        clipboard_text = "# KEY POINTS FOR PRESENTATION\n\n"
                        for i, point in enumerate(st.session_state.key_points):
                            clipboard_text += f"## Point {i+1}\n"
                            clipboard_text += f"{point['content']}\n"
                            clipboard_text += f"Source: {point['source']}\n\n"
                        
                        # Set to clipboard (this is just a visual confirmation - actual clipboard requires JS)
                        st.code(clipboard_text, language="markdown")
                        st.info("Copy the text above to your clipboard for use in presentations.")
                
                with col2:
                    st.download_button(
                        label="Download as Text",
                        data="\n\n".join([f"Point {i+1}: {point['content']}\nSource: {point['source']}" 
                                         for i, point in enumerate(st.session_state.key_points)]),
                        file_name="key_points.txt",
                        mime="text/plain"
                    )
            else:
                st.success("Documents loaded and ready for queries.")
                # Display chat interface
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Voice input section
                st.markdown("### 🎤 Voice Input")
                
                # Initialize recorder session state if not exists
                if "audio_recorder_state" not in st.session_state:
                    st.session_state.audio_recorder_state = {"is_recording": False, "has_recording": False}
                
                # Add WebRTC audio recorder
                recorder_available = True
                audio_recorder = AudioRecorder()
                
                webrtc_ctx = webrtc_streamer(
                    key="admin-voice-recorder",
                    mode=WebRtcMode.SENDONLY,
                    audio_processor_factory=lambda: audio_recorder,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"audio": True, "video": False},
                )
                
                # Check if recording is active
                if webrtc_ctx.state.playing:
                    st.session_state.audio_recorder_state["is_recording"] = True
                    st.warning("Recording audio... Speak your query.")
                elif st.session_state.audio_recorder_state["is_recording"]:
                    # Recording just stopped
                    st.session_state.audio_recorder_state["is_recording"] = False
                    st.session_state.audio_recorder_state["has_recording"] = True
                    st.success("Recording complete! Click 'Process' to transcribe your query.")
                
                # Process recorded audio
                if st.session_state.audio_recorder_state["has_recording"] and st.button("Process Voice Query"):
                    with st.spinner("Transcribing audio..."):
                        # Get audio data from recorder
                        audio_data = audio_recorder.get_audio_data()
                        
                        if audio_data:
                            # Transcribe audio to text
                            voice_query = audio_to_text(audio_data)
                            if voice_query and voice_query != "Error transcribing audio. Please try again.":
                                # Add user message to chat history
                                st.session_state.chat_history.append({"role": "user", "content": voice_query})
                                
                                # Display user message
                                with st.chat_message("user"):
                                    st.markdown(f"🎤 {voice_query}")
                                
                                # Set the query to process
                                query = voice_query
                                # Process the query
                                process_query(query)
                            else:
                                st.error("Could not transcribe your voice. Please try typing your query.")
                                query = None
                
                # Text input (regular chat input)
                if query := st.chat_input("Ask about banking procedures..."):
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": query})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(query)
                    
                    # Process the query
                    process_query(query)
    else:
        # Regular users can only query the system
        # Documents are already loaded on startup, show feedback to the user
        if st.session_state.documents_loaded:
            st.success("Banking procedures database loaded and ready for queries.")
        
        # Display chat interface
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Voice input section
        st.markdown("### 🎤 Voice Input")
        
        # Initialize recorder session state if not exists
        if "user_audio_recorder_state" not in st.session_state:
            st.session_state.user_audio_recorder_state = {"is_recording": False, "has_recording": False}
        
        # Add WebRTC audio recorder
        recorder_available = True
        user_audio_recorder = AudioRecorder()
        
        user_webrtc_ctx = webrtc_streamer(
            key="user-voice-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=lambda: user_audio_recorder,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False},
        )
        
        # Check if recording is active
        if user_webrtc_ctx.state.playing:
            st.session_state.user_audio_recorder_state["is_recording"] = True
            st.warning("Recording audio... Speak your query.")
        elif st.session_state.user_audio_recorder_state["is_recording"]:
            # Recording just stopped
            st.session_state.user_audio_recorder_state["is_recording"] = False
            st.session_state.user_audio_recorder_state["has_recording"] = True
            st.success("Recording complete! Click 'Process' to transcribe your query.")
        
        # Process recorded audio
        if st.session_state.user_audio_recorder_state["has_recording"] and st.button("Process Voice Query", key="user_process_voice"):
            with st.spinner("Transcribing audio..."):
                # Get audio data from recorder
                audio_data = user_audio_recorder.get_audio_data()
                
                if audio_data:
                    # Transcribe audio to text
                    voice_query = audio_to_text(audio_data)
                    if voice_query and voice_query != "Error transcribing audio. Please try again.":
                        # Add user message to chat history
                        st.session_state.chat_history.append({"role": "user", "content": voice_query})
                        
                        # Display user message
                        with st.chat_message("user"):
                            st.markdown(f"🎤 {voice_query}")
                        
                        # Set the query to process
                        query = voice_query
                        # Process the query
                        process_query(query)
                    else:
                        st.error("Could not transcribe your voice. Please try typing your query.")
                        query = None
        
        # Text input (regular chat input)
        if query := st.chat_input("Ask about banking procedures..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            
            # Process the query
            process_query(query)
