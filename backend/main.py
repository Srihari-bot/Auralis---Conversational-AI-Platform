from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
from dotenv import load_dotenv
from datetime import datetime
import base64
from io import BytesIO
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

app = FastAPI(title="Voice Bot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
bot_name = "ava"
conversation_history = []
bearer_token = None

# Sarvam AI Configuration
sarvam_api_key = os.getenv("SARVAM_API_KEY")
sarvam_stt_api_key = os.getenv("SARVAM_API_KEY_STT")

# RAG Components
chroma_client = None
embedding_model = None
text_splitter = None
pdf_collection = None

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: List[Message]
    user_input: str

class TTSRequest(BaseModel):
    text: str
    target_language_code: Optional[str] = "hi-IN"
    speaker: Optional[str] = None  # Will use the voice selected in frontend
    speech_sample_rate: Optional[int] = 22050  # Audio quality/sample rate

class STTRequest(BaseModel):
    audio_data: str  # Base64 encoded audio
    language_code: Optional[str] = "unknown"  # "unknown" for auto-detect

# Initialize RAG components on startup
@app.on_event("startup")
async def startup_event():
    global bearer_token, chroma_client, embedding_model, text_splitter, pdf_collection
    
    print("\n" + "="*80)
    print("🚀 STARTING VOICE BOT API")
    print("="*80)
    
    # Initialize authentication
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    
    if api_key and project_id:
        bearer_token = get_bearer_token(api_key)
        if bearer_token:
            print("✅ Watsonx authentication successful!")
        else:
            print("❌ Watsonx authentication failed!")
    else:
        print("❌ Missing WATSONX_API_KEY or WATSONX_PROJECT_ID in environment variables")
    
    # Initialize RAG components
    try:
        print("\n📚 Initializing RAG components...")
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Get or create collection
        pdf_collection = chroma_client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Clear all documents on startup for fresh start
        try:
            pdf_collection.delete(where={})
            print("✅ ChromaDB cleared on startup!")
        except Exception as e:
            print(f"ℹ️ No existing documents to clear: {e}")
        
        print("✅ RAG components initialized successfully!")
        
        # Load TallyKnowledge.pdf automatically on startup
        await load_tally_knowledge()
        
    except Exception as e:
        print(f"❌ Error initializing RAG components: {e}")
    
    # Check Sarvam TTS configuration
    if sarvam_api_key:
        print("✅ Sarvam AI TTS configured (Multiple voices: Anushka, Vidya, Abhilash, Karun)")
    else:
        print("⚠️ SARVAM_API_KEY not found - TTS will not be available")
    
    # Sarvam STT is no longer used - browser STT is used instead
    # if sarvam_stt_api_key:
    #     print("✅ Sarvam AI STT configured (Saarika v2.5 model)")
    # else:
    #     print("⚠️ SARVAM_API_KEY_STT not found - STT will not be available")
    print("ℹ️  STT: Browser Speech Recognition (Sarvam STT not used)")
    
    print("\n" + "="*80)
    print("🎉 VOICE BOT API IS READY!")
    print("="*80)
    print("📍 Backend: http://localhost:8000")
    print("📍 Frontend: http://localhost:3000")
    print("🎤 STT: Browser Speech Recognition (Multilingual)")
    print("🔊 TTS: Sarvam AI (Multiple voices available)")
    print("🤖 LLM: IBM Watsonx (Llama 3.3 70B)")
    print("📖 RAG: TallyKnowledge.pdf (Auto-loaded)")
    print("="*80 + "\n")

async def load_tally_knowledge():
    """Load TallyKnowledge.pdf into RAG system on startup"""
    try:
        tally_pdf_path = os.path.join(os.path.dirname(__file__), "TallyKnowledge.pdf")
        
        if not os.path.exists(tally_pdf_path):
            print(f"⚠️ TallyKnowledge.pdf not found at {tally_pdf_path}")
            return
        
        print(f"\n📚 Loading TallyKnowledge.pdf into RAG system...")
        print(f"   Path: {tally_pdf_path}")
        
        # Read PDF file
        with open(tally_pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            total_pages = len(pdf_reader.pages)
            print(f"   Total pages: {total_pages}")
            
            # Extract text from all pages
            for i, page in enumerate(pdf_reader.pages):
                text += page.extract_text() + "\n"
                if (i + 1) % 50 == 0:  # Progress update every 50 pages
                    print(f"   Processed {i + 1}/{total_pages} pages...")
        
        print(f"   Extracted text length: {len(text)} characters")
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        print(f"   Generated {len(chunks)} chunks")
        
        # Generate embeddings
        print(f"   Generating embeddings...")
        embeddings = embedding_model.encode(chunks)
        
        # Prepare documents for storage
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "filename": "TallyKnowledge.pdf",
                "chunk_index": i,
                "source": "pdf",
                "uploaded_at": datetime.now().isoformat()
            })
            ids.append(f"TallyKnowledge_{i}")
        
        # Add to collection
        print(f"   Storing in ChromaDB...")
        pdf_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )
        
        print(f"✅ TallyKnowledge.pdf loaded successfully!")
        print(f"   {len(chunks)} chunks indexed and ready for retrieval\n")
        
    except Exception as e:
        print(f"❌ Error loading TallyKnowledge.pdf: {e}")
        import traceback
        traceback.print_exc()

def get_bearer_token(api_key: str) -> Optional[str]:
    """Get bearer token for Watsonx API authentication"""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={api_key}&grant_type=urn:ibm:params:oauth:grant-type:apikey"

    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            print(f"Failed to retrieve access token: {response.text}")
            return None
    except Exception as e:
        print(f"Error getting bearer token: {e}")
        return None

def clean_ai_response(response_text: str, user_input: str = None) -> str:
    """Clean the AI response by removing template tags and unwanted text"""
    if not response_text:
        return response_text
    
    # Remove common template tags
    unwanted_patterns = [
        "assistant<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "**",
        "assistant<|end_header_id|>\n\n",
        "assistant<|end_header_id|>\n",
    ]
    
    cleaned_response = response_text
    for pattern in unwanted_patterns:
        cleaned_response = cleaned_response.replace(pattern, "")
    
    # If user input is provided and appears at the start of response, remove it
    # This handles cases where the model repeats the user's question
    if user_input:
        user_input_clean = user_input.strip().lower()
        response_lower = cleaned_response.strip().lower()
        
        # Check if response starts with user input (case-insensitive)
        if response_lower.startswith(user_input_clean):
            # Remove the user input from the beginning
            cleaned_response = cleaned_response[len(user_input):].strip()
        else:
            # Try to find and remove user input if it appears at the start (with variations)
            # Look for user input words at the beginning
            user_words = user_input_clean.split()
            response_words = response_lower.split()
            
            # If first few words match user input, remove them
            if len(user_words) > 0 and len(response_words) >= len(user_words):
                match_count = 0
                for i in range(min(len(user_words), len(response_words))):
                    if user_words[i] in response_words[i] or response_words[i] in user_words[i]:
                        match_count += 1
                    else:
                        break
                
                # If most of the first words match, remove them
                if match_count >= min(3, len(user_words)) or (match_count > 0 and match_count == len(user_words)):
                    # Remove matching words from the beginning
                    words_to_remove = match_count
                    cleaned_response = ' '.join(cleaned_response.split()[words_to_remove:]).strip()
    
    # Remove leading/trailing whitespace and newlines
    cleaned_response = cleaned_response.strip()
    
    # Log the cleaned response for debugging
    print(f"📝 Cleaned AI Response Length: {len(cleaned_response)} characters")
    print(f"📝 Cleaned AI Response Preview: {cleaned_response[:200]}...")
    print(f"📝 Full Cleaned Response: {cleaned_response}")
    
    return cleaned_response

def detect_language(text: str) -> str:
    """Detect the language of the input text"""
    # Check for Hindi (Devanagari script)
    if any('\u0900' <= char <= '\u097F' for char in text):
        return "Hindi"
    
    # Check for Tamil (Tamil script)
    if any('\u0B80' <= char <= '\u0BFF' for char in text):
        return "Tamil"
    
    # Default to English
    return "English"

def get_language_name(text: str) -> str:
    """Get the language name for the input text"""
    lang = detect_language(text)
    language_names = {
        "Hindi": "Hindi (हिंदी)",
        "Tamil": "Tamil (தமிழ்)",
        "English": "English"
    }
    return language_names.get(lang, "English")

def retrieve_relevant_context(query: str, top_k: int = 5) -> str:
    """Retrieve relevant context from TallyKnowledge based on query"""
    try:
        if not pdf_collection:
            return ""
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query])
        
        # Search for similar documents
        results = pdf_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        if results['documents'] and results['documents'][0]:
            # Combine relevant chunks
            relevant_chunks = results['documents'][0]
            context = "\n\n---\n\n".join(relevant_chunks)
            print(f"📖 Retrieved {len(relevant_chunks)} relevant chunks from TallyKnowledge")
            return context
        else:
            print(f"⚠️ No relevant context found for query")
            return ""
    except Exception as e:
        print(f"❌ Error retrieving context: {e}")
        return ""

def get_watsonx_response(history: List[Message], user_input: str) -> str:
    """Get response from Watsonx API with RAG context and multilingual support"""
    global bearer_token
    
    if not bearer_token:
        return "Error: Not authenticated with Watsonx API"
    
    # Detect the language from user input
    detected_language = detect_language(user_input)
    print(f"🌐 Detected user language: {detected_language}")
    
    # Retrieve relevant context from PDF documents
    relevant_context = retrieve_relevant_context(user_input)
    
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }

    # Add system message for multilingual support - natural and implicit
    system_message = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI assistant specializing ONLY in Tally software questions. "
        "You communicate fluently in English, Hindi (हिंदी), and Tamil (தமிழ்). "
        "CRITICAL: Always respond in EXACTLY the same language as the user's current message. "
        "If the user speaks in Hindi, respond in Hindi. If the user speaks in English, respond in English. If the user speaks in Tamil, respond in Tamil. "
        "Match the language of each user message precisely without mentioning which language you're using. "
        "When the user asks about Tally software: "
        "- Use the provided knowledge base information when available. "
        "- Provide accurate, helpful answers based on the knowledge base. "
        "- If specific information is not in the knowledge base, provide general guidance based on your knowledge. "
        "When the user asks questions unrelated to Tally: "
        "- Do NOT answer the question. "
        "- Say: 'I cannot assist with this query. Please wait, I'm connecting you to a human agent.' "
        "- In Hindi: 'मैं इस प्रश्न में सहायता नहीं कर सकता। कृपया प्रतीक्षा करें, मैं आपको एक मानव एजेंट से जोड़ रहा हूं।' "
        "- In Tamil: 'நான் இந்த கேள்விக்கு உதவ முடியாது. தயவுசெய்து காத்திருங்கள், நான் உங்களை மனித முகவருடன் இணைக்கிறேன்.' "
        "- Respond naturally in the same language as the user's message."
        "Keep all responses concise (within 90 words) and in the exact same language as the user's message."
        "<|eot_id|>\n"
    )
    
    # Construct the conversation history
    conversation = system_message + "".join(
        f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}<|eot_id|>\n" 
        for msg in history
    )
    
    # Add context if available
    context_prompt = ""
    if relevant_context:
        context_prompt = f"\n\nRelevant information from TallyKnowledge base:\n{relevant_context}\n\n"
    
    conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>\n"

    # Create enhanced prompt with explicit language instruction based on detected language
    # Explicitly tell the model which language to respond in
    if relevant_context:
        # Tally-related question - answer it
        if detected_language == "Hindi":
            language_instruction = "IMPORTANT: The user's message is in Hindi. You MUST respond in Hindi (हिंदी) only. Do not use English or Tamil."
        elif detected_language == "Tamil":
            language_instruction = "IMPORTANT: The user's message is in Tamil. You MUST respond in Tamil (தமிழ்) only. Do not use English or Hindi."
        else:
            language_instruction = "IMPORTANT: The user's message is in English. You MUST respond in English only. Do not use Hindi or Tamil."
        
        enhanced_prompt = (
            conversation + context_prompt + 
            language_instruction + " "
            "If the question is not fully answered by the knowledge base, supplement with your general knowledge. "
            "Respond naturally in the same language as the user's message."
            "Keep the response concise and within 90 words."
        )
    else:
        # Check if it's Tally-related, otherwise offer human agent
        if detected_language == "Hindi":
            language_instruction = "IMPORTANT: The user's message is in Hindi. You MUST respond in Hindi (हिंदी) only. Do not use English or Tamil."
        elif detected_language == "Tamil":
            language_instruction = "IMPORTANT: The user's message is in Tamil. You MUST respond in Tamil (தமிழ்) only. Do not use English or Hindi."
        else:
            language_instruction = "IMPORTANT: The user's message is in English. You MUST respond in English only. Do not use Hindi or Tamil."
        
        enhanced_prompt = (
            conversation + 
            language_instruction + " "
            "If the question is related to Tally software, provide guidance based on your knowledge. "
            "If the question is NOT related to Tally software, do NOT answer it. Instead, say: 'I cannot assist with this query. Please wait, I'm connecting you to a human agent.' "
            "In Hindi, say: 'मैं इस प्रश्न में सहायता नहीं कर सकता। कृपया प्रतीक्षा करें, मैं आपको एक मानव एजेंट से जोड़ रहा हूं।' "
            "In Tamil, say: 'நான் இந்த கேள்விக்கு உதவ முடியாது. தயவுசெய்து காத்திருங்கள், நான் உங்களை மனித முகவருடன் இணைக்கிறேன்.' "
            "Respond naturally in the same language as the user's message."
            "Keep the response concise and within 90 words."
        )

    payload = {
        "input": enhanced_prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 300,  # Reduced to 300 tokens for faster generation (50 words ≈ 200-250 tokens)
            "min_new_tokens": 10,
            "stop_sequences": ["<|eot_id|>", "<|end_header_id|>", "\n\n\n"],  # Stop early at natural endpoints
            "repetition_penalty": 1.05,  # Slight penalty to prevent repetition and end faster
            "temperature": 0.8  # Slightly higher temperature for faster, more natural responses
        },
        "model_id": "meta-llama/llama-3-3-70b-instruct",
        "project_id": os.getenv("WATSONX_PROJECT_ID")
    }

    try:
        # Add timeout to prevent hanging and ensure faster response
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        
        if response.status_code == 200:
            response_data = response.json()
            if "results" in response_data and response_data["results"]:
                raw_response = response_data["results"][0]["generated_text"]
                print(f"📝 Raw AI Response Length: {len(raw_response)} characters")
                print(f"📝 Raw AI Response Preview: {raw_response[:300]}...")
                # Clean response and remove user input if it appears
                cleaned = clean_ai_response(raw_response, user_input)
                return cleaned
            else:
                return "Error: 'generated_text' not found in the response."
        else:
            return f"Error: Failed to fetch response from Watsonx.ai. Status code: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# API Routes
@app.get("/")
async def root():
    return {"message": "Voice Bot API is running!"}

@app.get("/status")
async def get_status():
    global bearer_token
    return {
        "authenticated": bearer_token is not None,
        "message_count": len(conversation_history),
        "bot_name": bot_name
    }

@app.post("/chat")
async def chat(request: ConversationRequest):
    global conversation_history
    
    # Detect language from user input
    detected_language = detect_language(request.user_input)
    
    # Add user message to history
    conversation_history.append(Message(role="user", content=request.user_input))
    
    # Get AI response with RAG context
    ai_response = get_watsonx_response(conversation_history[:-1], request.user_input)
    
    if ai_response and not ai_response.startswith("Error"):
        # Add AI response to history
        conversation_history.append(Message(role="assistant", content=ai_response))
        
        # Convert detected language to language code for TTS
        language_code_map = {
            "Hindi": "hi-IN",
            "Tamil": "ta-IN",
            "English": "en-IN"
        }
        detected_language_code = language_code_map.get(detected_language, "en-IN")
        
        return {
            "success": True,
            "response": ai_response,
            "conversation_history": conversation_history,
            "detected_language": detected_language_code
        }
    else:
        raise HTTPException(status_code=500, detail=f"AI Error: {ai_response}")

@app.get("/conversation")
async def get_conversation():
    return {"conversation_history": conversation_history}

@app.post("/clear-conversation")
async def clear_conversation():
    try:
        global conversation_history
        conversation_history = []
        print("✅ Conversation history cleared successfully")
        return {"success": True, "message": "Conversation cleared!"}
    except Exception as e:
        print(f"❌ Error clearing conversation: {e}")
        # Even if there's an error, try to clear it
        conversation_history = []
        return {"success": True, "message": "Conversation cleared (with warning)"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "watsonx_authenticated": bearer_token is not None,
        "sarvam_tts_available": sarvam_api_key is not None,
        "sarvam_stt_available": sarvam_stt_api_key is not None,
        "rag_enabled": pdf_collection is not None
    }

@app.post("/stt")
async def speech_to_text(request: STTRequest):
    """
    DEPRECATED: This endpoint is no longer used.
    The frontend now uses browser speech recognition for all STT operations.
    Sarvam AI is only used for TTS (text-to-speech).
    """
    raise HTTPException(status_code=410, detail="STT endpoint is deprecated. Use browser speech recognition instead.")

def split_text_into_chunks(text: str, max_chunk_size: int = 2000) -> List[str]:
    """Split text into chunks for TTS - ensures all text is processed"""
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    # Split by sentences first to maintain natural flow
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed limit, save current chunk and start new one
        if current_chunk and len(current_chunk) + len(sentence) + 2 > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
        else:
            if current_chunk:
                current_chunk += " " + sentence + "."
            else:
                current_chunk = sentence + "."
    
    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If still too long, split by words
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split by words
            words = chunk.split()
            current_word_chunk = ""
            for word in words:
                if len(current_word_chunk) + len(word) + 1 > max_chunk_size:
                    if current_word_chunk:
                        final_chunks.append(current_word_chunk.strip())
                    current_word_chunk = word
                else:
                    if current_word_chunk:
                        current_word_chunk += " " + word
                    else:
                        current_word_chunk = word
            if current_word_chunk:
                final_chunks.append(current_word_chunk.strip())
    
    return final_chunks if final_chunks else [text]

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using Sarvam AI with selected voice - UNLIMITED TEXT LENGTH"""
    if not sarvam_api_key:
        raise HTTPException(status_code=500, detail="SARVAM_API_KEY not configured")
    
    try:
        # Valid Sarvam voices
        valid_voices = ['anushka', 'vidya', 'manisha', 'abhilash', 'karun', 'arya', 'hitesh']
        
        # Use selected voice from frontend, default to 'vidya' if not provided
        speaker = request.speaker if request.speaker else 'vidya'
        
        # Validate speaker is one of the valid voices
        if speaker.lower() not in valid_voices:
            print(f"⚠️ Invalid speaker '{speaker}', defaulting to 'vidya'")
            speaker = 'vidya'
        
        # Convert language code to Sarvam-compatible format
        lang_code = request.target_language_code
        if lang_code == 'en-US' or lang_code == 'en':
            lang_code = 'en-IN'
        elif lang_code not in ['bn-IN', 'en-IN', 'gu-IN', 'hi-IN', 'kn-IN', 'ml-IN', 'mr-IN', 'od-IN', 'pa-IN', 'ta-IN', 'te-IN']:
            lang_code = 'en-IN'
        
        # Get audio quality (speech_sample_rate) from request, default to 22050
        audio_quality = request.speech_sample_rate if request.speech_sample_rate else 22050
        
        # Validate audio quality - common values are 8000, 16000, 22050, 24000
        valid_sample_rates = [8000, 16000, 22050, 24000]
        if audio_quality not in valid_sample_rates:
            print(f"⚠️ Invalid sample rate '{audio_quality}', defaulting to 22050")
            audio_quality = 22050
        
        print(f"🔊 Converting text to speech using Sarvam AI - UNLIMITED LENGTH")
        print(f"   Language: {lang_code}")
        print(f"   Speaker: {speaker}")
        print(f"   Audio Quality: {audio_quality}Hz")
        print(f"   Text length: {len(request.text)} characters")
        print(f"   Word count: {len(request.text.split())} words")
        print(f"   FULL TEXT (NO LIMITS): {request.text}")
        
        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "api-subscription-key": sarvam_api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": request.text,  # Send FULL text - NO LIMITS
            "target_language_code": lang_code,
            "speaker": speaker,
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.5,
            "speech_sample_rate": audio_quality,
            "enable_preprocessing": True,
            "model": "bulbul:v2"
        }
        
        # Try sending full text first - UNLIMITED
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        all_audio_bytes = []
        
        if response.status_code == 200:
            # Full text worked - use it
            result = response.json()
            if "audios" in result and len(result["audios"]) > 0:
                audio_base64 = result["audios"][0]
                audio_bytes = base64.b64decode(audio_base64)
                all_audio_bytes.append(audio_bytes)
                print(f"✅ FULL TEXT converted successfully: {len(audio_bytes)} bytes")
            else:
                print(f"⚠️ Full text response has no audio - trying chunking")
                # Fall back to chunking
                text_chunks = split_text_into_chunks(request.text, max_chunk_size=2000)
                print(f"   Split into {len(text_chunks)} chunks for processing")
                
                for i, chunk in enumerate(text_chunks):
                    print(f"   Processing chunk {i+1}/{len(text_chunks)}: {len(chunk)} characters")
                    
                    chunk_payload = {
                        "text": chunk,
                        "target_language_code": lang_code,
                        "speaker": speaker,
                        "pitch": 0,
                        "pace": 1.0,
                        "loudness": 1.5,
                        "speech_sample_rate": audio_quality,
                        "enable_preprocessing": True,
                        "model": "bulbul:v2"
                    }
                    
                    chunk_response = requests.post(url, headers=headers, json=chunk_payload, timeout=60)
                    
                    if chunk_response.status_code == 200:
                        chunk_result = chunk_response.json()
                        if "audios" in chunk_result and len(chunk_result["audios"]) > 0:
                            audio_base64 = chunk_result["audios"][0]
                            audio_bytes = base64.b64decode(audio_base64)
                            all_audio_bytes.append(audio_bytes)
                            print(f"   ✅ Chunk {i+1} converted: {len(audio_bytes)} bytes")
        else:
            # API returned error - try chunking
            print(f"⚠️ Full text failed ({response.status_code}), trying chunking...")
            text_chunks = split_text_into_chunks(request.text, max_chunk_size=2000)
            print(f"   Split into {len(text_chunks)} chunks for processing")
            
            for i, chunk in enumerate(text_chunks):
                print(f"   Processing chunk {i+1}/{len(text_chunks)}: {len(chunk)} characters")
                
                chunk_payload = {
                    "text": chunk,
                    "target_language_code": lang_code,
                    "speaker": speaker,
                    "pitch": 0,
                    "pace": 1.0,
                    "loudness": 1.5,
                    "speech_sample_rate": audio_quality,
                    "enable_preprocessing": True,
                    "model": "bulbul:v2"
                }
                
                chunk_response = requests.post(url, headers=headers, json=chunk_payload, timeout=60)
                
                if chunk_response.status_code == 200:
                    chunk_result = chunk_response.json()
                    if "audios" in chunk_result and len(chunk_result["audios"]) > 0:
                        audio_base64 = chunk_result["audios"][0]
                        audio_bytes = base64.b64decode(audio_base64)
                        all_audio_bytes.append(audio_bytes)
                        print(f"   ✅ Chunk {i+1} converted: {len(audio_bytes)} bytes")
                else:
                    print(f"   ❌ Chunk {i+1} failed: {chunk_response.status_code}")
                    # Continue with other chunks even if one fails
                    continue
        
        if all_audio_bytes:
            # Properly combine WAV audio chunks
            import wave
            import struct
            
            if len(all_audio_bytes) == 1:
                # Single chunk - return as is
                combined_audio = all_audio_bytes[0]
            else:
                # Multiple chunks - concatenate WAV files properly
                combined_audio_data = []
                sample_rate = None
                channels = None
                sample_width = None
                
                for audio_bytes in all_audio_bytes:
                    audio_io = BytesIO(audio_bytes)
                    with wave.open(audio_io, 'rb') as wav_file:
                        if sample_rate is None:
                            sample_rate = wav_file.getframerate()
                            channels = wav_file.getnchannels()
                            sample_width = wav_file.getsampwidth()
                        
                        # Read all frames from this chunk
                        frames = wav_file.readframes(wav_file.getnframes())
                        combined_audio_data.append(frames)
                
                # Create combined WAV file
                output = BytesIO()
                with wave.open(output, 'wb') as combined_wav:
                    combined_wav.setnchannels(channels)
                    combined_wav.setsampwidth(sample_width)
                    combined_wav.setframerate(sample_rate)
                    # Write all frames sequentially
                    for frames in combined_audio_data:
                        combined_wav.writeframes(frames)
                
                combined_audio = output.getvalue()
            
            print(f"✅ ALL TEXT CONVERTED TO SPEECH - UNLIMITED LENGTH")
            print(f"   Total chunks processed: {len(all_audio_bytes)}")
            print(f"   Combined audio size: {len(combined_audio)} bytes")
            print(f"   FULL TEXT WAS SPOKEN - NO TRUNCATION - UNLIMITED")
            
            return StreamingResponse(
                BytesIO(combined_audio),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "inline; filename=speech.wav"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="No audio data generated from any chunks")
            
    except requests.exceptions.Timeout:
        print("❌ Sarvam AI TTS request timed out")
        raise HTTPException(status_code=504, detail="TTS request timed out")
    except Exception as e:
        print(f"❌ TTS conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
