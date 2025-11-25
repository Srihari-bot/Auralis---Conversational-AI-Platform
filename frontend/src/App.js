import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { 
  FaMicrophone, 
  FaMicrophoneSlash, 
  FaTrash, 
  FaPlay,
  FaStop,
  FaRobot,
  FaUser,
  FaVolumeUp,
  FaVolumeMute
} from 'react-icons/fa';
import './App.css';

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [conversationHistory, setConversationHistory] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [continuousMode, setContinuousMode] = useState(false);
  
  const { transcript, listening, resetTranscript } = useSpeechRecognition();
  const conversationEndRef = useRef(null);
  
  // Speech synthesis using browser's built-in API
  const [speechSynthesis, setSpeechSynthesis] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  
  // Speech detection state
  const [isProcessingSpeech, setIsProcessingSpeech] = useState(false);
  const [speechTimeout, setSpeechTimeout] = useState(null);
  const [lastTranscript, setLastTranscript] = useState('');
  
  // Interruption handling
  const [currentRequest, setCurrentRequest] = useState(null);
  const abortControllerRef = useRef(null);
  const currentAudioRef = useRef(null);
  
  // Language support
  const [currentLanguage, setCurrentLanguage] = useState('en-IN');
  const [availableVoices, setAvailableVoices] = useState([]);
  
  // Sarvam TTS Voice selection
  const [selectedVoice, setSelectedVoice] = useState('vidya');
  
  // Audio Quality selection (speech_sample_rate)
  const [audioQuality, setAudioQuality] = useState('8000'); // Default 22.05kHz
  
  // Removed MediaRecorder - now using browser STT for all modes

  // Auto-scroll to bottom of conversation
  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversationHistory]);

  // Initialize speech synthesis and load voices
  useEffect(() => {
    if ('speechSynthesis' in window) {
      setSpeechSynthesis(window.speechSynthesis);
      
      // Load available voices
      const loadVoices = () => {
        const voices = window.speechSynthesis.getVoices();
        setAvailableVoices(voices);
      };
      
      // Load voices immediately if available
      loadVoices();
      
      // Listen for voices to be loaded
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }
  }, []);

  // Check authentication status on component mount
  useEffect(() => {
    checkAuthStatus();
  }, []);

  // Handle speech recognition with intelligent pause detection and interruption
  useEffect(() => {
    if (transcript && isListening) {
      const cleanTranscript = transcript.trim();
      
      // Only process if transcript is not empty and has changed
      if (cleanTranscript && cleanTranscript !== lastTranscript) {
        setLastTranscript(cleanTranscript);
        
        // Only interrupt TTS if transcript is meaningful (at least 3 characters)
        // This prevents accidental interruptions from background noise
        if (cleanTranscript.length >= 3 && isSpeaking) {
          console.log('🛑 User interrupting TTS with speech:', cleanTranscript);
          stopSpeaking();
          
          // If there's an ongoing request, cancel it
          if (abortControllerRef.current) {
            abortControllerRef.current.abort();
          }
        }
        
        // Clear any existing timeout
        if (speechTimeout) {
          clearTimeout(speechTimeout);
        }
        
        // Set a timeout to wait for more speech (2 seconds)
        const timeout = setTimeout(() => {
          if (cleanTranscript.length >= 3) {
            setIsProcessingSpeech(true);
            handleVoiceInput(cleanTranscript);
            resetTranscript();
            setLastTranscript('');
            setIsProcessingSpeech(false);
          }
        }, 2000); // Wait 2 seconds after last speech input
        
        setSpeechTimeout(timeout);
      }
    }
  }, [transcript, isListening, isProcessingSpeech, lastTranscript, speechTimeout, isSpeaking]);

  const checkAuthStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/status`);
      setIsAuthenticated(response.data.authenticated);
      if (response.data.conversation_history) {
        setConversationHistory(response.data.conversation_history);
      }
    } catch (error) {
      console.error('Error checking auth status:', error);
      setIsAuthenticated(false);
    }
  };

  const validateSpeechInput = (text) => {
    const cleanText = text.trim();
    
    // Basic validation - only check length, accept any language
    if (!cleanText || cleanText.length < 3) {
      return { isValid: false, error: 'Speech too short. Please try speaking again.' };
    }
    
    return { isValid: true, text: cleanText };
  };

  const handleVoiceInput = async (text) => {
    // Validate the speech input
    const validation = validateSpeechInput(text);
    
    if (!validation.isValid) {
      setError(validation.error);
      return;
    }
    
    const cleanText = validation.text;
    
    // Check if this is the same as the last message to prevent duplicates
    if (conversationHistory.length > 0 && 
        conversationHistory[conversationHistory.length - 1].content === cleanText) {
      return;
    }
    
    // Detect language from user input - this will be used for TTS response
    const userInputLanguage = detectLanguage(cleanText);
    
    setIsLoading(true);
    setError('');
    // Create a new AbortController for this request
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    setCurrentRequest(abortController);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        messages: conversationHistory,
        user_input: cleanText
      }, {
        signal: abortController.signal
      });

      if (response.data.success) {
        setConversationHistory(response.data.conversation_history);
        
        // Get the full AI response text - NO TRUNCATION
        const aiResponseText = response.data.response;
        
        // Get detected language from backend (which matches user's input language)
        // Backend now returns detected_language that was detected from user input
        const detectedLanguageFromBackend = response.data.detected_language || userInputLanguage;
        
        // Log the full response for debugging
        console.log('📝 Full AI Response (COMPLETE):', aiResponseText);
        console.log('📏 Response Length:', aiResponseText?.length || 0);
        console.log('📏 Response Word Count:', aiResponseText?.split(' ').length || 0);
        console.log('🌐 Detected Language from Backend:', detectedLanguageFromBackend);
        
        // Speak the COMPLETE AI response - NO LIMITATIONS
        // Use the detected language from backend which matches user's input language
        // The model will respond in the same language as user input, so TTS should use that language too
        if (aiResponseText && aiResponseText.trim().length > 0) {
          // Send ENTIRE response to TTS - no truncation, no limits
          // Use detected language from backend (which is the language the user spoke)
          const fullTextForTTS = aiResponseText.trim();
          console.log('🔊 Sending FULL text to TTS:', fullTextForTTS.length, 'characters');
          console.log('🔊 Using detected language for TTS:', detectedLanguageFromBackend);
          speakText(fullTextForTTS, detectedLanguageFromBackend);
        }
        
        setSuccess('Message sent successfully!');
        setTimeout(() => setSuccess(''), 3000);
      }
    } catch (error) {
      // Don't show error if request was aborted (interrupted)
      if (error.name !== 'CanceledError' && error.code !== 'ERR_CANCELED') {
        setError(error.response?.data?.detail || 'Error processing voice input');
      }
    } finally {
      setIsLoading(false);
      setCurrentRequest(null);
      abortControllerRef.current = null;
    }
  };

  const startListening = () => {
    setIsListening(true);
    SpeechRecognition.startListening({ continuous: true, language: 'en-US' });
  };

  const stopListening = () => {
    setIsListening(false);
    setContinuousMode(false);
    
    // Stop browser speech recognition if active
    if (SpeechRecognition.browserSupportsSpeechRecognition()) {
      SpeechRecognition.stopListening();
    }
    
    // Process any remaining transcript
    if (lastTranscript && lastTranscript.trim().length >= 3) {
      handleVoiceInput(lastTranscript.trim());
    }
    
    // Clear timeouts and reset
    if (speechTimeout) {
      clearTimeout(speechTimeout);
    }
    resetTranscript();
    setLastTranscript('');
    setIsProcessingSpeech(false);
    
    // Cancel any ongoing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const startContinuousMode = () => {
    // Reset any previous transcript
    resetTranscript();
    setContinuousMode(true);
    setIsListening(true);
    
    // Use browser speech recognition for all modes (including auto-detect)
    // For auto-detect, browser will use its default auto-detection
    const language = currentLanguage === 'auto' ? undefined : currentLanguage;
    SpeechRecognition.startListening({ continuous: true, language });
    console.log('🎤 Using browser speech recognition for:', language || 'auto-detect');
  };

  const clearConversation = async () => {
    try {
      // Stop listening and reset transcript
      if (isListening) {
        // Stop browser speech recognition if active
        if (SpeechRecognition.browserSupportsSpeechRecognition()) {
          SpeechRecognition.stopListening();
        }
        
        setIsListening(false);
        setContinuousMode(false);
      }
      resetTranscript();
      
      // Stop speaking and cancel any ongoing requests
      stopSpeaking();
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
      // Clear conversation history immediately on frontend
      setConversationHistory([]);
      
      // Try to clear on backend, but don't fail if it errors
      try {
        await axios.post(`${API_BASE_URL}/clear-conversation`, {}, {
          timeout: 5000
        });
      } catch (backendError) {
        // Log but don't show error - frontend is already cleared
        console.warn('Backend clear conversation failed, but frontend cleared:', backendError);
      }
      
      setSuccess('Conversation cleared!');
      setTimeout(() => setSuccess(''), 3000);
    } catch (error) {
      console.error('Error clearing conversation:', error);
      // Even if there's an error, clear the frontend conversation
      setConversationHistory([]);
      setError('Error clearing conversation on server, but local conversation cleared');
      setTimeout(() => setError(''), 3000);
    }
  };

  const detectLanguage = (text) => {
    // Simple language detection for English, Hindi, and Tamil
    if (/[\u0900-\u097F]/.test(text)) return 'hi-IN'; // Hindi
    if (/[\u0B80-\u0BFF]/.test(text)) return 'ta-IN'; // Tamil
    return 'en-IN'; // Default to English (Sarvam uses en-IN, not en-US)
  };

  // Convert language code to Sarvam-compatible format
  const convertToSarvamLanguageCode = (langCode) => {
    // Sarvam TTS only accepts -IN format codes
    if (langCode === 'en-US' || langCode === 'en') {
      return 'en-IN';
    }
    // Already in correct format or auto
    if (langCode === 'auto' || langCode === 'hi-IN' || langCode === 'ta-IN') {
      return langCode === 'auto' ? 'en-IN' : langCode;
    }
    // Default to en-IN
    return 'en-IN';
  };

  const speakText = async (text, preferredLanguage = null) => {
    let fallbackTimeout = null;
    let audioElement = null;
    
    try {
      // Ensure text is not empty and is a string
      if (!text || typeof text !== 'string' || text.trim().length === 0) {
        console.warn('⚠️ Empty or invalid text provided to TTS');
        return;
      }
      
      // Clean and prepare text for TTS - NO TRUNCATION, NO LIMITS
      const cleanText = text.trim();
      
      // Log what's being sent to TTS - FULL TEXT
      console.log('🔊 TTS Request - FULL TEXT (NO LIMITS):');
      console.log('   Text Length:', cleanText.length, 'characters');
      console.log('   Word Count:', cleanText.split(' ').length, 'words');
      console.log('   Full Text:', cleanText);
      console.log('   First 200 chars:', cleanText.substring(0, 200));
      console.log('   Last 200 chars:', cleanText.substring(Math.max(0, cleanText.length - 200)));
      
      // Use preferred language if provided (from user input), otherwise detect from response text
      // If currentLanguage is set to a specific language (not auto), use it
      let detectedLanguage;
      if (preferredLanguage) {
        // Use the language detected from user input
        detectedLanguage = preferredLanguage;
      } else if (currentLanguage !== 'auto') {
        // Use the selected language setting
        detectedLanguage = currentLanguage;
      } else {
        // Auto-detect from response text (fallback)
        detectedLanguage = detectLanguage(cleanText);
      }
      
      // Convert to Sarvam-compatible language code
      detectedLanguage = convertToSarvamLanguageCode(detectedLanguage);
      
      setIsSpeaking(true);
      
      // Don't stop listening during TTS in continuous mode
      // This allows user to interrupt if needed, but won't cause accidental interruptions
      // The interruption logic will only trigger on meaningful speech (3+ chars)
      
      // Note: Fallback timeout will be set after audio is loaded with proper duration
      
      // Call backend TTS endpoint with the full cleaned text
      const response = await axios.post(`${API_BASE_URL}/tts`, {
        text: cleanText,
        target_language_code: detectedLanguage,
        speaker: selectedVoice,
        speech_sample_rate: parseInt(audioQuality)
      }, {
        responseType: 'blob'
      });
      
      // Create audio element and play
      const audioBlob = new Blob([response.data], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      audioElement = new Audio(audioUrl);
      currentAudioRef.current = audioElement;
      
      // Wait for audio to be fully loaded before playing
      await new Promise((resolve, reject) => {
        const loadingTimeout = setTimeout(() => {
          if (audioElement.readyState >= 2) { // HAVE_CURRENT_DATA
            console.log('🔊 Audio loaded (timeout fallback)');
            resolve();
          } else {
            reject(new Error('Audio loading timeout'));
          }
        }, 5000);
        
        audioElement.oncanplaythrough = () => {
          clearTimeout(loadingTimeout);
          console.log('🔊 Audio loaded and ready to play');
          console.log(`🔊 Audio duration: ${audioElement.duration.toFixed(2)}s`);
          resolve();
        };
        audioElement.onerror = (error) => {
          clearTimeout(loadingTimeout);
          console.error('❌ Audio loading error:', error);
          reject(error);
        };
      });
      
      // Calculate timeout based on actual audio duration if available, otherwise estimate
      let timeoutDuration;
      if (audioElement.duration && !isNaN(audioElement.duration) && isFinite(audioElement.duration)) {
        // Use actual audio duration + 3 second buffer
        timeoutDuration = (audioElement.duration * 1000) + 3000;
        console.log(`🔊 Using actual audio duration: ${audioElement.duration.toFixed(2)}s, timeout: ${(timeoutDuration / 1000).toFixed(1)}s`);
      } else {
        // Estimate: ~150 words per minute, ~2.5 words per second
        const estimatedDuration = (cleanText.split(' ').length / 2.5) * 1000; // in milliseconds
        timeoutDuration = Math.max(estimatedDuration + 5000, 30000); // At least 30 seconds, or estimated + 5s buffer
        console.log(`🔊 Estimated audio duration: ${(estimatedDuration / 1000).toFixed(1)}s, timeout: ${(timeoutDuration / 1000).toFixed(1)}s`);
      }
      
      // Set fallback timeout based on audio duration (for all modes)
      // This is only a safety net - audio will continue playing until onended fires
      if (fallbackTimeout) {
        clearTimeout(fallbackTimeout);
      }
      fallbackTimeout = setTimeout(() => {
        console.log('⚠️ TTS timeout reached - but audio will continue playing');
        // IMPORTANT: Don't stop the audio or clear the reference
        // Just mark as not speaking to allow new interactions if needed
        // The audio will continue playing and onended will fire when complete
        if (audioElement && !audioElement.ended) {
          console.log(`⚠️ Audio still playing: ${audioElement.currentTime.toFixed(2)}s / ${audioElement.duration.toFixed(2)}s`);
          // Keep audio playing - don't interfere
        }
        // Only mark as not speaking if audio has actually ended
        if (audioElement && audioElement.ended) {
          setIsSpeaking(false);
        }
      }, timeoutDuration);
      
      // Set up audio event handlers - this ensures full audio plays when not interrupted
      audioElement.onended = () => {
        const finalDuration = audioElement.duration || 0;
        const finalTime = audioElement.currentTime || 0;
        console.log('✅ TTS completed - FULL AUDIO PLAYED');
        console.log(`✅ Audio duration: ${finalDuration.toFixed(2)}s`);
        console.log(`✅ Played until: ${finalTime.toFixed(2)}s`);
        console.log(`✅ Text length: ${cleanText.length} characters, ${cleanText.split(' ').length} words`);
        console.log(`✅ FULL TEXT WAS SPOKEN - NO TRUNCATION`);
        setIsSpeaking(false);
        
        // Clear fallback timeout
        if (fallbackTimeout) {
          clearTimeout(fallbackTimeout);
          fallbackTimeout = null;
        }
        
        // Clean up audio URL and ref only after completion
        URL.revokeObjectURL(audioUrl);
        currentAudioRef.current = null;
        
        // Resume listening if in continuous mode
        if (continuousMode) {
          setTimeout(() => {
            console.log('🔄 Resuming speech recognition after TTS completion...');
            const language = currentLanguage === 'auto' ? undefined : currentLanguage;
            if (!isListening) {
              SpeechRecognition.startListening({ continuous: true, language });
              setIsListening(true);
            }
          }, 500);
        }
      };
      
      // Add paused event to track if audio is paused (shouldn't happen unless interrupted)
      audioElement.onpause = () => {
        if (audioElement.currentTime > 0 && !audioElement.ended) {
          console.log('⚠️ Audio was paused - likely due to user interruption');
        }
      };
      
      audioElement.onerror = (error) => {
        console.log('❌ TTS playback error occurred');
        console.error('Audio playback error:', error);
        setIsSpeaking(false);
        
        // Clear fallback timeout
        if (fallbackTimeout) {
          clearTimeout(fallbackTimeout);
          fallbackTimeout = null;
        }
        
        // Clean up audio URL and ref
        URL.revokeObjectURL(audioUrl);
        currentAudioRef.current = null;
        
        // Resume listening if in continuous mode
        if (continuousMode) {
          setTimeout(() => {
            const language = currentLanguage === 'auto' ? undefined : currentLanguage;
            if (!isListening) {
              SpeechRecognition.startListening({ continuous: true, language });
              setIsListening(true);
            }
          }, 500);
        }
      };
      
      // Add progress tracking to ensure full audio plays
      audioElement.addEventListener('timeupdate', () => {
        if (audioElement.duration && !isNaN(audioElement.duration)) {
          const progress = (audioElement.currentTime / audioElement.duration) * 100;
          const remaining = audioElement.duration - audioElement.currentTime;
          if (progress > 0 && progress < 100) {
            console.log(`🔊 TTS playing: ${progress.toFixed(1)}% complete (${remaining.toFixed(1)}s remaining)`);
          }
        }
      });
      
      // Verify audio duration matches expected text length
      if (audioElement.duration && !isNaN(audioElement.duration)) {
        const expectedDuration = (cleanText.split(' ').length / 2.5); // ~2.5 words per second
        console.log(`🔊 Audio duration: ${audioElement.duration.toFixed(2)}s`);
        console.log(`🔊 Expected duration: ${expectedDuration.toFixed(2)}s`);
        if (audioElement.duration < expectedDuration * 0.7) {
          console.warn('⚠️ Audio duration seems shorter than expected - may be truncated!');
        }
      }
      
      // Play the audio and wait for it to start
      await audioElement.play();
      console.log('🔊 Audio playback started - FULL TEXT WILL BE SPOKEN');
      
    } catch (error) {
      console.error('🔊 TTS API error:', error);
      setIsSpeaking(false);
      
      // Clear fallback timeout
      if (fallbackTimeout) {
        clearTimeout(fallbackTimeout);
        fallbackTimeout = null;
      }
      
      setError('Failed to generate speech. Please try again.');
      setTimeout(() => setError(''), 3000);
      
      // Resume listening if in continuous mode
      if (continuousMode && isListening) {
        setTimeout(() => {
          const language = currentLanguage === 'auto' ? undefined : currentLanguage;
          SpeechRecognition.startListening({ continuous: true, language });
        }, 500);
      }
    }
  };

  const speakMessage = (text) => {
    // For manual playback, detect language from text
    const textLanguage = currentLanguage === 'auto' ? detectLanguage(text) : currentLanguage;
    speakText(text, textLanguage);
  };

  const submitCurrentSpeech = () => {
    if (lastTranscript && lastTranscript.trim().length >= 3 && !isProcessingSpeech) {
      setIsProcessingSpeech(true);
      handleVoiceInput(lastTranscript.trim());
      resetTranscript();
      setLastTranscript('');
      setIsProcessingSpeech(false);
      
      // Clear timeout
      if (speechTimeout) {
        clearTimeout(speechTimeout);
      }
    }
  };

  const stopSpeaking = () => {
    // Stop Sarvam AI audio if playing - this is ONLY called when user interrupts
    if (currentAudioRef.current) {
      console.log('🛑 User interrupted TTS playback');
      console.log(`🛑 Stopped at: ${currentAudioRef.current.currentTime.toFixed(2)}s / ${currentAudioRef.current.duration.toFixed(2)}s`);
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      // Clean up the audio URL if it exists
      if (currentAudioRef.current.src) {
        try {
          URL.revokeObjectURL(currentAudioRef.current.src);
        } catch (e) {
          // Ignore cleanup errors
        }
      }
      currentAudioRef.current = null;
    }
    
    setIsSpeaking(false);
    
    // Resume listening if in continuous mode
    if (continuousMode) {
      setTimeout(() => {
        const language = currentLanguage === 'auto' ? undefined : currentLanguage;
        if (!isListening) {
          SpeechRecognition.startListening({ continuous: true, language });
          setIsListening(true);
        }
      }, 300);
    }
  };

  return (
    <div className="app-container">
      <div className="main-container">
        <header className="header">
          <h1 className="title">Tally Voice Agent</h1>
        </header>

        <div className="content">
          {/* Left Sidebar */}
          <div className="left-sidebar">
            <div className={`status-indicator ${isAuthenticated ? 'authenticated' : 'not-authenticated'}`}>
              {isAuthenticated ? 'Connected' : 'Not Connected'}
            </div>
            
            <div className="status-info">
            </div>

            <div className="language-selector">
              <label htmlFor="language-select">🌐 Speech Recognition: </label>
              <select 
                id="language-select"
                value={currentLanguage}
                onChange={(e) => setCurrentLanguage(e.target.value)}
                disabled={isListening}
              >
                <option value="auto">Auto Detect</option>
                <option value="en-IN">English (India)</option>
                <option value="hi-IN">Hindi (India)</option>
                <option value="ta-IN">Tamil (India)</option>
              </select>
            </div>

            {error && <div className="error-message">{error}</div>}
            {success && <div className="success-message">{success}</div>}

            <div className="voice-controls">
              <button
                className="btn btn-primary"
                onClick={startContinuousMode}
                disabled={!isAuthenticated || isListening}
              >
                {isLoading ? <div className="loading-spinner" /> : <FaMicrophone />}
                Start Continuous Voice Chat
              </button>
              <button
                className="btn btn-secondary"
                onClick={stopListening}
                disabled={!isListening}
              >
                <FaStop />
                Stop Voice Chat
              </button>
              <button 
                className="btn btn-success" 
                onClick={submitCurrentSpeech}
                disabled={!isListening || !lastTranscript || isProcessingSpeech}
              >
                <FaPlay />
                Send Current Speech
              </button>
              <button className="btn btn-danger" onClick={clearConversation}>
                <FaTrash />
                Clear Conversation
              </button>
            </div>

            {/* Speech Status Indicator */}
            {isListening && (
              <div className="speech-status">
                {lastTranscript ? (
                  <div className="speech-detected">
                    <div className="speech-indicator">
                      <FaMicrophone style={{ color: '#28a745', animation: 'pulse 1s infinite' }} />
                      {isSpeaking ? 'Interrupting bot...' : 'Detecting speech...'} (Auto-send in 2s)
                      {currentLanguage === 'auto' && (
                        <span className="auto-detect-badge">🌐 Auto</span>
                      )}
                    </div>
                    <div className="current-transcript">
                      "{lastTranscript}"
                    </div>
                  </div>
                ) : (
                  <div className="speech-waiting">
                    <FaMicrophone style={{ color: '#6c757d' }} />
                    Waiting for speech...
                    {currentLanguage === 'auto' && (
                      <span className="auto-detect-badge">🌐 Auto</span>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Center - Conversation Area */}
          <div className="conversation-container">
            {conversationHistory.length === 0 ? (
              <div className="empty-state">
                <FaRobot size={48} className="empty-state-icon" />
                <p>No conversation yet. Start by clicking 'Start Voice Chat' or speaking!</p>
              </div>
            ) : (
              conversationHistory.map((message, index) => (
                <div className="message-container" key={index}>
                  {message.role === 'user' ? (
                    <>
                      <div style={{ flex: 1 }}></div>
                      <div className="message-bubble user-message">
                        <div>{message.content}</div>
                      </div>
                      <div className="avatar user-avatar">
                        <FaUser />
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="avatar assistant-avatar">
                        <FaRobot />
                      </div>
                      <div className="message-bubble assistant-message">
                        <div>{message.content}</div>
                      </div>
                    </>
                  )}
                </div>
              ))
            )}
            
            {/* Interruption indicator */}
            {isListening && lastTranscript && isSpeaking && (
              <div className="message-container">
                <div className="avatar assistant-avatar">
                  <FaRobot />
                </div>
                <div className="message-bubble assistant-message interruption-indicator">
                  <div>🔄 Interrupting current response...</div>
                </div>
              </div>
            )}
            
            <div ref={conversationEndRef} />
          </div>

          {/* Right Sidebar - Voice Selection and Audio Quality */}
          <div className="right-sidebar">
            <div className="voice-selector">
              <label htmlFor="voice-select">🔊 Voice: </label>
              <select 
                id="voice-select"
                value={selectedVoice}
                onChange={(e) => setSelectedVoice(e.target.value)}
                disabled={isSpeaking}
              >
                <option value="anushka">Anushka</option>
                <option value="vidya">Vidya</option>
                <option value="manisha">Manisha</option>
                <option value="abhilash">Abhilash</option>
                <option value="karun">Karun</option>
                <option value="arya">Arya</option>
                <option value="hitesh">Hitesh</option>
              </select>
            </div>
            
            <div className="voice-selector">
              <label htmlFor="audio-quality-select">🎵 Audio Quality: </label>
              <select 
                id="audio-quality-select"
                value={audioQuality}
                onChange={(e) => setAudioQuality(e.target.value)}
                disabled={isSpeaking}
              >
                <option value="8000">8kHz</option>
                <option value="16000">16kHz</option>
                <option value="22050">22.05kHz (default)</option>
                <option value="24000">24kHz</option>
              </select>
            </div>
        </div>
      </div>
    </div>
    </div>
  );
}

export default App;