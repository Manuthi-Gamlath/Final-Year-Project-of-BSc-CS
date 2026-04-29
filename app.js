// Initialize Speech Recognition and Synthesis
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
const synth = window.speechSynthesis;

// Configure Speech Recognition
recognition.continuous = true;
recognition.interimResults = false;
recognition.lang = 'en-US';
recognition.maxAlternatives = 1;

// Avatar Controller
let avatarController = null;

// DOM Elements
const webcamFeed = document.getElementById('webcamFeed');
const faceCanvas = document.getElementById('faceCanvas');
const voiceIndicator = document.getElementById('voiceIndicator');
const statusDot = document.getElementById('statusDot');
const statusMessage = document.getElementById('statusMessage');

// Application State
let isListening = false;
let isSpeaking = false;
let recognitionActive = false;
let webcamStream = null;
let emotionData = { emotion: 'neutral', valence: 0.5, arousal: 0.5 };
let conversationHistory = [];
let requestStartTime = 0;

// Emotion Icons Map
const emotionIcons = {
    neutral: '😐',
    happy: '😊',
    sad: '😢',
    angry: '😠',
    surprised: '😲',
    fearful: '😨',
    disgusted: '🤢'
};

// Initialize Application
window.addEventListener('load', () => {
    initializeWebcam();
    
    // Wait for voices to load
    if (synth.getVoices().length === 0) {
        synth.addEventListener('voiceschanged', initializeApp);
    } else {
        initializeApp();
    }
});

// Initialize Webcam
async function initializeWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 },
            audio: false 
        });
        webcamFeed.srcObject = webcamStream;
        
        // Start emotion detection loop (placeholder for future ML integration)
        startEmotionDetection();
        
        console.log('Webcam initialized');
    } catch (error) {
        console.error('Webcam access denied:', error);
        updateStatus('error', 'Webcam access required');
    }
}

// Emotion Detection Loop (Placeholder for ML model integration)
function startEmotionDetection() {
    setInterval(() => {
        // TODO: Replace with actual emotion detection model
        // MediaPipe Face Mesh + AffectNet model
        detectEmotionFromFrame();
    }, 1000);
}

function detectEmotionFromFrame() {
    // Placeholder: Simulate emotion detection
    // In production, integrate MediaPipe Face Mesh + AffectNet model
    const emotions = ['neutral', 'happy', 'sad', 'surprised'];
    const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];
    
    emotionData.emotion = randomEmotion;
    emotionData.valence = 0.4 + Math.random() * 0.3;
    emotionData.arousal = 0.4 + Math.random() * 0.3;
}

async function initializeApp() {
    // Check if animated avatar is enabled
    if (SELA_CONFIG.USE_ANIMATED_AVATAR) {
        updateStatus('loading', 'Loading avatar...');
        
        try {
            // Initialize D-ID Avatar
            avatarController = new AvatarController(
                SELA_CONFIG.DID_API_KEY,
                SELA_CONFIG.AVATAR_IMAGE
            );
            
            const initialized = await avatarController.initialize();
            
            if (!initialized) {
                console.error('Failed to initialize avatar, falling back to voice only');
                updateStatus('warning', 'Avatar unavailable - using voice only');
                avatarController = null;
            } else {
                console.log('Avatar initialized successfully!');
                updateStatus('ready', 'Avatar ready');
            }
        } catch (error) {
            console.error('Avatar initialization error:', error);
            updateStatus('warning', 'Avatar unavailable - using voice only');
            avatarController = null;
        }
    }
    
    setTimeout(() => {
        greetUser();
    }, 1500);
}

// Greet User on Load
function greetUser() {
    const greeting = "Hello! I'm SELA, your socially-aware assistant. I can see and hear you. How can I help you today?";
    speakText(greeting);
}

// Text-to-Speech Function
async function speakText(text) {
    isSpeaking = true;
    updateStatus('speaking', 'Speaking...');
    voiceIndicator.classList.add('active');

    // Use animated avatar if available
    if (avatarController && SELA_CONFIG.USE_ANIMATED_AVATAR) {
        const emotion = emotionData.emotion || 'neutral';
        await avatarController.speak(text, emotion);
        
        // Estimate speech duration
        const duration = (text.length / 15) * 1000; // ~15 chars per second
        
        setTimeout(() => {
            isSpeaking = false;
            voiceIndicator.classList.remove('active');
            startListening();
        }, duration);
    } else {
        // Fallback to browser TTS
        if (synth.speaking) {
            synth.cancel();
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.95;
        utterance.pitch = 1.05;
        utterance.volume = 1;

        // Select voice
        const voices = synth.getVoices();
        const preferredVoice = voices.find(v => 
            v.name.includes('Female') || 
            v.name.includes('Zira') || 
            v.name.includes('Google') ||
            v.lang.startsWith('en')
        );
        
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }

        utterance.onend = () => {
            isSpeaking = false;
            voiceIndicator.classList.remove('active');
            
            // Start listening after speaking
            setTimeout(() => {
                startListening();
            }, 800);
        };

        utterance.onerror = (event) => {
            console.error('Speech synthesis error:', event);
            isSpeaking = false;
            voiceIndicator.classList.remove('active');
            startListening();
        };

        synth.speak(utterance);
    }
}

// Start Listening
function startListening() {
    if (isListening || isSpeaking || recognitionActive || !isMicOn) return;

    try {
        recognition.start();
        recognitionActive = true;
        isListening = true;
        updateStatus('listening', 'Listening...');
        voiceIndicator.classList.add('active');
    } catch (error) {
        console.log('Recognition start error:', error);
        recognitionActive = false;
    }
}

// Stop Listening
function stopListening() {
    if (!isListening) return;

    try {
        recognition.stop();
    } catch (error) {
        console.log('Recognition stop error:', error);
    }
    
    isListening = false;
    recognitionActive = false;
    voiceIndicator.classList.remove('active');
}

// Update Status Display
function updateStatus(state, message) {
    statusDot.className = `status-dot ${state}`;
    statusMessage.textContent = message;
}

// Speech Recognition Event Handlers
recognition.onstart = () => {
    recognitionActive = true;
    isListening = true;
};

recognition.onresult = (event) => {
    const lastResultIndex = event.results.length - 1;
    const transcript = event.results[lastResultIndex][0].transcript.trim();
    
    if (transcript && transcript.length > 0) {
        console.log('User said:', transcript);
        
        // Add to conversation history
        conversationHistory.push({ role: 'user', content: transcript });
        
        // Stop listening
        stopListening();
        
        // Process and respond
        setTimeout(() => {
            processUserSpeech(transcript);
        }, 600);
    }
};

recognition.onerror = (event) => {
    console.error('Recognition error:', event.error);
    recognitionActive = false;
    
    if (event.error === 'no-speech') {
        isListening = false;
        setTimeout(() => startListening(), 1500);
    } else if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        updateStatus('error', 'Microphone access denied');
        isListening = false;
    } else if (event.error === 'aborted') {
        isListening = false;
        recognitionActive = false;
    } else {
        isListening = false;
        setTimeout(() => startListening(), 2000);
    }
};

recognition.onend = () => {
    recognitionActive = false;
    
    if (isListening && !isSpeaking) {
        setTimeout(() => {
            if (!isSpeaking && !recognitionActive) {
                startListening();
            }
        }, 1000);
    } else {
        isListening = false;
    }
};

// Process User Speech and Generate Response
function processUserSpeech(userInput) {
    updateStatus('thinking', 'Processing...');
    
    // Simulate processing delay
    setTimeout(() => {
        // Generate emotion-aware response
        const response = generateEmotionAwareResponse(userInput, emotionData);
        
        // Add to conversation history
        conversationHistory.push({ role: 'assistant', content: response });
        
        speakText(response);
    }, 700);
}

// Generate Emotion-Aware Response
function generateEmotionAwareResponse(input, emotion) {
    const lower = input.toLowerCase();

    // Greetings
    if (lower.match(/\b(hello|hi|hey|greetings)\b/)) {
        if (emotion.emotion === 'sad') {
            return "Hello. I notice you might be feeling down. I'm here to listen and help in any way I can.";
        } else if (emotion.emotion === 'happy') {
            return "Hi! You seem to be in a great mood! That's wonderful. What can I do for you today?";
        }
        return "Hello! It's wonderful to talk with you. What would you like to know?";
    }

    // How are you
    if (lower.match(/how are you|how're you/)) {
        if (emotion.emotion === 'sad') {
            return "I'm here for you. More importantly, how are you feeling? I'm here to help.";
        }
        return "I'm doing fantastic, thank you for asking! I'm always excited to help. What about you?";
    }

    // Emotional support
    if (lower.match(/sad|depressed|down|upset/)) {
        return "I'm sorry you're feeling this way. Remember, it's okay to feel emotions. I'm here to listen and support you. Would you like to talk about it?";
    }

    if (lower.match(/happy|excited|great|wonderful/)) {
        return "That's fantastic! I'm so glad to hear you're feeling positive. Your energy is contagious! What's making you feel so good?";
    }

    // Name/Identity
    if (lower.match(/your name|who are you|what are you/)) {
        return "I'm SELA - a Socially-aware Embodied Language Agent. I can see your expressions, hear your voice, and adapt my responses to how you're feeling!";
    }

    // Help
    if (lower.match(/\bhelp\b/)) {
        return "I'm here to assist you! I can sense your emotions through your facial expressions and voice, and I adapt my responses accordingly. Just speak naturally!";
    }

    // Time
    if (lower.match(/what time|current time|time is it/)) {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit', 
            hour12: true 
        });
        return `The current time is ${timeStr}.`;
    }

    // Date
    if (lower.match(/what date|today's date|what day/)) {
        const now = new Date();
        const dateStr = now.toLocaleDateString('en-US', { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });
        return `Today is ${dateStr}.`;
    }

    // Thank you
    if (lower.match(/thank you|thanks|appreciate/)) {
        if (emotion.emotion === 'happy') {
            return "You're so welcome! Your gratitude means a lot. I'm thrilled I could help!";
        }
        return "You're very welcome! Happy to help anytime.";
    }

    // Goodbye
    if (lower.match(/bye|goodbye|see you|talk later/)) {
        return "Goodbye! It was wonderful talking with you. Take care, and come back anytime!";
    }

    // Default emotion-aware response
    if (emotion.emotion === 'sad') {
        return `I understand you said: "${input}". I'm here to help you with that. Let's work through this together.`;
    } else if (emotion.emotion === 'happy') {
        return `Great! You mentioned: "${input}". I'd love to help you with that! Let's dive in.`;
    }
    
    return `I heard you say: "${input}". That's interesting! How can I help you with that?`;
}

// Handle Page Visibility
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopListening();
        if (synth.speaking) {
            synth.cancel();
        }
        isSpeaking = false;
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', async () => {
    stopListening();
    if (synth.speaking) {
        synth.cancel();
    }
    if (avatarController) {
        await avatarController.destroy();
    }
});


// Control Button Functionality
let isCameraOn = true;
let isMicOn = true;

// Camera Toggle
const cameraToggle = document.getElementById('cameraToggle');
const webcamContainer = document.querySelector('.webcam-container');

cameraToggle.addEventListener('click', () => {
    isCameraOn = !isCameraOn;
    
    if (isCameraOn) {
        // Turn camera ON
        cameraToggle.classList.remove('off');
        webcamContainer.classList.remove('off');
        
        if (webcamStream) {
            webcamStream.getVideoTracks().forEach(track => track.enabled = true);
        } else {
            initializeWebcam();
        }
    } else {
        // Turn camera OFF
        cameraToggle.classList.add('off');
        webcamContainer.classList.add('off');
        
        if (webcamStream) {
            webcamStream.getVideoTracks().forEach(track => track.enabled = false);
        }
    }
});

// Microphone Toggle
const micToggle = document.getElementById('micToggle');

micToggle.addEventListener('click', () => {
    isMicOn = !isMicOn;
    
    if (isMicOn) {
        // Turn mic ON
        micToggle.classList.remove('off');
        updateStatus('ready', 'Microphone enabled');
        
        // Restart listening if not speaking
        if (!isSpeaking) {
            setTimeout(() => startListening(), 500);
        }
    } else {
        // Turn mic OFF
        micToggle.classList.add('off');
        updateStatus('muted', 'Microphone muted');
        
        // Stop listening
        stopListening();
    }
});


// Display logged-in user name
window.addEventListener('load', () => {
    const currentUser = JSON.parse(localStorage.getItem('selaCurrentUser') || '{}');
    const userNameElement = document.getElementById('userName');
    
    if (currentUser.fullname) {
        userNameElement.textContent = currentUser.fullname;
    }
});
