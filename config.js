// SELA Configuration File
// Update these settings for your project

const SELA_CONFIG = {
    // D-ID API Configuration
    // Get your API key from: https://studio.d-id.com/
    DID_API_KEY: 'bWFudXNlbmFuZ2FAZ21haWwuY29t:ULFDGqNXkfZA1a7G5V1p3',  // Your D-ID API key
    
    // Avatar Settings
    AVATAR_IMAGE: 'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=800&h=1000&fit=crop&crop=faces',  // Temporary online avatar
    
    // Voice Settings
    VOICE_PROVIDER: 'microsoft',
    VOICE_ID: 'en-US-JennyNeural',  // Female voice
    // Other options: 'en-US-GuyNeural' (Male), 'en-GB-SoniaNeural' (British Female)
    
    // Emotion Expression Intensity
    EXPRESSION_INTENSITY: 0.8,  // 0.0 to 1.0
    
    // Speech Settings
    SPEECH_RATE: 0.95,
    SPEECH_PITCH: 1.05,
    
    // Feature Flags
    USE_ANIMATED_AVATAR: true,  // Set to false to use static image
    USE_EMOTION_DETECTION: true,
    USE_PERSONALITY_INFERENCE: true,
    
    // Debug Mode
    DEBUG_MODE: true  // Set to false in production
};

// Export configuration
window.SELA_CONFIG = SELA_CONFIG;
