# Real-Time Human-Like Avatar Implementation Guide

## Overview
This guide shows you how to add a realistic, animated avatar with:
- ✅ Lip-sync (mouth movements matching speech)
- ✅ Facial expressions (happy, sad, neutral, etc.)
- ✅ Natural movements (blinking, head tilts, breathing)
- ✅ Real-time response (no delay)

---

## 🎯 Best Solution for Your Project: D-ID Streaming API

D-ID provides the most realistic real-time avatars with minimal setup.

### Why D-ID?
- Professional quality (used by major companies)
- Real-time streaming (no video generation delay)
- Facial expressions based on emotion
- Easy integration
- Free tier available (10 minutes/month)

---

## 📋 Step-by-Step Implementation

### Step 1: Get D-ID API Access

1. Go to https://studio.d-id.com/
2. Sign up for free account
3. Go to Settings → API Keys
4. Copy your API key
5. Note: Free tier gives you 10 minutes/month for testing

---

### Step 2: Update Your HTML

Replace the avatar section in `index.html`:

```html
<!-- Avatar Container -->
<div class="avatar-container">
    <!-- Animated Avatar Video -->
    <div class="human-avatar">
        <video id="avatarVideo" autoplay playsinline></video>
        <canvas id="avatarCanvas"></canvas>
    </div>
</div>
```

---

### Step 3: Create Avatar Animation Module

Create a new file: `avatar-controller.js`

```javascript
// D-ID Streaming Avatar Controller
class AvatarController {
    constructor(apiKey, avatarImageUrl) {
        this.apiKey = apiKey;
        this.avatarImageUrl = avatarImageUrl;
        this.streamId = null;
        this.sessionId = null;
        this.peerConnection = null;
        this.videoElement = document.getElementById('avatarVideo');
    }

    // Initialize the avatar stream
    async initialize() {
        try {
            // Create a new stream session
            const response = await fetch('https://api.d-id.com/talks/streams', {
                method: 'POST',
                headers: {
                    'Authorization': `Basic ${btoa(this.apiKey + ':')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    source_url: this.avatarImageUrl,
                    driver_url: 'bank://lively'  // Natural idle animation
                })
            });

            const data = await response.json();
            this.streamId = data.id;
            this.sessionId = data.session_id;

            // Setup WebRTC connection
            await this.setupWebRTC(data);
            
            console.log('Avatar initialized successfully');
            return true;
        } catch (error) {
            console.error('Failed to initialize avatar:', error);
            return false;
        }
    }

    // Setup WebRTC for real-time streaming
    async setupWebRTC(streamData) {
        this.peerConnection = new RTCPeerConnection({
            iceServers: streamData.ice_servers || []
        });

        // Handle incoming video stream
        this.peerConnection.ontrack = (event) => {
            if (event.track.kind === 'video') {
                this.videoElement.srcObject = event.streams[0];
            }
        };

        // Create offer
        const offer = await this.peerConnection.createOffer();
        await this.peerConnection.setLocalDescription(offer);

        // Send offer to D-ID
        const response = await fetch(
            `https://api.d-id.com/talks/streams/${this.streamId}/sdp`,
            {
                method: 'POST',
                headers: {
                    'Authorization': `Basic ${btoa(this.apiKey + ':')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    answer: offer,
                    session_id: this.sessionId
                })
            }
        );

        const { answer } = await response.json();
        await this.peerConnection.setRemoteDescription(
            new RTCSessionDescription(answer)
        );
    }

    // Make avatar speak with emotion
    async speak(text, emotion = 'neutral') {
        if (!this.streamId) {
            console.error('Avatar not initialized');
            return;
        }

        try {
            // Map emotions to D-ID expressions
            const expressionMap = {
                'happy': 'happy',
                'sad': 'sad',
                'angry': 'serious',
                'surprised': 'surprised',
                'neutral': 'neutral'
            };

            const response = await fetch(
                `https://api.d-id.com/talks/streams/${this.streamId}`,
                {
                    method: 'POST',
                    headers: {
                        'Authorization': `Basic ${btoa(this.apiKey + ':')}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        script: {
                            type: 'text',
                            input: text,
                            provider: {
                                type: 'microsoft',
                                voice_id: 'en-US-JennyNeural'  // Female voice
                            }
                        },
                        config: {
                            stitch: true,
                            driver_expressions: {
                                expressions: [
                                    {
                                        expression: expressionMap[emotion] || 'neutral',
                                        start_frame: 0,
                                        intensity: 0.8
                                    }
                                ]
                            }
                        },
                        session_id: this.sessionId
                    })
                }
            );

            return await response.json();
        } catch (error) {
            console.error('Failed to make avatar speak:', error);
        }
    }

    // Destroy the stream when done
    async destroy() {
        if (this.streamId) {
            await fetch(
                `https://api.d-id.com/talks/streams/${this.streamId}`,
                {
                    method: 'DELETE',
                    headers: {
                        'Authorization': `Basic ${btoa(this.apiKey + ':')}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        session_id: this.sessionId
                    })
                }
            );
        }

        if (this.peerConnection) {
            this.peerConnection.close();
        }
    }
}

// Export for use in main app
window.AvatarController = AvatarController;
```

---

### Step 4: Update Your Main App.js

Modify `app.js` to use the avatar controller:

```javascript
// Add at the top of app.js
let avatarController = null;

// Replace the initializeApp function
async function initializeApp() {
    // Initialize D-ID Avatar
    const DID_API_KEY = 'YOUR_API_KEY_HERE';  // Replace with your key
    const avatarImagePath = 'avatar.png';
    
    avatarController = new AvatarController(DID_API_KEY, avatarImagePath);
    
    updateStatus('loading', 'Loading avatar...');
    const initialized = await avatarController.initialize();
    
    if (initialized) {
        setTimeout(() => {
            greetUser();
        }, 1500);
    } else {
        updateStatus('error', 'Failed to load avatar');
    }
}

// Replace the speakText function
async function speakText(text) {
    if (!avatarController) {
        console.error('Avatar not initialized');
        return;
    }

    isSpeaking = true;
    updateStatus('speaking', 'Speaking...');
    voiceIndicator.classList.add('active');

    // Get current emotion from emotion detection
    const emotion = emotionData.emotion || 'neutral';

    // Make avatar speak with emotion
    await avatarController.speak(text, emotion);

    // Wait for speech to complete (estimate based on text length)
    const estimatedDuration = (text.length / 15) * 1000; // ~15 chars per second
    
    setTimeout(() => {
        isSpeaking = false;
        voiceIndicator.classList.remove('active');
        startListening();
    }, estimatedDuration);
}

// Add cleanup on page unload
window.addEventListener('beforeunload', async () => {
    if (avatarController) {
        await avatarController.destroy();
    }
});
```

---

### Step 5: Update index.html to Include Avatar Controller

Add before the closing `</body>` tag:

```html
<script src="avatar-controller.js"></script>
<script src="app.js"></script>
```

---

### Step 6: Update CSS for Video Avatar

Add to `styles.css`:

```css
#avatarVideo {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center top;
    display: block;
    border-radius: 20px;
}

#avatarCanvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
}
```

---

## 🎭 Emotion-Based Expressions

The avatar will automatically show expressions based on detected emotions:

| Detected Emotion | Avatar Expression |
|-----------------|-------------------|
| Happy           | Smiling, bright eyes |
| Sad             | Downturned mouth, soft eyes |
| Angry           | Serious, furrowed brow |
| Surprised       | Wide eyes, open mouth |
| Neutral         | Calm, natural expression |

---

## 💰 Cost Breakdown

### D-ID Pricing:
- **Free Tier**: 10 minutes/month (good for testing)
- **Lite Plan**: $5.90/month - 15 minutes
- **Basic Plan**: $29/month - 90 minutes
- **Pro Plan**: Custom pricing

### For Your Project:
- Development: Use free tier
- Demo/Presentation: Upgrade to Lite ($5.90 for one month)
- Final Submission: Record demo videos to save credits

---

## 🔄 Alternative: Open Source Solution (Wav2Lip + SadTalker)

If you want a completely free solution:

### Setup:

1. **Install Dependencies**
```bash
# Install Python packages
pip install torch torchvision opencv-python
pip install face-alignment librosa scipy

# Clone repositories
git clone https://github.com/Rudrabha/Wav2Lip.git
git clone https://github.com/OpenTalker/SadTalker.git
```

2. **Create Python Backend**
```python
# backend_server.py
from flask import Flask, request, send_file
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app)

@app.route('/animate', methods=['POST'])
def animate_avatar():
    text = request.json['text']
    emotion = request.json.get('emotion', 'neutral')
    
    # Generate audio from text using TTS
    # ... (use pyttsx3 or gTTS)
    
    # Generate video with Wav2Lip
    subprocess.run([
        'python', 'Wav2Lip/inference.py',
        '--checkpoint_path', 'checkpoints/wav2lip_gan.pth',
        '--face', 'avatar.png',
        '--audio', 'temp_audio.wav',
        '--outfile', 'result.mp4'
    ])
    
    return send_file('result.mp4', mimetype='video/mp4')

if __name__ == '__main__':
    app.run(port=5000)
```

3. **Update Frontend**
```javascript
async function speakWithAnimation(text, emotion) {
    const response = await fetch('http://localhost:5000/animate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, emotion })
    });
    
    const videoBlob = await response.blob();
    const videoUrl = URL.createObjectURL(videoBlob);
    
    const videoElement = document.getElementById('avatarVideo');
    videoElement.src = videoUrl;
    await videoElement.play();
}
```

**Pros**: Free, full control
**Cons**: Requires GPU, slower (2-3 seconds delay), more complex setup

---

## 📊 Comparison Table

| Feature | D-ID API | Wav2Lip + SadTalker | Current (CSS) |
|---------|----------|---------------------|---------------|
| Setup Time | 30 mins | 2-3 hours | 5 mins |
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Real-time | Yes (<1s) | No (2-3s) | Yes |
| Cost | $5.90/month | Free | Free |
| Lip-sync | Perfect | Good | None |
| Expressions | Yes | Limited | None |
| GPU Required | No | Yes | No |

---

## 🎯 Recommendation for Your Project

**Best Approach:**

1. **Phase 1 (Now)**: Use current CSS animation for development
2. **Phase 2 (Testing)**: Integrate D-ID with free tier
3. **Phase 3 (Demo)**: Upgrade to D-ID Lite for presentation
4. **Phase 4 (Optional)**: Implement Wav2Lip for final submission if budget is tight

**For Impressive Demo**: D-ID is worth the $5.90 for one month during your presentation period.

---

## 🚀 Quick Start (D-ID)

1. Get API key from https://studio.d-id.com/
2. Replace `YOUR_API_KEY_HERE` in the code
3. Create `avatar-controller.js` file
4. Update `app.js` with new functions
5. Test with: "Hello, how are you?"

The avatar will speak with lip-sync and facial expressions!

---

## 📝 Notes for Your Project Report

Include in your documentation:
- "Real-time avatar animation using D-ID Streaming API"
- "Emotion-aware facial expressions synchronized with detected user emotions"
- "WebRTC-based streaming for low-latency interaction"
- "Multimodal output: voice synthesis + visual embodiment"

This aligns perfectly with your SELA project objectives! 🎓
