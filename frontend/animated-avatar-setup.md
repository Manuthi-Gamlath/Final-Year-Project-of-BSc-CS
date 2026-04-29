# How to Add Animated Avatar to SELA

## Option 1: D-ID Streaming API (Recommended)

D-ID provides real-time streaming avatars with lip-sync and facial animation.

### Setup Steps:

1. **Get API Key**
   - Go to https://studio.d-id.com/
   - Sign up for free account
   - Get your API key from dashboard

2. **Install D-ID SDK**
   Add to your HTML before closing `</body>`:
   ```html
   <script src="https://cdn.jsdelivr.net/npm/@d-id/client-sdk@0.2.0/dist/index.js"></script>
   ```

3. **Replace Static Image with Video Stream**
   In `index.html`, replace the avatar img with:
   ```html
   <video id="avatarVideo" autoplay playsinline></video>
   ```

4. **Add D-ID Integration Code**
   Create new file `avatar-animation.js`:
   ```javascript
   const DID_API_KEY = 'your-api-key-here';
   let streamingClient;

   async function initializeAvatar() {
       const response = await fetch('https://api.d-id.com/talks/streams', {
           method: 'POST',
           headers: {
               'Authorization': `Basic ${DID_API_KEY}`,
               'Content-Type': 'application/json'
           },
           body: JSON.stringify({
               source_url: 'avatar.png' // Your avatar image
           })
       });
       
       const data = await response.json();
       streamingClient = new DID.StreamingClient({
           streamId: data.id,
           sessionId: data.session_id
       });
       
       const videoElement = document.getElementById('avatarVideo');
       await streamingClient.connect(videoElement);
   }

   async function speakWithAnimation(text) {
       await streamingClient.say(text);
   }
   ```

### Cost: Free tier available (up to 10 minutes/month)

---

## Option 2: Synthesia API (Professional Quality)

### Setup:
1. Sign up at https://www.synthesia.io/
2. Get API access (paid plans only)
3. Use their API to generate talking videos
4. Stream to your avatar box

### Cost: Starts at $30/month

---

## Option 3: Wav2Lip (Open Source, Local)

Run lip-sync locally without API costs.

### Setup:

1. **Install Python Dependencies**
   ```bash
   pip install torch opencv-python librosa
   git clone https://github.com/Rudrabha/Wav2Lip.git
   cd Wav2Lip
   ```

2. **Download Pre-trained Model**
   ```bash
   wget 'https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?download=1' -O 'checkpoints/wav2lip.pth'
   ```

3. **Create Python Backend**
   ```python
   # backend.py
   from flask import Flask, request, send_file
   import subprocess

   app = Flask(__name__)

   @app.route('/animate', methods=['POST'])
   def animate_avatar():
       audio_file = request.files['audio']
       audio_file.save('temp_audio.wav')
       
       # Run Wav2Lip
       subprocess.run([
           'python', 'inference.py',
           '--checkpoint_path', 'checkpoints/wav2lip.pth',
           '--face', 'avatar.png',
           '--audio', 'temp_audio.wav',
           '--outfile', 'result.mp4'
       ])
       
       return send_file('result.mp4')

   if __name__ == '__main__':
       app.run(port=5000)
   ```

4. **Update Frontend to Use Backend**
   ```javascript
   async function speakWithAnimation(text) {
       // Generate audio from text
       const utterance = new SpeechSynthesisUtterance(text);
       // ... record audio to blob
       
       // Send to backend
       const formData = new FormData();
       formData.append('audio', audioBlob);
       
       const response = await fetch('http://localhost:5000/animate', {
           method: 'POST',
           body: formData
       });
       
       const videoBlob = await response.blob();
       const videoUrl = URL.createObjectURL(videoBlob);
       
       // Play animated video
       const videoElement = document.getElementById('avatarVideo');
       videoElement.src = videoUrl;
       videoElement.play();
   }
   ```

### Cost: Free (but requires GPU for real-time performance)

---

## Option 4: Simple CSS Animation (Quick Solution)

Add basic animations without external services.

### Add to `styles.css`:
```css
#avatarImage {
    animation: subtle-movement 3s ease-in-out infinite;
}

#avatarImage.speaking {
    animation: speaking-animation 0.3s ease-in-out infinite;
}

@keyframes subtle-movement {
    0%, 100% { transform: translateY(0) scale(1); }
    50% { transform: translateY(-5px) scale(1.02); }
}

@keyframes speaking-animation {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.03); }
}
```

### Update `app.js`:
```javascript
const avatarImage = document.getElementById('avatarImage');

// When speaking starts
avatarImage.classList.add('speaking');

// When speaking ends
avatarImage.classList.remove('speaking');
```

### Cost: Free (but not realistic lip-sync)

---

## Recommendation for Your Project:

**For Final Year Project:**
- **Development/Testing**: Use Option 4 (CSS Animation) - quick and free
- **Demo/Presentation**: Use Option 1 (D-ID) - professional quality, free tier available
- **Production**: Consider Option 3 (Wav2Lip) - no ongoing costs, full control

**Best Balance**: Start with D-ID API for impressive demos, then implement Wav2Lip for the final system.

---

## Implementation Priority:

1. ✅ Static avatar (Done)
2. 🔄 Basic CSS animation (Quick win)
3. 🎯 D-ID integration (For demos)
4. 🚀 Wav2Lip (Final implementation)

Would you like me to implement Option 4 (CSS Animation) right now as a quick solution?
