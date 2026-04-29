# Easy Step-by-Step Guide: Add Animated Avatar to SELA

## 🎯 What You'll Get
- Professional animated avatar with perfect lip-sync
- Facial expressions (happy, sad, surprised, etc.)
- Real-time response (no delay)
- High quality like Synthesia

---

## ⏱️ Total Time: 30 minutes

---

## Step 1: Get Your D-ID Account (5 minutes)

### 1.1 Sign Up
1. Open browser and go to: https://studio.d-id.com/
2. Click "Sign Up" button
3. Enter your email and create password
4. Verify your email (check inbox)

### 1.2 Get API Key
1. Log in to D-ID Studio
2. Click your profile picture (top right)
3. Click "Settings"
4. Click "API Keys" tab
5. Click "Create API Key"
6. Copy the key (looks like: `abc123xyz...`)
7. Save it in Notepad - you'll need it soon!

**Free Tier**: 10 minutes per month (enough for testing)

---

## Step 2: Update Your Files (10 minutes)

### 2.1 Update config.js

1. Open `config.js` file
2. Find this line:
   ```javascript
   DID_API_KEY: 'YOUR_DID_API_KEY_HERE',
   ```
3. Replace `YOUR_DID_API_KEY_HERE` with your actual key:
   ```javascript
   DID_API_KEY: 'abc123xyz...',  // Paste your key here
   ```
4. Save the file (Ctrl+S)

### 2.2 Update index.html

1. Open `index.html`
2. Find this section (around line 24):
   ```html
   <div class="human-avatar">
       <img src="avatar.png" id="avatarImage" alt="SELA Avatar" 
            onerror="this.style.display='none'; this.parentElement.classList.add('no-image');" />
   ```

3. Replace it with:
   ```html
   <div class="human-avatar">
       <video id="avatarVideo" autoplay playsinline muted></video>
       <canvas id="avatarCanvas"></canvas>
   ```

4. Find this line (near the end, before `</body>`):
   ```html
   <script src="app.js"></script>
   ```

5. Add these lines BEFORE it:
   ```html
   <script src="config.js"></script>
   <script src="avatar-controller.js"></script>
   <script src="app.js"></script>
   ```

6. Save the file (Ctrl+S)

### 2.3 Update styles.css

1. Open `styles.css`
2. Find this section:
   ```css
   #avatarImage {
       width: 100%;
       height: 100%;
       object-fit: cover;
       object-position: center top;
       display: block;
       position: relative;
       z-index: 2;
       transition: transform 0.3s ease;
   }
   ```

3. Replace it with:
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

4. Save the file (Ctrl+S)

---

## Step 3: Update app.js (10 minutes)

### 3.1 Add Avatar Controller Variable

1. Open `app.js`
2. Find this line at the top:
   ```javascript
   // DOM Elements
   const webcamFeed = document.getElementById('webcamFeed');
   ```

3. Add this line BEFORE it:
   ```javascript
   // Avatar Controller
   let avatarController = null;

   // DOM Elements
   const webcamFeed = document.getElementById('webcamFeed');
   ```

### 3.2 Update Initialize Function

1. Find this function:
   ```javascript
   function initializeApp() {
       setTimeout(() => {
           greetUser();
       }, 1500);
   }
   ```

2. Replace it with:
   ```javascript
   async function initializeApp() {
       // Check if animated avatar is enabled
       if (SELA_CONFIG.USE_ANIMATED_AVATAR) {
           updateStatus('loading', 'Loading avatar...');
           
           // Initialize D-ID Avatar
           avatarController = new AvatarController(
               SELA_CONFIG.DID_API_KEY,
               SELA_CONFIG.AVATAR_IMAGE
           );
           
           const initialized = await avatarController.initialize();
           
           if (!initialized) {
               console.error('Failed to initialize avatar, falling back to voice only');
               avatarController = null;
           }
       }
       
       setTimeout(() => {
           greetUser();
       }, 1500);
   }
   ```

### 3.3 Update Speak Function

1. Find this function:
   ```javascript
   function speakText(text) {
       // Cancel any ongoing speech
       if (synth.speaking) {
           synth.cancel();
       }

       isSpeaking = true;
       updateStatus('speaking', 'Speaking...');
       voiceIndicator.classList.add('active');
       
       // Animate avatar when speaking
       if (avatarImage) {
           avatarImage.classList.add('speaking');
       }

       const utterance = new SpeechSynthesisUtterance(text);
   ```

2. Replace it with:
   ```javascript
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
   ```

3. Keep the rest of the function as is

### 3.4 Add Cleanup

1. Find this at the end of the file:
   ```javascript
   // Cleanup on page unload
   window.addEventListener('beforeunload', () => {
       stopListening();
       if (synth.speaking) {
           synth.cancel();
       }
   });
   ```

2. Replace it with:
   ```javascript
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
   ```

4. Save the file (Ctrl+S)

---

## Step 4: Test Your Avatar (5 minutes)

### 4.1 Open in Browser
1. Close any open browser tabs with your app
2. Right-click `index.html`
3. Choose "Open with" → Chrome

### 4.2 Allow Permissions
1. Click "Allow" for microphone
2. Click "Allow" for camera

### 4.3 Watch the Magic!
- You'll see "Loading avatar..." status
- After 3-5 seconds, the avatar will appear
- Avatar will say: "Hello! I'm SELA..."
- The avatar's mouth will move perfectly with the speech!
- Facial expressions will match emotions

### 4.4 Test Conversation
Say: "Hello"
- Avatar will respond with lip-sync
- Try: "How are you?" "What time is it?"

---

## ✅ Success Checklist

- [ ] D-ID account created
- [ ] API key copied and pasted in config.js
- [ ] index.html updated (video element added)
- [ ] styles.css updated (video styles added)
- [ ] app.js updated (avatar controller integrated)
- [ ] All files saved
- [ ] Browser opened and permissions allowed
- [ ] Avatar loads and speaks with lip-sync

---

## 🐛 Troubleshooting

### Problem: "Loading avatar..." never finishes

**Solution:**
1. Open browser console (F12)
2. Check for errors
3. Verify API key is correct in config.js
4. Make sure you have internet connection

### Problem: Avatar doesn't appear

**Solution:**
1. Check console for errors (F12)
2. Verify avatar.png exists in folder
3. Try refreshing page (F5)

### Problem: "Failed to initialize avatar"

**Solution:**
1. Check your D-ID account has credits
2. Verify API key is correct (no extra spaces)
3. Check internet connection

### Problem: No sound from avatar

**Solution:**
1. Remove `muted` from video tag in index.html:
   ```html
   <video id="avatarVideo" autoplay playsinline></video>
   ```

---

## 💡 Tips for Best Quality

1. **Good Avatar Image**
   - Use high-resolution image (at least 800x1000px)
   - Clear face, front-facing
   - Good lighting
   - Neutral background

2. **Internet Connection**
   - Stable connection required
   - Minimum 5 Mbps recommended

3. **Browser**
   - Use Chrome or Edge (best support)
   - Keep browser updated

---

## 📊 What's Happening Behind the Scenes

1. **When app loads:**
   - Connects to D-ID servers
   - Uploads your avatar image
   - Creates a video stream
   - Starts WebRTC connection

2. **When SELA speaks:**
   - Sends text to D-ID
   - D-ID generates lip-sync animation
   - Streams video back in real-time
   - Shows emotion-matched expressions

3. **Result:**
   - Perfect lip-sync
   - Natural facial movements
   - Emotion-aware expressions
   - Professional quality

---

## 🎓 For Your Project Report

Include these points:
- "Implemented real-time avatar animation using D-ID Streaming API"
- "WebRTC-based video streaming for low-latency interaction"
- "Emotion-aware facial expressions synchronized with user emotion detection"
- "Professional-grade lip-sync using AI-powered animation"

---

## 💰 Cost Management

**Free Tier (10 min/month):**
- Perfect for development
- ~20-30 test conversations

**When to Upgrade ($5.90/month):**
- Week before presentation
- During demo preparation
- For final testing

**Tip:** Record your best demo for submission to save credits!

---

## 🎉 You're Done!

Your SELA now has a professional animated avatar with:
✅ Perfect lip-sync
✅ Facial expressions
✅ Real-time response
✅ High quality animation

Enjoy your amazing voice assistant! 🚀
