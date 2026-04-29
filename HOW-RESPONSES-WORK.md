# How SELA Responds - Complete Guide

## 🎯 Main File: `app.js`

This is the BRAIN of SELA. All conversation logic is here!

---

## 📋 The Response Flow

### Step 1: User Opens App
```javascript
// Line ~128 in app.js
function greetUser() {
    const greeting = "Hello! I'm SELA, your socially-aware assistant. I can see and hear you. How can I help you today?";
    speakText(greeting);
}
```
**This makes SELA say hello when the app loads!**

---

### Step 2: User Speaks
```javascript
// Line ~250 in app.js
recognition.onresult = (event) => {
    const transcript = event.results[lastResultIndex][0].transcript.trim();
    // User's speech is captured here
    processUserSpeech(transcript);
}
```
**Speech Recognition captures what you say**

---

### Step 3: Process User Input
```javascript
// Line ~290 in app.js
function processUserSpeech(userInput) {
    updateStatus('thinking', 'Processing...');
    
    // Generate emotion-aware response
    const response = generateEmotionAwareResponse(userInput, emotionData);
    
    speakText(response);
}
```
**Sends your speech to the response generator**

---

### Step 4: Generate Response (THE IMPORTANT PART!)
```javascript
// Line ~310 in app.js
function generateEmotionAwareResponse(input, emotion) {
    const lower = input.toLowerCase();

    // Check what user said and respond accordingly
    
    // Example 1: Greetings
    if (lower.match(/\b(hello|hi|hey|greetings)\b/)) {
        return "Hello! It's wonderful to talk with you. What would you like to know?";
    }

    // Example 2: How are you
    if (lower.match(/how are you|how're you/)) {
        return "I'm doing fantastic, thank you for asking!";
    }

    // Example 3: Time
    if (lower.match(/what time|current time|time is it/)) {
        const now = new Date();
        return `The current time is ${now.toLocaleTimeString()}`;
    }

    // ... more responses ...
}
```
**This function decides what SELA says!**

---

## 🔧 How to Add New Responses

### Example: Add response for "What's your favorite color?"

Open `app.js` and find the `generateEmotionAwareResponse` function (around line 310).

Add this code BEFORE the "Default emotion-aware response" section:

```javascript
// Favorite color
if (lower.match(/favorite color|favourite color/)) {
    return "I love purple and blue! They remind me of the beautiful gradient background we're on right now.";
}
```

### Example: Add response for "Tell me a joke"

```javascript
// Jokes
if (lower.match(/joke|funny|make me laugh/)) {
    const jokes = [
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "Why did the AI go to school? To improve its learning algorithms!",
        "What's a computer's favorite snack? Microchips!"
    ];
    const randomJoke = jokes[Math.floor(Math.random() * jokes.length)];
    return randomJoke;
}
```

---

## 📊 Current Responses in SELA

| User Says | SELA Responds |
|-----------|---------------|
| "Hello", "Hi", "Hey" | "Hello! It's wonderful to talk with you..." |
| "How are you?" | "I'm doing fantastic, thank you for asking!" |
| "Who are you?" | "I'm SELA - a Socially-aware Embodied Language Agent..." |
| "What time is it?" | "The current time is [current time]" |
| "What's the date?" | "Today is [current date]" |
| "Thank you" | "You're very welcome! Happy to help anytime." |
| "Goodbye" | "Goodbye! It was wonderful talking with you..." |
| "Help" | "I'm here to assist you! I can sense your emotions..." |
| Anything else | "I heard you say: [your words]. That's interesting!..." |

---

## 🎭 Emotion-Aware Responses

SELA changes responses based on detected emotion:

### Example: User says "Hello"

**If user looks SAD:**
```javascript
return "Hello. I notice you might be feeling down. I'm here to listen and help in any way I can.";
```

**If user looks HAPPY:**
```javascript
return "Hi! You seem to be in a great mood! That's wonderful. What can I do for you today?";
```

**If user looks NEUTRAL:**
```javascript
return "Hello! It's wonderful to talk with you. What would you like to know?";
```

---

## 🔄 Complete Conversation Flow Diagram

```
1. App Loads
   ↓
2. greetUser() runs
   ↓
3. SELA says: "Hello! I'm SELA..."
   ↓
4. User speaks
   ↓
5. Speech Recognition captures words
   ↓
6. processUserSpeech() is called
   ↓
7. generateEmotionAwareResponse() decides what to say
   ↓
8. speakText() makes SELA speak
   ↓
9. Avatar animates (if D-ID is working)
   ↓
10. SELA starts listening again
```

---

## 📝 Files Involved

### Primary (Response Logic):
- **app.js** - ALL conversation logic (lines 310-400)

### Supporting:
- **config.js** - Settings (voice, API keys)
- **avatar-controller.js** - Makes avatar speak with lip-sync
- **index.html** - UI structure
- **styles.css** - Visual appearance

---

## 🎯 Quick Customization Guide

### Change the Greeting:
**File:** `app.js`
**Line:** ~130
```javascript
function greetUser() {
    const greeting = "YOUR NEW GREETING HERE";
    speakText(greeting);
}
```

### Add New Response:
**File:** `app.js`
**Line:** ~310 (inside `generateEmotionAwareResponse`)
```javascript
// Your new response
if (lower.match(/YOUR TRIGGER WORDS/)) {
    return "YOUR RESPONSE HERE";
}
```

### Change Voice:
**File:** `config.js`
**Line:** ~12
```javascript
VOICE_ID: 'en-US-JennyNeural',  // Change this
```

---

## 🚀 Advanced: Connect to AI (ChatGPT, etc.)

To make SELA smarter, replace the `generateEmotionAwareResponse` function with an API call:

```javascript
async function generateEmotionAwareResponse(input, emotion) {
    // Call ChatGPT API
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': 'Bearer YOUR_API_KEY',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: 'gpt-3.5-turbo',
            messages: [
                {
                    role: 'system',
                    content: `You are SELA, a friendly AI assistant. The user's current emotion is ${emotion.emotion}. Respond empathetically.`
                },
                {
                    role: 'user',
                    content: input
                }
            ]
        })
    });
    
    const data = await response.json();
    return data.choices[0].message.content;
}
```

---

## 💡 Summary

**To change what SELA says:**
1. Open `app.js`
2. Find `generateEmotionAwareResponse` function (line ~310)
3. Add your new `if` statement with trigger words
4. Return your response text
5. Save and refresh browser!

**That's it!** All responses are controlled in this ONE function. Easy to customize! 🎉
