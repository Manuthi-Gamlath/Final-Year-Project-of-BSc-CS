# SELA - Socially-aware Embodied Language Agent

## Project Overview
SELA is a multimodal conversational agent that integrates visual (facial expressions), audio (speech + prosody), and language understanding to create empathetic, socially-aware interactions.

## Current Implementation Status

### ✅ Completed (Frontend MVP)
- **Two-panel layout**: Webcam feed (left) + Avatar & conversation (right)
- **Webcam integration**: Live video capture with emotion overlay
- **2D Avatar**: Animated character with facial expressions
- **Voice interaction**: Speech-to-text (Web Speech API) and text-to-speech
- **Real-time transcript**: Live conversation history display
- **Emotion display**: Valence/arousal meters and discrete emotion labels
- **Personality panel**: Big Five traits visualization
- **Latency tracking**: Response time monitoring
- **Hands-free conversation**: Continuous listening after responses

### 🔄 Placeholder Components (Need ML Integration)
These are currently simulated and need to be replaced with actual models:

1. **Emotion Detection** (`detectEmotionFromFrame()`)
   - Current: Random emotion simulation
   - Needed: MediaPipe Face Mesh + AffectNet model
   - Output: Valence, arousal, discrete emotion (neutral, happy, sad, angry, surprised, fearful)

2. **Personality Inference** (`inferPersonalityFromSpeech()`)
   - Current: Simple keyword heuristics
   - Needed: Big Five personality model from text/prosody
   - Output: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism

3. **Response Generation** (`generateEmotionAwareResponse()`)
   - Current: Rule-based with emotion conditioning
   - Needed: LLM integration (GPT-4, Phi-3, LLaMA) with emotion-aware prompting
   - Output: Contextually and emotionally appropriate responses

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SELA System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT LAYER                                                │
│  ├─ Webcam (video) → Face Detection → Emotion Recognition  │
│  ├─ Microphone (audio) → STT → Prosody Analysis            │
│  └─ Text → Personality Inference                            │
│                                                             │
│  PROCESSING LAYER                                           │
│  ├─ Multimodal Feature Fusion                              │
│  ├─ Emotion + Personality State Estimation                 │
│  └─ LLM Reasoning (emotion-conditioned prompts)            │
│                                                             │
│  OUTPUT LAYER                                               │
│  ├─ Text Response Generation                               │
│  ├─ TTS (speech synthesis)                                 │
│  └─ Avatar Animation (emotion-matched expressions)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
SELA/
├── index.html          # Main UI structure
├── styles.css          # Styling and animations
├── app.js              # Core application logic
└── README.md           # This file
```

## How to Run

1. **Open in browser**:
   - Double-click `index.html` or
   - Right-click → Open with Chrome/Edge

2. **Grant permissions**:
   - Allow webcam access
   - Allow microphone access

3. **Start conversation**:
   - SELA will greet you automatically
   - Just speak naturally - no buttons needed

## Next Steps for Full Implementation

### Phase 1: Emotion Recognition (Weeks 1-4)
- [ ] Integrate MediaPipe Face Mesh for facial landmarks
- [ ] Implement AffectNet or FER+ model for emotion classification
- [ ] Add temporal smoothing for stable emotion detection
- [ ] Extract valence/arousal from facial action units

### Phase 2: Personality Inference (Weeks 5-8)
- [ ] Collect/prepare personality-labeled dataset
- [ ] Train lightweight Big Five classifier on text features
- [ ] Add prosody features (pitch, energy, speaking rate)
- [ ] Implement online personality estimation with session memory

### Phase 3: LLM Integration (Weeks 9-12)
- [ ] Set up LLM backend (local: LLaMA/Phi-3 or API: GPT-4)
- [ ] Design emotion-aware prompt templates
- [ ] Implement personality-conditioned response generation
- [ ] Add conversation context management

### Phase 4: Avatar Enhancement (Weeks 13-16)
- [ ] Create emotion-specific avatar expressions
- [ ] Sync avatar lip movements with TTS
- [ ] Add gaze tracking and eye contact simulation
- [ ] Implement smooth transitions between emotional states

### Phase 5: Optimization & Evaluation (Weeks 17-20)
- [ ] Optimize for <2.5s end-to-end latency
- [ ] Conduct user studies (engagement, empathy, satisfaction)
- [ ] Measure emotion recognition accuracy
- [ ] Evaluate response quality (BLEU, ROUGE, human ratings)

## Technical Requirements

### Current (Frontend Only)
- Modern browser (Chrome/Edge recommended)
- Webcam + microphone
- Internet connection (for Web Speech API)

### Future (Full System)
- Python 3.8+
- PyTorch / TensorFlow
- CUDA-capable GPU (recommended)
- Libraries: MediaPipe, librosa, transformers, FastAPI
- 8GB+ RAM, 4GB+ VRAM

## Integration Points

### Backend API Structure (To Be Implemented)
```javascript
// POST /api/process
{
  "video_frame": "base64_encoded_image",
  "audio_chunk": "base64_encoded_audio",
  "transcript": "user speech text",
  "conversation_history": [...],
  "emotion_state": {...},
  "personality_state": {...}
}

// Response
{
  "emotion": { "emotion": "happy", "valence": 0.8, "arousal": 0.6 },
  "personality": { "extraversion": 0.7, ... },
  "response_text": "Generated response",
  "avatar_emotion": "happy",
  "latency_ms": 1850
}
```

## Evaluation Metrics

### Objective
- Emotion recognition accuracy (F1-score)
- Response latency (target: <2500ms)
- Personality inference correlation
- Text quality (BLEU, ROUGE, perplexity)

### Subjective (User Studies)
- Perceived empathy (Likert scale)
- Engagement level
- Trust and rapport
- Naturalness of interaction

## References
Based on research from:
- Affective Computing (Poria et al., 2017)
- Multimodal Emotion Recognition (Lian et al., 2023)
- Embodied Conversational Agents (Chang et al., 2023)
- Personality-aware LLMs (Sonlu, 2024)

## License
Educational/Research Project - Final Year Undergraduate Work

## Contact
For questions about SELA implementation, refer to project documentation.
