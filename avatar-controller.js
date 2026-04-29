// D-ID Streaming Avatar Controller for SELA
// Real-time animated avatar with lip-sync and facial expressions

class AvatarController {
    constructor(apiKey, avatarImageUrl) {
        this.apiKey = apiKey;
        this.avatarImageUrl = avatarImageUrl;
        this.streamId = null;
        this.sessionId = null;
        this.peerConnection = null;
        this.videoElement = document.getElementById('avatarVideo');
        this.isInitialized = false;
    }

    // Initialize the avatar stream
    async initialize() {
        try {
            console.log('Initializing D-ID avatar...');
            
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

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.streamId = data.id;
            this.sessionId = data.session_id;

            // Setup WebRTC connection
            await this.setupWebRTC(data);
            
            this.isInitialized = true;
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
            iceServers: streamData.ice_servers || [
                { urls: 'stun:stun.l.google.com:19302' }
            ]
        });

        // Handle incoming video stream
        this.peerConnection.ontrack = (event) => {
            console.log('Received video track');
            if (event.track.kind === 'video') {
                this.videoElement.srcObject = event.streams[0];
            }
        };

        // Handle ICE connection state
        this.peerConnection.oniceconnectionstatechange = () => {
            console.log('ICE connection state:', this.peerConnection.iceConnectionState);
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
        if (!this.isInitialized) {
            console.error('Avatar not initialized');
            return false;
        }

        try {
            // Map emotions to D-ID expressions
            const expressionMap = {
                'happy': 'happy',
                'sad': 'sad',
                'angry': 'serious',
                'surprised': 'surprised',
                'neutral': 'neutral',
                'fearful': 'serious'
            };

            console.log(`Avatar speaking with ${emotion} emotion:`, text);

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

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return true;
        } catch (error) {
            console.error('Failed to make avatar speak:', error);
            return false;
        }
    }

    // Destroy the stream when done
    async destroy() {
        console.log('Destroying avatar stream...');
        
        if (this.streamId) {
            try {
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
            } catch (error) {
                console.error('Error destroying stream:', error);
            }
        }

        if (this.peerConnection) {
            this.peerConnection.close();
        }

        this.isInitialized = false;
    }
}

// Export for use in main app
window.AvatarController = AvatarController;
