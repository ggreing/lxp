document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let vectorstoreId = null;
    let chatSessionId = null;
    let eventSource = null;
    let mediaRecorder = null;
    let audioChunks = [];

    // --- RAG Elements ---
    const fileUpload = document.getElementById('file-upload');
    const uploadButton = document.getElementById('upload-button');
    const ragPrompt = document.getElementById('rag-prompt');
    const ragButton = document.getElementById('rag-button');
    const ragResult = document.getElementById('rag-result');
    const ragEvidence = document.getElementById('rag-evidence');

    // --- Chat Elements ---
    const chatModeText = document.getElementById('mode-text');
    const chatModeVoice = document.getElementById('mode-voice');
    const startChatButton = document.getElementById('start-chat-button');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input-text');
    const sendChatButton = document.getElementById('send-chat-button');
    const sttButton = document.getElementById('stt-button');
    const ttsControls = document.querySelector('.tts-controls');
    const ttsAudioPlayer = document.getElementById('tts-audio-player');

    // --- Simple TTS Elements ---
    const simpleTtsText = document.getElementById('simple-tts-text');
    const simpleTtsButton = document.getElementById('simple-tts-button');

    // --- Utility Functions ---
    function log(element, message, type = 'info') {
        const p = document.createElement('p');
        p.textContent = message;
        p.className = `log log-${type}`;
        element.prepend(p);
    }

    function displayChatMessage(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        messageDiv.textContent = message;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // --- RAG Logic (remains the same) ---
    uploadButton.addEventListener('click', async () => {
        const file = fileUpload.files[0];
        if (!file) { alert('Please select a file to upload.'); return; }
        ragResult.innerHTML = '';
        ragEvidence.innerHTML = '';
        log(ragResult, '1. Creating vector store...');
        try {
            const vsResponse = await fetch('/vectorstores/', { method: 'POST' });
            if (!vsResponse.ok) throw new Error(`Failed to create vector store (${vsResponse.status})`);
            const vsData = await vsResponse.json();
            vectorstoreId = vsData.id;
            log(ragResult, `   Vector store created: ${vectorstoreId}`);

            log(ragResult, '2. Uploading file...');
            const formData = new FormData();
            formData.append('file', file);
            const uploadResponse = await fetch(`/files/upload?user_id=frontend_user&vectorstore_id=${vectorstoreId}`, { method: 'POST', body: formData });
            if (!uploadResponse.ok) throw new Error(`Failed to upload file (${uploadResponse.status})`);
            log(ragResult, '   File uploaded successfully.');

            log(ragResult, '3. Indexing file...');
            const indexResponse = await fetch(`/vectorstores/${vectorstoreId}/index/`, { method: 'POST' });
            if (!indexResponse.ok) throw new Error(`Failed to index file (${indexResponse.status})`);
            const indexData = await indexResponse.json();
            log(ragResult, `   Indexing complete. ${indexData.indexed} points added.`);
            log(ragResult, 'Ready to ask questions!', 'success');
        } catch (error) {
            log(ragResult, `Error: ${error.message}`, 'error');
        }
    });

    ragButton.addEventListener('click', async () => {
        const prompt = ragPrompt.value;
        if (!prompt) { alert('Please enter a question.'); return; }
        if (!vectorstoreId) { alert('Please upload and index a file first.'); return; }
        log(ragResult, `Asking: ${prompt}`);
        ragResult.textContent = 'Thinking...';
        ragEvidence.innerHTML = '';
        try {
            const response = await fetch('/assist', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: 'frontend_user', prompt: prompt, vectorstore_id: vectorstoreId }),
            });
            if (!response.ok) { const errText = await response.text(); throw new Error(`Assist endpoint failed: ${errText}`); }
            const data = await response.json();
            log(ragResult, 'Job created. Waiting for results via SSE...');
            const sse = new EventSource(`/events/jobs/${data.job_id}`);
            sse.onmessage = (event) => {
                const eventData = JSON.parse(event.data);
                if (eventData.result) {
                    ragResult.textContent = eventData.result.answer;
                    if (eventData.result.evidence && eventData.result.evidence.length > 0) {
                        eventData.result.evidence.forEach(item => {
                            const evidenceDiv = document.createElement('div');
                            evidenceDiv.className = 'evidence-item';
                            evidenceDiv.innerHTML = `<strong>Source: ${item.filename || 'N/A'} (Score: ${item.score.toFixed(2)})</strong><p>${item.text}</p>`;
                            ragEvidence.appendChild(evidenceDiv);
                        });
                    }
                    sse.close();
                }
                if (eventData.error) { ragResult.textContent = `Error in job: ${eventData.error}`; sse.close(); }
            };
            sse.onerror = () => { log(ragResult, 'SSE connection error.', 'error'); sse.close(); };
        } catch (error) {
            ragResult.textContent = `Error: ${error.message}`;
        }
    });

    // --- Simple TTS Logic (remains the same) ---
    simpleTtsButton.addEventListener('click', async () => {
        const text = simpleTtsText.value;
        if (!text) { alert('Please enter text to synthesize.'); return; }
        try {
            simpleTtsButton.disabled = true;
            simpleTtsButton.textContent = 'Generating...';
            const response = await fetch('/sales/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            if (!response.ok) { const errText = await response.text(); throw new Error(`Failed to generate audio (${response.status}): ${errText}`); }
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            audio.play();
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            simpleTtsButton.disabled = false;
            simpleTtsButton.textContent = '음성 생성 및 재생';
        }
    });

    // --- NEW Chat Logic ---

    function handleUIMode() {
        if (chatModeVoice.checked) {
            sttButton.style.display = 'inline-block';
            ttsControls.style.display = 'flex';
        } else {
            sttButton.style.display = 'none';
            ttsControls.style.display = 'none';
        }
    }
    chatModeText.addEventListener('change', handleUIMode);
    chatModeVoice.addEventListener('change', handleUIMode);
    handleUIMode(); // Set initial state

    async function playTextAsSpeech(text) {
        if (!text) return;
        try {
            const response = await fetch('/sales/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            if (!response.ok) throw new Error('Simple TTS failed');
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            ttsAudioPlayer.src = audioUrl;
            ttsAudioPlayer.play();
        } catch (error) {
            console.error("Could not play TTS:", error);
        }
    }

    function setupSseListeners() {
        let currentAiMessage = '';
        let aiMessageDiv = null;

        eventSource.onmessage = (event) => {
            const chunk = JSON.parse(event.data);
            if (!aiMessageDiv) {
                aiMessageDiv = document.createElement('div');
                aiMessageDiv.classList.add('message', 'ai-message');
                chatMessages.appendChild(aiMessageDiv);
            }
            currentAiMessage += chunk;
            aiMessageDiv.textContent = currentAiMessage;
            chatMessages.scrollTop = chatMessages.scrollHeight;

            if (chatModeVoice.checked && (chunk.endsWith('.') || chunk.endsWith('?') || chunk.endsWith('!'))) {
                playTextAsSpeech(currentAiMessage);
                currentAiMessage = ''; // Reset for the next sentence
            }
        };

        eventSource.addEventListener('end', (event) => {
            const endMessage = JSON.parse(event.data);
            displayChatMessage(`--- ${endMessage} ---`, 'ai');
            if (currentAiMessage && chatModeVoice.checked) {
                playTextAsSpeech(currentAiMessage); // Play any remaining text
            }
            chatInput.disabled = true;
            sendChatButton.disabled = true;
            eventSource.close();
        });

        eventSource.onerror = (err) => {
            console.error("EventSource error:", err);
            displayChatMessage('--- Connection error or stream ended. ---', 'ai');
            chatInput.disabled = false;
            sendChatButton.disabled = false;
            if(eventSource) eventSource.close();
        };
    }

    startChatButton.addEventListener('click', async () => {
        chatMessages.innerHTML = '';
        chatInput.disabled = true;
        sendChatButton.disabled = true;
        startChatButton.disabled = true;
        startChatButton.textContent = 'Starting...';

        if (eventSource) {
            eventSource.close();
        }

        try {
            const sessionResponse = await fetch('/sales/session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: 'frontend_seller' }),
            });
            if (!sessionResponse.ok) throw new Error('Failed to start chat session.');
            const data = await sessionResponse.json();
            chatSessionId = data.session_id;

            displayChatMessage(data.greeting, 'ai');
            if (chatModeVoice.checked) {
                playTextAsSpeech(data.greeting);
            }

            // Connect to the SSE stream
            eventSource = new EventSource(`/sales/stream/${chatSessionId}`);
            setupSseListeners();

            chatInput.disabled = false;
            sendChatButton.disabled = false;
            startChatButton.textContent = 'Restart Chat';

        } catch (error) {
            displayChatMessage(`Error starting session: ${error.message}`, 'ai');
        } finally {
            startChatButton.disabled = false;
        }
    });

    const sendChatMessage = async () => {
        const message = chatInput.value;
        if (!message || !chatSessionId) return;

        displayChatMessage(message, 'user');
        chatInput.value = '';

        try {
            await fetch('/sales/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: chatSessionId,
                    seller_msg: message,
                    user_id: 'frontend_seller'
                })
            });
        } catch (error) {
            displayChatMessage(`Error sending message: ${error.message}`, 'ai');
        }
    };

    sendChatButton.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendChatMessage();
        }
    });

    // --- STT Logic (remains the same) ---
    sttButton.addEventListener('click', async () => {
        if (!mediaRecorder || mediaRecorder.state === 'inactive') {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                audioChunks = [];
                mediaRecorder.ondataavailable = event => { audioChunks.push(event.data); };
                mediaRecorder.onstop = async () => {
                    sttButton.classList.remove('recording');
                    sttButton.textContent = '🎤';
                    displayChatMessage('Processing audio...', 'user');
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const formData = new FormData();
                    formData.append('file', audioBlob, 'recording.webm');
                    try {
                        const response = await fetch('/stt', { method: 'POST', body: formData });
                        if (!response.ok) throw new Error('STT request failed.');
                        const data = await response.json();
                        chatInput.value = data.text;
                        chatMessages.removeChild(chatMessages.lastChild);
                    } catch (error) {
                        displayChatMessage(`STT Error: ${error.message}`, 'ai');
                    }
                };
                mediaRecorder.start();
                sttButton.classList.add('recording');
                sttButton.textContent = '🛑';
            } catch (error) {
                alert('Could not get audio stream. Please grant microphone permission.');
                console.error('Error getting user media:', error);
            }
        } else {
            mediaRecorder.stop();
        }
    });
});
