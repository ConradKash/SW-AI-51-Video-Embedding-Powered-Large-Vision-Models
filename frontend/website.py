import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Live Video Prediction", layout="wide")
st.title("🎥 Live Video Prediction via WebSocket")

# Explain how it works
st.markdown("""
This app uses your webcam to send live video frames to a FastAPI backend via WebSocket.
Predictions are displayed below.
""")

# Embed JavaScript for webcam + WebSocket
components.html(
    """
    <div style="display: flex; flex-direction: column; align-items: center;">
        <video id="video" width="640" height="480" autoplay playsinline style="border: 2px solid #007BFF; border-radius: 8px;"></video>
        <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        <div id="status" style="margin-top: 10px; font-weight: bold;">Connecting...</div>
        <div id="prediction" style="margin-top: 10px; font-size: 1.2em; color: #28a745;"></div>
        <button id="startBtn" onclick="start()" style="margin-top: 15px; padding: 8px 16px; font-size: 16px;">Start Analysis</button>
        <button id="stopBtn" onclick="stop()" style="margin-top: 10px; padding: 8px 16px; font-size: 16px;" disabled>Stop Analysis</button>
    </div>

    <script>
        let ws;
        let streaming = false;
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let ctx = canvas.getContext('2d');
        let statusDiv = document.getElementById('status');
        let predDiv = document.getElementById('prediction');
        let startBtn = document.getElementById('startBtn');
        let stopBtn = document.getElementById('stopBtn');

        // Connect to WebSocket
        function connect() {
            ws = new WebSocket('ws://localhost:8000/ws/predict');

            ws.onopen = () => {
                statusDiv.innerText = '✅ Connected to backend';
                statusDiv.style.color = 'green';
            };

            ws.onmessage = (event) => {
                let msg = JSON.parse(event.data);
                if (msg.type === 'prediction') {
                    predDiv.innerHTML = `<b>Prediction:</b> ${msg.predicted_class} (Confidence: ${(msg.confidence * 100).toFixed(1)}%)<br><small>Timestamp: ${msg.timestamp}</small>`;
                } else if (msg.type === 'status') {
                    statusDiv.innerText = msg.message;
                    statusDiv.style.color = 'blue';
                } else if (msg.type === 'error') {
                    statusDiv.innerText = '❌ Error: ' + msg.message;
                    statusDiv.style.color = 'red';
                } else if (msg.type === 'ack') {
                    // Optional: show frame count
                }
            };

            ws.onclose = () => {
                statusDiv.innerText = '⚠️ Disconnected. Refresh to retry.';
                statusDiv.style.color = 'orange';
                streaming = false;
                startBtn.disabled = false;
                stopBtn.disabled = true;
            };

            ws.onerror = (error) => {
                statusDiv.innerText = '❌ WebSocket error';
                statusDiv.style.color = 'red';
            };
        }

        // Start webcam
        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                video.srcObject = stream;
            } catch (err) {
                statusDiv.innerText = '❌ Camera access denied: ' + err.message;
                statusDiv.style.color = 'red';
            }
        }

        // Send frame every 200ms (5 FPS)
        let captureInterval;
        function startCapture() {
            captureInterval = setInterval(() => {
                if (!streaming) return;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                let dataUrl = canvas.toDataURL('image/jpeg', 0.7);
                // Send as base64 string
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ frame: dataUrl }));
                }
            }, 200); // ~5 FPS
        }

        function start() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                connect();
            }
            setTimeout(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ action: "start" }));
                    streaming = true;
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    startCapture();
                }
            }, 500);
        }

        function stop() {
            streaming = false;
            clearInterval(captureInterval);
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: "stop" }));
            }
            startBtn.disabled = false;
            stopBtn.disabled = true;
            predDiv.innerHTML = '';
        }

        // Initialize
        startCamera();
        connect();
    </script>
    """,
    height=700,
)
