import streamlit as st
import streamlit.components.v1 as components
import datetime

st.set_page_config(
    page_title="🦷 Dental AI Diagnosis",
    layout="wide",
    page_icon="🦷"
)


def handle_new_prediction(prediction_data):
    """Store new prediction in session state"""
    st.session_state.chat_history.append({
        "type": "prediction",
        "predicted_class": prediction_data["predicted_class"],
        "confidence": prediction_data["confidence"],
        "timestamp": prediction_data["timestamp"]
    })
    st.session_state.total_predictions += 1


# Light, lively CSS with animations
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 14px 16px;
        margin: 10px 0;
        border-radius: 18px;
        max-width: 95%;
        animation: slideInUp 0.3s ease-out;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .ai-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-left-radius: 6px;
    }
    
    .system-message {
        background: #f8f9fa;
        color: #6c757d;
        text-align: center;
        font-size: 0.85em;
        border: 1px solid #e9ecef;
    }
    
    .confidence-bar {
        height: 6px;
        background: rgba(255,255,255,0.3);
        border-radius: 3px;
        margin: 8px 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: rgba(255,255,255,0.9);
        border-radius: 3px;
        transition: width 0.5s ease;
    }
    
    @keyframes slideInUp {
        from { 
            opacity: 0; 
            transform: translateY(15px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    
    .video-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.2);
    }
    
    .status-badge {
        padding: 8px 16px;
        border-radius: 25px;
        font-size: 0.85em;
        font-weight: 600;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    .status-active {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
    }
    
    .status-inactive {
        background: #6c757d;
        color: white;
    }
    
    .prediction-card {
        background: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }
    
    .floating-controls {
        background: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

# Header with gradient
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 2.5rem;">🦷 Dental AI Assistant</h1>
    <p style="margin:0; opacity: 0.9; font-size: 1.1rem;">Real-time dental condition analysis with AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for predictions
with st.sidebar:
    st.markdown("### 📋 Diagnosis History")

    # Status badge with animation
    status_class = "status-active" if st.session_state.analysis_active else "status-inactive"
    status_text = "🔍 Live Analysis" if st.session_state.analysis_active else "⏸️ Ready"
    st.markdown(
        f'<div class="status-badge {status_class}">{status_text}</div>', unsafe_allow_html=True)

    # Quick stats
    if st.session_state.chat_history:
        predictions = [
            m for m in st.session_state.chat_history if m["type"] == "prediction"]
        if predictions:
            st.markdown(f"""
            <div class="stats-card">
                <div style="font-size: 0.9em; opacity: 0.9;">Total Predictions</div>
                <div style="font-size: 1.8em; font-weight: bold;">{len(predictions)}</div>
            </div>
            """, unsafe_allow_html=True)

    # Chat history
    chat_container = st.container(height=450)

    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="chat-message system-message">
                🎯 Start analysis to see real-time predictions here
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show last 12 messages in reverse order (newest at bottom)
            for message in st.session_state.chat_history[-12:]:
                if message["type"] == "prediction":
                    confidence = message.get("confidence", 0)
                    confidence_percent = f"{confidence * 100:.1f}%"
                    class_name = message['predicted_class'].replace(
                        '_', ' ').title()

                    # Different colors based on confidence
                    if confidence > 0.8:
                        confidence_color = "rgba(255,255,255,0.9)"
                    elif confidence > 0.6:
                        confidence_color = "rgba(255,255,255,0.7)"
                    else:
                        confidence_color = "rgba(255,255,255,0.5)"

                    st.markdown(f"""
                    <div class="chat-message ai-message">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <strong>🦷 {class_name}</strong>
                            <small style="opacity: 0.8;">{message['timestamp']}</small>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence * 100}%; background: {confidence_color};"></div>
                        </div>
                        <div style="font-size: 0.8em; opacity: 0.9; display: flex; justify-content: space-between;">
                            <span>Confidence: {confidence_percent}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif message["type"] == "system":
                    st.markdown(f"""
                    <div class="chat-message system-message">
                        {message['message']}
                        <br>
                        <small>{message['timestamp']}</small>
                    </div>
                    """, unsafe_allow_html=True)

    # Controls in floating panel
    st.markdown('<div class="floating-controls">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
            st.session_state.chat_history.clear()
            st.session_state.total_predictions = 0
            st.rerun()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, type="secondary"):
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📹 Live Camera Feed")

    # Server configuration
    with st.expander("⚙️ Settings", expanded=False):
        server_url = st.text_input(
            "WebSocket Server URL", "ws://13.60.82.136:8000/ws/predict")
        st.caption("Make sure your FastAPI server is running at this address")

    # Video component
    components.html(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; gap: 1.5rem;">
            <!-- Video Feed -->
            <div class="video-container">
                <video id="video" width="640" height="480" autoplay playsinline
                       style="display: block; background: #000; border-radius: 12px;"></video>
            </div>
            
            <!-- Current Prediction Display -->
            <div id="currentPrediction" style="display: none; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; text-align: center; min-width: 300px; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <div style="font-size: 0.9em; opacity: 0.9;">Current Diagnosis</div>
                <div id="predictionText" style="font-size: 1.4em; font-weight: bold; margin: 8px 0;"></div>
                <div id="confidenceText" style="font-size: 1em; opacity: 0.9; margin-bottom: 8px;"></div>
                <div class="confidence-bar">
                    <div id="confidenceBar" class="confidence-fill" style="width: 0%"></div>
                </div>
            </div>
             <!-- Prediction object / direct Streamlit call removed:
                  Predictions are handled in JavaScript and sent to Streamlit via
                  window.parent.postMessage (see handleMessage -> postMessage). -->
            
            <!-- Status & Controls -->
            <div style="display: flex; flex-direction: column; align-items: center; gap: 1rem; width: 100%; max-width: 640px;">
                <div id="status" style="padding: 12px 24px; background: #f8f9fa; border-radius: 25px; font-weight: 600; width: 100%; text-align: center; border: 2px solid #e9ecef;">
                    🔄 Initializing...
                </div>
                
                <div style="display: flex; gap: 12px; width: 100%;">
                    <button id="startBtn" onclick="startAnalysis()" 
                            style="flex: 1; padding: 14px 24px; background: linear-gradient(135deg, #28a745, #20c997); color: white; border: none; border-radius: 25px; font-weight: 600; cursor: pointer; font-size: 1.1em; transition: all 0.3s ease;">
                        ▶ Start Analysis
                    </button>
                    <button id="stopBtn" onclick="stopAnalysis()" 
                            style="flex: 1; padding: 14px 24px; background: linear-gradient(135deg, #dc3545, #e83e8c); color: white; border: none; border-radius: 25px; font-weight: 600; cursor: pointer; font-size: 1.1em; opacity: 0.6; transition: all 0.3s ease;" disabled>
                        ⏹ Stop
                    </button>
                </div>
            </div>
        </div>

        <canvas id="canvas" style="display: none;"></canvas>

        <script>
            let ws;
            let streaming = false;
            let video = document.getElementById('video');
            let canvas = document.createElement('canvas');
            let ctx = canvas.getContext('2d');
            let statusDiv = document.getElementById('status');
            let startBtn = document.getElementById('startBtn');
            let stopBtn = document.getElementById('stopBtn');
            let currentPrediction = document.getElementById('currentPrediction');
            let predictionText = document.getElementById('predictionText');
            let confidenceText = document.getElementById('confidenceText');
            let confidenceBar = document.getElementById('confidenceBar');
            let frameCount = 0;

            // Connect to WebSocket
            function connect() {{
                try {{
                    ws = new WebSocket('{server_url}');
                    
                    ws.onopen = () => {{
                        updateStatus('✅ Connected to AI Server', 'success');
                        startBtn.disabled = false;
                        startBtn.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
                    }};
                    
                    ws.onmessage = async (event) => {{
                        try {{
                            let msg;
                            if (event.data instanceof Blob) {{
                                const text = await event.data.text();
                                msg = JSON.parse(text);
                            }} else {{
                                msg = JSON.parse(event.data);
                            }}
                            handleMessage(msg);
                        }} catch (e) {{
                            console.error('Parse error:', e);
                        }}
                    }};
                    
                    ws.onclose = () => {{
                        updateStatus('⚠️ Disconnected - Refresh to retry', 'warning');
                        streaming = false;
                        toggleButtons(false);
                        currentPrediction.style.display = 'none';
                    }};
                    
                    ws.onerror = (error) => {{
                        updateStatus('❌ Connection failed', 'error');
                    }};
                    
                }} catch (error) {{
                    updateStatus('❌ WebSocket error', 'error');
                }}
            }}
            
            function handleMessage(msg) {{
                if (msg.type === 'prediction') {{
                    // Update current prediction display
                    const className = msg.predicted_class.replace(/_/g, ' ').replace(/\\w\\S*/g, 
                        txt => txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase());
                    const confidence = (msg.confidence * 100).toFixed(1);
                    
                    predictionText.textContent = className;
                    confidenceText.textContent = `Confidence: ${{confidence}}%`;
                    confidenceBar.style.width = `${{msg.confidence * 100}}%`;
                    currentPrediction.style.display = 'block';
                    
                    // Send to Streamlit for chat history - FIXED THIS PART
                    const predictionData = {{
                        type: 'prediction',
                        predicted_class: msg.predicted_class,
                        confidence: msg.confidence,
                        timestamp: new Date().toLocaleTimeString()
                    }};
                    
                    // Send to parent window (Streamlit)
                    window.parent.postMessage({{
                        type: 'streamlit_prediction',
                        data: predictionData
                    }}, '*');
                    
                }} else if (msg.type === 'status') {{
                    updateStatus('🔍 ' + msg.message, 'info');
                    
                    // Send system messages to Streamlit
                    const systemData = {{
                        type: 'system', 
                        message: msg.message,
                        timestamp: new Date().toLocaleTimeString()
                    }};
                    
                    window.parent.postMessage({{
                        type: 'streamlit_system',
                        data: systemData
                    }}, '*');
                    
                }} else if (msg.type === 'error') {{
                    updateStatus('❌ ' + msg.message, 'error');
                }}
            }}
            
            function updateStatus(message, type) {{
                statusDiv.textContent = message;
                const colors = {{
                    success: 'linear-gradient(135deg, #28a745, #20c997)',
                    error: 'linear-gradient(135deg, #dc3545, #e83e8c)', 
                    warning: 'linear-gradient(135deg, #ffc107, #fd7e14)',
                    info: 'linear-gradient(135deg, #17a2b8, #6f42c1)'
                }};
                statusDiv.style.background = colors[type] || '#f8f9fa';
                statusDiv.style.color = 'white';
                statusDiv.style.border = 'none';
            }}
            
            // Start webcam
            async function startCamera() {{
                try {{
                    const stream = await navigator.mediaDevices.getUserMedia({{
                        video: {{ 
                            width: {{ ideal: 640 }},
                            height: {{ ideal: 480 }},
                            frameRate: {{ ideal: 30 }}
                        }}, 
                        audio: false 
                    }});
                    video.srcObject = stream;
                    
                    video.addEventListener('loadedmetadata', () => {{
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                    }});
                    
                }} catch (err) {{
                    updateStatus('❌ Camera access required', 'error');
                }}
            }}
            
            // Real-time frame capture
            function captureFrame() {{
                if (!streaming || !video.videoWidth) return;
                
                try {{
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
                    
                    if (ws && ws.readyState === WebSocket.OPEN && streaming) {{
                        ws.send(JSON.stringify({{
                            frame: dataUrl
                        }}));
                    }}
                }} catch (e) {{
                    console.error('Frame capture error:', e);
                }}
                
                if (streaming) {{
                    setTimeout(() => requestAnimationFrame(captureFrame), 150); // ~6-7 FPS
                }}
            }}
            
            function toggleButtons(starting) {{
                startBtn.disabled = starting;
                startBtn.style.opacity = starting ? '0.6' : '1';
                stopBtn.disabled = !starting;
                stopBtn.style.opacity = starting ? '1' : '0.6';
            }}
            
            function startAnalysis() {{
                if (!ws || ws.readyState !== WebSocket.OPEN) {{
                    connect();
                    setTimeout(startAnalysis, 500);
                    return;
                }}
                
                streaming = true;
                toggleButtons(true);
                updateStatus('🔍 Analyzing Live Video...', 'info');
                currentPrediction.style.display = 'none';
                
                captureFrame();
                
                ws.send(JSON.stringify({{
                    action: "start",
                    predict_every_frames: 1
                }}));
                
                // Notify Streamlit
                window.parent.postMessage({{
                    type: 'analysis_started'
                }}, '*');
            }}
            
            function stopAnalysis() {{
                streaming = false;
                toggleButtons(false);
                updateStatus('⏸️ Ready for Analysis', 'info');
                currentPrediction.style.display = 'none';
                
                if (ws && ws.readyState === WebSocket.OPEN) {{
                    ws.send(JSON.stringify({{ action: "stop" }}));
                }}
                
                window.parent.postMessage({{
                    type: 'analysis_stopped'  
                }}, '*');
            }}
            
            // Initialize
            startCamera();
            connect();
        </script>
        """,
        height=700
    )

with col2:
    st.markdown("### 📊 Live Stats")

    predictions = [
        m for m in st.session_state.chat_history if m["type"] == "prediction"]

    if predictions:
        latest = predictions[-1]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current", latest['predicted_class'].replace(
                '_', ' ').title())
        with col2:
            st.metric("Confidence",
                      f"{latest.get('confidence', 0) * 100:.1f}%")

        # Most common prediction
        if predictions:
            classes = [p['predicted_class'] for p in predictions]
            most_common = max(set(classes), key=classes.count)
            st.metric("Most Common", most_common.replace('_', ' ').title())

    st.markdown("---")
    st.markdown("#### 🦷 Conditions")
    st.markdown("""
    - **Cavities** - Tooth decay
    - **Gum Disease** - Inflammation  
    - **Plaque** - Tartar buildup
    - **Ulcers** - Mouth sores
    - **Stains** - Discoloration
    - **Missing Teeth** - Hypodontia
    - **Healthy** - Normal teeth
    """)

    st.markdown("---")
    st.markdown("#### 💡 Tips")
    st.markdown("""
    • Ensure good lighting
    • Focus on specific areas
    • Keep camera steady
    • Move slowly for best results
    • Start/stop as needed
    """)

# JavaScript message handler - FIXED PREDICTION STORAGE
st.markdown("""
<script>
    // Handle messages from the video iframe at the top-level Streamlit page.
    // This script runs in the main page (not inside an iframe) so it can receive
    // postMessage events sent from the video component iframe.
    (function() {
        function safeSetComponentValue(payload) {
            try {
                if (window.Streamlit && typeof window.Streamlit.setComponentValue === 'function') {
                    window.Streamlit.setComponentValue(payload);
                } else {
                    // Fallback: log so developer can see messages in browser console
                    console.log('Streamlit.setComponentValue not available; payload:', payload);
                }
            } catch (e) {
                console.error('Error calling Streamlit.setComponentValue:', e, payload);
            }
        }

        window.addEventListener('message', function(event) {
            try {
                if (!event || !event.data) return;
                const msg = event.data;

                if (msg.type === 'streamlit_prediction') {
                    const prediction = msg.data || {};
                    // Attempt to forward to Streamlit component API (if available)
                    safeSetComponentValue({
                        type: 'new_prediction',
                        data: prediction
                    });

                    // Always log for debugging in the browser console
                    console.log('🎯 New prediction received (main page):', prediction);
                } else if (msg.type === 'streamlit_system') {
                    const systemMsg = msg.data || {};
                    safeSetComponentValue({
                        type: 'system_message',
                        data: systemMsg
                    });
                    console.log('ℹ️ System message received (main page):', systemMsg);
                } else if (msg.type === 'analysis_started' || msg.type === 'analysis_stopped') {
                    safeSetComponentValue({
                        type: msg.type,
                        data: msg.data || { timestamp: new Date().toLocaleTimeString() }
                    });
                    console.log('🔁 Analysis event received (main page):', msg.type, msg.data);
                } else {
                    // Ignore unrelated messages
                }
            } catch (err) {
                console.error('Error handling postMessage in main page:', err);
            }
        }, false);
    })();
</script>
""", unsafe_allow_html=True)

# Handle the actual storage of predictions


def handle_system_message(system_data):
    """Store system message in session state"""
    st.session_state.chat_history.append({
        "type": "system",
        "message": system_data["message"],
        "timestamp": system_data["timestamp"]
    })


# Manual test to verify storage is working
with st.expander("🔧 Debug Tools", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧪 Add Test Prediction"):
            test_prediction = {
                "type": "prediction",
                "predicted_class": "caries",
                "confidence": 0.92,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }
            handle_new_prediction(test_prediction)
            st.rerun()

    with col2:
        if st.button("📊 Show Session State"):
            st.write("Chat History:", st.session_state.chat_history)
            st.write("Total Predictions:", st.session_state.total_predictions)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; font-size: 0.9em; padding: 2rem;'>"
    "🦷 Real-time AI Dental Analysis • Powered by Streamlit + FastAPI"
    "</div>",
    unsafe_allow_html=True
)
