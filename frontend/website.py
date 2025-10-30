import streamlit as st
import cv2
from PIL import Image
import time
from datetime import datetime
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Real-time AI Video Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat interface
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c);
}
.chat-container {
    background: rgba(0, 0, 0, 0.7);
    border-radius: 15px;
    padding: 20px;
    height: 70vh;
    overflow-y: auto;
    margin-bottom: 20px;
}
.message {
    max-width: 85%;
    padding: 12px 16px;
    border-radius: 18px;
    margin-bottom: 15px;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.ai-message {
    background: rgba(74, 144, 226, 0.3);
    align-self: flex-start;
    border-bottom-left-radius: 5px;
}
.user-message {
    background: rgba(46, 204, 113, 0.3);
    align-self: flex-end;
    border-bottom-right-radius: 5px;
    margin-left: auto;
}
.prediction-result {
    background: rgba(231, 76, 60, 0.2);
    padding: 10px;
    border-radius: 8px;
    margin-top: 8px;
    font-weight: 600;
}
.confidence {
    float: right;
    font-size: 0.9rem;
    opacity: 0.8;
}
.status-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 0, 0, 0.7);
    padding: 8px 12px;
    border-radius: 20px;
    font-size: 0.9rem;
    width: fit-content;
}
.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #e74c3c;
}
.status-dot.active {
    background: #2ecc71;
    box-shadow: 0 0 10px #2ecc71;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hello! I'm ready to analyze your video feed in real-time. Start the analysis to begin."}
    ]
if 'analysis_active' not in st.session_state:
    st.session_state.analysis_active = False
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = 0
if 'captured_frame' not in st.session_state:
    # stored as JPEG bytes
    st.session_state.captured_frame = None
if 'capture_request' not in st.session_state:
    # flag used to request a capture from the live video loop
    st.session_state.capture_request = False

# Function to send frame to AI model (replace with your actual API endpoint)


def send_frame_to_model(frame):
    """
    Sends frame to AI model and returns prediction.
    Replace this with your actual API call.
    """
    # Convert frame to base64 for API transmission
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    # Example API call structure (replace with your actual endpoint)
    # api_url = "https://your-api-endpoint.com/predict"
    # response = requests.post(api_url, json={"image": frame_base64})
    # return response.json()

    # Simulate API response with random predictions
    import random
    predictions = [
        {"label": "Healthy Tooth", "confidence": 92},
        {"label": "Cavity Detected", "confidence": 87},
        {"label": "Gum Inflammation", "confidence": 78},
        {"label": "Plaque Buildup", "confidence": 85},
        {"label": "Healthy Tooth", "confidence": 95},
        {"label": "Cracked Tooth", "confidence": 82}
    ]
    return random.choice(predictions)

# Function to add message to chat


def add_message(role, content, prediction=None):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "prediction": prediction,
        "timestamp": datetime.now().strftime("%H:%M")
    })

# Function to process video frame


def process_frame(frame):
    # Add timestamp to frame
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Send to model if enough time has passed (simulate 1.5s interval)
    current_time = time.time()
    if current_time - st.session_state.last_prediction_time > 1.5:
        try:
            prediction = send_frame_to_model(frame)
            add_message(
                "ai", f"Analysis complete: {prediction['label']}", prediction)
            st.session_state.prediction_count += 1
            st.session_state.last_prediction_time = current_time
        except Exception as e:
            add_message("system", f"Error: {str(e)}")

    return frame


# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📹 Real-time Video Feed")

    # Status indicator
    status_placeholder = st.empty()
    with status_placeholder.container():
        if st.session_state.analysis_active:
            st.markdown('<div class="status-indicator"><div class="status-dot active"></div><span>Connected</span></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator"><div class="status-dot"></div><span>Disconnected</span></div>',
                        unsafe_allow_html=True)

    # Video display
    video_placeholder = st.empty()

    # Controls
    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        start_btn = st.button("▶ Start Analysis", use_container_width=True,
                              disabled=st.session_state.analysis_active)
    with control_col2:
        stop_btn = st.button("⏹ Stop Analysis", use_container_width=True,
                             disabled=not st.session_state.analysis_active)
    with control_col3:
        capture_btn = st.button("📷 Capture Frame", use_container_width=True)

    # Instructions
    st.markdown("<div style='text-align: center; margin-top: 10px; opacity: 0.7;'>Allow camera access to begin real-time analysis</div>",
                unsafe_allow_html=True)

with col2:
    st.markdown("### 💬 AI Predictions")
    st.markdown(f"**Predictions:** {st.session_state.prediction_count}")

    # Chat container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            role_class = "ai-message" if msg["role"] == "ai" else "user-message"
            st.markdown(f"""
            <div class="message {role_class}">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.85rem; opacity: 0.8;">
                    <span>{'AI Model' if msg['role'] == 'ai' else 'You'}</span>
                    <span>{msg.get('timestamp', 'Just now')}</span>
                </div>
                <div>{msg['content']}</div>
                {f'''
                <div class="prediction-result">
                    Prediction: {msg['prediction']['label']}
                    <span class="confidence">{msg['prediction']['confidence']}%</span>
                </div>
                ''' if msg.get('prediction') else ''}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Model info
    st.markdown("""
    <div style="background: rgba(0, 0, 0, 0.8); padding: 10px; border-radius: 8px; text-align: center; font-size: 0.9rem; opacity: 0.8; margin-top: 10px;">
        Connected to: Tooth Analysis Model v2.1 | Confidence threshold: 85%
    </div>
    """, unsafe_allow_html=True)

    # Preview of last captured frame
    st.markdown("### 🎞️ Recording Preview")
    if st.session_state.captured_frame:
        # Show the preview image and a download button
        st.image(st.session_state.captured_frame, use_column_width=True)
        st.download_button("⬇️ Download Capture", data=st.session_state.captured_frame,
                           file_name="capture.jpg", mime="image/jpeg")
        if st.button("Clear Preview"):
            st.session_state.captured_frame = None
            st.experimental_rerun()
    else:
        st.markdown("No capture yet. Press 📷 Capture Frame to take a snapshot.")

# Handle button clicks
if start_btn:
    st.session_state.analysis_active = True
    add_message(
        "system", "Real-time analysis started. Sending frames to AI model...")
    st.rerun()

if stop_btn:
    st.session_state.analysis_active = False
    add_message("system", "Real-time analysis stopped.")
    st.rerun()

# Capture button behavior:
if capture_btn:
    # If live analysis is running, request capture from the video loop
    if st.session_state.analysis_active:
        st.session_state.capture_request = True
        add_message("system", "Capture requested — will save next frame.")
        st.rerun()
    else:
        # If not running, open webcam once, grab a frame and store it
        cap_one = cv2.VideoCapture(0)
        if not cap_one.isOpened():
            st.error(
                "Cannot access camera to capture frame. Ensure permissions are granted.")
        else:
            ret_one, frame_one = cap_one.read()
            cap_one.release()
            if not ret_one or frame_one is None:
                st.error("Failed to capture frame")
            else:
                try:
                    _, buf = cv2.imencode('.jpg', frame_one)
                    st.session_state.captured_frame = buf.tobytes()
                    add_message("user", "Captured a frame for analysis")
                except Exception as e:
                    st.error(f"Error encoding captured frame: {e}")
        st.rerun()

# Video processing loop
if st.session_state.analysis_active:
    # Initialize camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error(
            "Cannot access camera. Please ensure you have a webcam and permissions are granted.")
    else:
        try:
            while st.session_state.analysis_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break

                # Process frame
                processed_frame = process_frame(frame)

                # If a capture was requested, save this processed frame as JPEG bytes
                if st.session_state.capture_request:
                    try:
                        _, buf = cv2.imencode('.jpg', processed_frame)
                        st.session_state.captured_frame = buf.tobytes()
                        add_message("user", "Captured a frame (live).")
                    except Exception as e:
                        add_message("system", f"Capture failed: {e}")
                    finally:
                        st.session_state.capture_request = False

                # Convert to RGB for Streamlit
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(
                    rgb_frame, channels="RGB", use_column_width=True)

                # Add small delay to prevent excessive CPU usage
                time.sleep(0.1)

        except Exception as e:
            st.error(f"Error during video processing: {str(e)}")
        finally:
            cap.release()

    # Reset after stopping
    if not st.session_state.analysis_active:
        video_placeholder.empty()
        st.rerun()

# Footer
st.markdown("<div style='text-align: center; margin-top: 20px; opacity: 0.7;'>Real-time AI Video Analysis System</div>",
            unsafe_allow_html=True)
