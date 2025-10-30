import streamlit as st
import asyncio
import websockets
import json
import base64
import cv2
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
import threading
import time

# Page configuration
st.set_page_config(
    page_title="Real-Time Dental Diagnosis",
    page_icon="🦷",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .prediction-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #ff4b4b;
    }
    .confidence-bar {
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        background-color: #ff4b4b;
        border-radius: 4px;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
        max-width: 80%;
    }
    .user-message {
        background-color: #0078ff;
        color: white;
        margin-left: auto;
    }
    .ai-message {
        background-color: #f1f1f1;
        color: black;
    }
</style>
""", unsafe_allow_html=True)


class VideoProcessor:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=10)
        self.prediction_queue = queue.Queue()
        self.websocket_uri = "ws://localhost:8000/ws/predict"
        self.connected = False
        self.websocket = None
        self.processing = False
        # send_mode: 'binary' or 'text_base64'
        self.send_mode = 'binary'
        # how many frames to skip between predictions (control sent to server)
        self.predict_every = 5

    async def connect_websocket(self):
        """Connect to FastAPI WebSocket endpoint"""
        try:
            self.websocket = await websockets.connect(self.websocket_uri)
            self.connected = True
            return True
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")
            return False

    async def send_frame(self, frame):
        """Send frame to WebSocket endpoint"""
        if self.connected and self.websocket:
            try:
                # Convert frame to bytes
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                if self.send_mode == 'binary':
                    # send as binary bytes
                    await self.websocket.send(frame_bytes)
                else:
                    # send as text JSON with base64-encoded frame under key 'frame'
                    b64 = base64.b64encode(frame_bytes).decode('ascii')
                    payload = json.dumps({"frame": b64})
                    await self.websocket.send(payload)

                prediction = await self.websocket.recv()
                return prediction
            except Exception as e:
                st.error(f"❌ WebSocket error: {e}")
                self.connected = False
                return None
        return None

    async def send_control_message(self, control: dict):
        """Send a JSON control message over the websocket (text).

        Example: {"predict_every": 5}
        The server expects text JSON control messages and will adjust behavior.
        """
        if self.connected and self.websocket:
            try:
                payload = json.dumps({"type": "control", **control})
                await self.websocket.send(payload)
                # Optionally read ack
                try:
                    ack = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                except Exception:
                    ack = None
                return ack
            except Exception as e:
                st.error(f"❌ Failed to send control message: {e}")
                return None
        return None

    def recv(self, frame):
        """Process incoming frames from webcam"""
        if self.processing:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))

            if not self.frame_queue.full():
                self.frame_queue.put(frame_resized)

            return frame_rgb
        return frame

    async def process_frames(self):
        """Process frames from queue and send to WebSocket"""
        while self.processing:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=1)
                    prediction_data = await self.send_frame(frame)

                    if prediction_data:
                        try:
                            prediction = json.loads(prediction_data)
                            self.prediction_queue.put(prediction)
                        except json.JSONDecodeError:
                            self.prediction_queue.put(
                                {"prediction": prediction_data})

            except queue.Empty:
                continue
            except Exception as e:
                st.error(f"❌ Processing error: {e}")
                await asyncio.sleep(0.1)


def main():
    # Initialize session state at the VERY beginning
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'processor' not in st.session_state:
        st.session_state.processor = VideoProcessor()
    if 'analysis_started' not in st.session_state:
        st.session_state.analysis_started = False
    if 'websocket_connected' not in st.session_state:
        st.session_state.websocket_connected = False
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    st.title("🦷 Real-Time Dental Diagnosis AI")
    st.markdown("Live video analysis for dental conditions using AI")

    # Sidebar for controls
    with st.sidebar:
        st.header("🔧 Controls")

        # Server configuration
        st.subheader("Server Settings")
        server_ip = st.text_input(
            "Server IP", value="localhost", key="server_ip")
        server_port = st.number_input(
            "Port", value=8000, min_value=1000, max_value=9999)

        # Prediction cadence and send mode
        predict_every = st.number_input(
            "Predict every N frames", value=5, min_value=1, max_value=60)
        send_as_base64 = st.checkbox(
            "Send frames as base64 text (instead of binary)", value=False)

        # Update WebSocket URI and processor settings
        st.session_state.processor.websocket_uri = f"ws://{server_ip}:{server_port}/ws/predict"
        # configure processor send mode and cadence
        st.session_state.processor.send_mode = 'text_base64' if send_as_base64 else 'binary'
        st.session_state.processor.predict_every = int(predict_every)

        # Connection controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🟢 Connect", use_container_width=True):
                async def connect():
                    success = await st.session_state.processor.connect_websocket()
                    st.session_state.websocket_connected = success
                    if success:
                        # send control message so server knows the cadence
                        try:
                            await st.session_state.processor.send_control_message({"predict_every": int(predict_every)})
                        except Exception:
                            pass
                        st.success("✅ Connected to AI server")
                    return success

                # Run the async connection
                import asyncio
                asyncio.run(connect())
                st.rerun()

        with col2:
            if st.button("🔴 Disconnect", use_container_width=True):
                st.session_state.processor.connected = False
                st.session_state.processor.processing = False
                st.session_state.analysis_started = False
                st.session_state.websocket_connected = False
                st.session_state.camera_active = False
                st.warning("Disconnected from server")
                st.rerun()

        # Connection status
        if st.session_state.websocket_connected:
            st.success("✅ Connected to server")
        else:
            st.error("❌ Not connected to server")

        st.markdown("---")
        st.subheader("📊 Statistics")
        st.metric("Total Predictions", len(st.session_state.predictions))

        if st.session_state.predictions:
            latest_pred = st.session_state.predictions[-1]
            if isinstance(latest_pred, dict) and 'confidence' in latest_pred:
                st.metric("Latest Confidence",
                          f"{latest_pred['confidence']:.2%}")

        # Arbitrary control message sender for debugging
        st.markdown("---")
        st.subheader("🔁 Control / Debug")
        control_text = st.text_area(
            "Control JSON (e.g. {\"predict_every\":5})", value='')
        if st.button("Send Control Message", use_container_width=True):
            if control_text.strip() == "":
                st.error("Enter JSON to send")
            else:
                try:
                    control_obj = json.loads(control_text)
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
                    control_obj = None

                if control_obj is not None:
                    import asyncio as _asyncio
                    try:
                        ack = _asyncio.run(
                            st.session_state.processor.send_control_message(control_obj))
                        st.success("Control message sent")
                        if ack:
                            st.write("Ack:", ack)
                    except Exception as e:
                        st.error(f"Failed to send control message: {e}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Camera Feed")

        # Control buttons above video
        control_col1, control_col2 = st.columns(2)
        with control_col1:
            if st.button("▶️ Start Analysis", use_container_width=True):
                if st.session_state.websocket_connected:
                    st.session_state.processor.processing = True
                    st.session_state.analysis_started = True
                    st.session_state.camera_active = True
                    # Start processing in background thread
                    threading.Thread(
                        target=run_async_processing, daemon=True).start()
                    st.success("🎬 Analysis started")
                    st.rerun()
                else:
                    st.error("❌ Please connect to server first")

        with control_col2:
            if st.button("⏹️ Stop Analysis", use_container_width=True):
                st.session_state.processor.processing = False
                st.session_state.analysis_started = False
                st.info("⏹️ Analysis stopped")
                st.rerun()

        # Only show camera if analysis is started
        if st.session_state.camera_active:
            webrtc_ctx = webrtc_streamer(
                key="dental-camera",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}]},
                # Create a fresh VideoProcessor inside the worker to avoid
                # accessing Streamlit session_state from the worker thread
                # (which raises AttributeError). The session_state-held
                # processor is used for UI/state; the worker will have its
                # own processor instance for frame handling.
                video_processor_factory=lambda: VideoProcessor(),
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 15}
                    },
                    "audio": False
                },
            )
        else:
            st.info("👆 Click 'Start Analysis' to activate camera")

        # Analysis status
        if st.session_state.analysis_started:
            st.success("🔍 Analysis running - processing frames")
        else:
            st.info("⏸️ Analysis paused")

    with col2:
        st.subheader("🧠 AI Predictions")

        # Prediction display area
        prediction_container = st.container(height=500)

        with prediction_container:
            if not st.session_state.predictions:
                st.info(
                    "👆 Connect to server and start analysis to see predictions here")
            else:
                for i, pred in enumerate(reversed(st.session_state.predictions[-10:])):
                    display_prediction(pred, i)

        # Clear predictions button
        if st.button("🗑️ Clear Predictions", use_container_width=True):
            st.session_state.predictions.clear()
            st.rerun()

    # Process incoming predictions
    process_predictions()


def run_async_processing():
    """Run async processing in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(st.session_state.processor.process_frames())


def process_predictions():
    """Process predictions from queue and update session state"""
    processor = st.session_state.processor
    try:
        while not processor.prediction_queue.empty():
            prediction = processor.prediction_queue.get_nowait()
            st.session_state.predictions.append(prediction)
            # Use experimental_rerun to avoid recursion issues
            st.rerun()
    except queue.Empty:
        pass


def display_prediction(prediction, index):
    """Display a prediction in chat-like format"""
    current_time = time.strftime('%H:%M:%S')

    if isinstance(prediction, dict):
        if 'class' in prediction and 'confidence' in prediction:
            confidence = prediction['confidence']
            class_name = prediction['class']

            confidence_html = f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence * 100}%"></div>
            </div>
            <small>Confidence: {confidence:.2%}</small>
            """

            st.markdown(f"""
            <div class="chat-message ai-message">
                <strong>🦷 Diagnosis:</strong> {class_name.replace('_', ' ').title()}
                {confidence_html}
                <small style="color: #666;">{current_time}</small>
            </div>
            """, unsafe_allow_html=True)
        elif 'prediction' in prediction:
            st.markdown(f"""
            <div class="chat-message ai-message">
                <strong>AI:</strong> {prediction['prediction']}
                <small style="color: #666;">{current_time}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message ai-message">
                <strong>AI:</strong> {str(prediction)}
                <small style="color: #666;">{current_time}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message ai-message">
            <strong>AI:</strong> {str(prediction)}
            <small style="color: #666;">{current_time}</small>
        </div>
        """, unsafe_allow_html=True)


# Streamlit executes this file as a script (not __main__), so call main()
# unconditionally to ensure session state is initialized and UI is built.
main()
