#!/usr/bin/env python3
"""
Unified Video Reasoning Pipeline with DeepSeek via Ollama
Senior Software Engineer Edition: Preview Mode with Manual Recording Start
"""

import cv2
import time
import json
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List
import requests
import numpy as np
from ultralytics import YOLO


class UnifiedVideoReasoner:
    def __init__(self, yolo_model: str = "yolov8n.pt",
                 ollama_base_url: str = "http://localhost:11434",
                 deepseek_model: str = "deepseek-r1:1.5b"):
        """
        Initialize the unified video reasoning pipeline

        Args:
            yolo_model: Path to YOLO model or model name
            ollama_base_url: Base URL for Ollama API
            deepseek_model: DeepSeek model name in Ollama
        """
        # Initialize YOLO model
        print("🚀 Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model)

        # Ollama configuration
        self.ollama_url = ollama_base_url
        self.deepseek_model = deepseek_model

        # State management
        self.is_previewing = False
        self.is_recording = False
        self.is_processing = False

        # Data storage
        self.detection_history = defaultdict(list)
        self.frame_buffer = deque(maxlen=30)
        self.video_info = {}

        # Recording control
        self.recording_start_time = 0
        self.recording_duration = 0
        self.output_path = ""

        # Verify Ollama connection and model
        self._verify_ollama_setup()

        print("✅ Unified Video Reasoner initialized successfully!")

    def _verify_ollama_setup(self):
        """Verify Ollama is running and DeepSeek model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code != 200:
                raise Exception("Ollama is not running. Please start Ollama first.")

            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]

            if self.deepseek_model not in model_names:
                print(f"⚠️  DeepSeek model '{self.deepseek_model}' not found in Ollama.")
                print("Available models:", [name for name in model_names if name])
                print(f"Please pull the model first: ollama pull {self.deepseek_model}")
                raise Exception(f"Model {self.deepseek_model} not available")

            print(f"✅ DeepSeek model '{self.deepseek_model}' is available")

        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")

    def _call_deepseek(self, prompt: str, max_tokens: int = 512) -> str:
        """Call DeepSeek model through Ollama API"""
        payload = {
            "model": self.deepseek_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload
            )
            print(response.json())
            response.raise_for_status()
            return response.json().get('response', '').strip()

        except requests.exceptions.RequestException as e:
            return f"Error calling DeepSeek: {str(e)}"

    def start_preview(self, video_source: Any = 0, output_path: str = "recorded_video.mp4") -> None:
        """
        Start preview mode - shows camera feed with YOLO detections
        Press 's' to start recording, 'q' to quit

        Args:
            video_source: Camera index, video file, or RTSP stream
            output_path: Path to save recorded video
        """
        if self.is_previewing or self.is_recording:
            print("⚠️  Preview or recording already in progress")
            return

        print("🎬 Starting preview mode...")
        print("📝 Controls:")
        print("   Press 's' - Start recording")
        print("   Press 'q' - Quit preview")
        print("   Press 'r' - Stop recording (while recording)")

        self.is_previewing = True
        self.output_path = output_path

        # Initialize video capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise Exception(f"Could not open video source: {video_source}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        # Initialize video writer (will be used when recording starts)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        # Preview loop
        frame_count = 0
        recording_frame_count = 0

        try:
            while self.is_previewing:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame from video source")
                    break

                # Run YOLO inference
                results = self.yolo_model(frame, verbose=False, conf=0.5)
                annotated_frame = results[0].plot()

                # Add recording status to display
                status_text = "RECORDING" if self.is_recording else "PREVIEW - Press 's' to start recording"
                status_color = (0, 0, 255) if self.is_recording else (0, 255, 0)

                cv2.putText(annotated_frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                if self.is_recording:
                    # Add recording timer
                    elapsed_time = time.time() - self.recording_start_time
                    timer_text = f"Time: {elapsed_time:.1f}s | Frames: {recording_frame_count}"
                    cv2.putText(annotated_frame, timer_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Display frame
                cv2.imshow('Video Preview - Press s to record, q to quit', annotated_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("⏹️  Quitting preview...")
                    break

                elif key == ord('s') and not self.is_recording:
                    # Start recording
                    self._start_recording(cap, out, fourcc, frame_width, frame_height, fps)
                    recording_frame_count = 0
                    print("🔴 Recording started! Press 'r' to stop recording.")

                elif key == ord('r') and self.is_recording:
                    # Stop recording but continue preview
                    self._stop_recording(out)
                    print("⏹️  Recording stopped. Press 's' to record again or 'q' to quit.")

                # If recording, process and store frames
                if self.is_recording:
                    if out is None:
                        # Initialize video writer on first recording frame
                        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

                    # Write frame
                    out.write(annotated_frame)

                    # Process and store detections
                    self._process_detections(results, frame_count, recording_frame_count)
                    recording_frame_count += 1

                frame_count += 1

        except Exception as e:
            print(f"❌ Preview error: {e}")

        finally:
            # Cleanup
            self.is_previewing = False
            if self.is_recording:
                self._stop_recording(out)
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            print("✅ Preview session ended")

    def _start_recording(self, cap, out, fourcc, width, height, fps):
        """Start recording session"""
        self.is_recording = True
        self.recording_start_time = time.time()
        self.detection_history.clear()
        self.frame_buffer.clear()

        # Store video info
        self.video_info = {
            'width': width,
            'height': height,
            'fps': fps,
            'start_time': self.recording_start_time
        }

        print("🔴 Recording started!")

    def _stop_recording(self, out):
        """Stop recording session"""
        self.is_recording = False
        recording_duration = time.time() - self.recording_start_time

        if out is not None:
            out.release()

        self.video_info['duration'] = recording_duration
        self.video_info['end_time'] = time.time()

        print(f"⏹️  Recording stopped. Duration: {recording_duration:.1f}s")
        print(f"📊 Detected {len(self.detection_history)} object types")

    def _process_detections(self, results, global_frame_count, recording_frame_count):
        """Process and store detection data"""
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = results[0].names[cls]

                detection_data = {
                    'global_frame_id': global_frame_count,
                    'recording_frame_id': recording_frame_count,
                    'timestamp': time.time() - self.recording_start_time,
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class_id': cls
                }
                # Store in history
                self.detection_history[class_name].append(detection_data)

                # Store in frame buffer
                self.frame_buffer.append(detection_data)

    def start_preview_with_duration(self, video_source: Any = 0, duration: int = 10,
                                    output_path: str = "recorded_video.mp4") -> None:
        """
        Start preview with automatic recording duration

        Args:
            video_source: Camera index, video file, or RTSP stream
            duration: Recording duration in seconds (after pressing 's')
            output_path: Path to save recorded video
        """
        self.recording_duration = duration
        print(f"⏰ Recording duration set to {duration} seconds")
        self.start_preview(video_source, output_path)

    def _create_detection_summary(self) -> Dict[str, Any]:
        """Create comprehensive summary of detected objects"""
        if not self.detection_history:
            return {"error": "No detection data available"}

        summary = []
        total_detections = 0

        for class_name, detections in self.detection_history.items():
            if detections:
                timestamps = [d['timestamp'] for d in detections]
                confidences = [d['confidence'] for d in detections]

                summary.append({
                    'object': class_name,
                    'count': len(detections),
                    'first_seen': min(timestamps),
                    'last_seen': max(timestamps),
                    'duration': max(timestamps) - min(timestamps),
                    'avg_confidence': sum(confidences) / len(confidences),
                    'max_confidence': max(confidences)
                })
                total_detections += len(detections)

        # Sort by count (most frequent first)
        summary.sort(key=lambda x: x['count'], reverse=True)

        return {
            'total_objects_detected': total_detections,
            'unique_object_types': len(summary),
            'objects': summary,
            'recording_duration': self.video_info.get('duration', 0)
        }

    def _detect_activity_patterns(self) -> Dict[str, List[str]]:
        """Detect temporal patterns in object appearances"""
        patterns = {
            'continuous_objects': [],
            'transient_objects': [],
            'frequent_objects': [],
            'high_confidence_objects': []
        }

        for class_name, detections in self.detection_history.items():
            if len(detections) == 0:
                continue

            timestamps = [d['timestamp'] for d in detections]
            confidences = [d['confidence'] for d in detections]
            duration = max(timestamps) - min(timestamps)
            total_duration = self.video_info.get('duration', 10)

            if duration > total_duration * 0.7:
                patterns['continuous_objects'].append(class_name)

            if len(detections) < 3:
                patterns['transient_objects'].append(class_name)

            if len(detections) > 10:
                patterns['frequent_objects'].append(class_name)

            if np.mean(confidences) > 0.8:
                patterns['high_confidence_objects'].append(class_name)

        return patterns

    def analyze_video(self, question: str = None) -> Dict[str, Any]:
        """Analyze the recorded video using DeepSeek"""
        if not self.detection_history:
            return {"error": "No detection data available. Please record a video first."}

        print("🧠 Starting DeepSeek analysis...")
        self.is_processing = True

        try:
            detection_summary = self._create_detection_summary()
            activity_patterns = self._detect_activity_patterns()

            if question is None:
                question = "Based on the objects detected throughout the video, describe what was happening in the scene, the main activities, and any interesting patterns you observe."

            prompt = self._build_analysis_prompt(detection_summary, activity_patterns, question)

            print("🤔 DeepSeek is reasoning about the video...")
            start_time = time.time()
            analysis = self._call_deepseek(prompt)
            processing_time = time.time() - start_time

            results = {
                'analysis': analysis,
                'detection_summary': detection_summary,
                'activity_patterns': activity_patterns,
                'processing_time_seconds': round(processing_time, 2),
                'question': question
            }

            print(f"✅ Analysis completed in {processing_time:.2f} seconds")
            return results

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
        finally:
            self.is_processing = False

    def _build_analysis_prompt(self, detection_summary: Dict, activity_patterns: Dict, question: str) -> str:
        """Build comprehensive prompt for DeepSeek"""

        object_descriptions = []
        for obj in detection_summary.get('objects', []):
            desc = (f"- {obj['object']}: appeared {obj['count']} times, "
                    f"visible for {obj['duration']:.1f}s, "
                    f"confidence: {obj['avg_confidence']:.2f}")
            object_descriptions.append(desc)

        pattern_descriptions = []
        for pattern_type, objects in activity_patterns.items():
            if objects:
                readable_type = pattern_type.replace('_', ' ').title()
                pattern_descriptions.append(f"- {readable_type}: {', '.join(objects)}")

        prompt = f"""You are a dental expert AI assistant analyzing video content based on object detection data.

VIDEO CONTEXT:
Recording Duration: {detection_summary.get('recording_duration', 0):.1f} seconds
Total Detections: {detection_summary.get('total_objects_detected', 0)}
Unique Object Types: {detection_summary.get('unique_object_types', 0)}

DETECTED OBJECTS:
{chr(10).join(object_descriptions)}

ACTIVITY PATTERNS:
{chr(10).join(pattern_descriptions) if pattern_descriptions else "- No significant patterns detected"}

ANALYSIS TASK: {question}

Please provide:
1. An analysis of the  dental disease detcted and how to treat it or avoid it
2. Notable patterns or anomalies
3. Answer to the specific question

Reason step by step and be factual based on the detection data:"""

        return prompt

    def interactive_analysis(self) -> None:
        """Run interactive analysis session"""
        if not self.detection_history:
            print("❌ No video data available. Please record a video first.")
            return

        print("\n" + "=" * 60)
        print("🤖 INTERACTIVE VIDEO ANALYSIS")
        print("=" * 60)

        print("\n📊 Generating initial analysis...")
        results = self.analyze_video()

        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return

        self._print_results(results)

        while True:
            print("\n" + "-" * 40)
            question = input("\n💭 Ask a question about the video (or 'quit' to exit): ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            print("🤔 Thinking...")
            results = self.analyze_video(question)

            if 'error' in results:
                print(f"❌ Error: {results['error']}")
            else:
                print(f"\n💡 DeepSeek's Answer:")
                print(f"{results['analysis']}")

    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted results"""
        print(f"\n📋 DETECTION SUMMARY:")
        print(f"   Total objects detected: {results['detection_summary']['total_objects_detected']}")
        print(f"   Unique object types: {results['detection_summary']['unique_object_types']}")

        # Show top 5 objects
        print(f"\n🏆 TOP OBJECTS:")
        for i, obj in enumerate(results['detection_summary']['objects'][:5]):
            print(f"   {i + 1}. {obj['object']}: {obj['count']} appearances")

        print(f"\n🏃 ACTIVITY PATTERNS:")
        patterns_printed = False
        for pattern_type, objects in results['activity_patterns'].items():
            if objects:
                readable_type = pattern_type.replace('_', ' ').title()
                print(f"   {readable_type}: {', '.join(objects)}")
                patterns_printed = True
        if not patterns_printed:
            print("   No significant patterns detected")

        print(f"\nANALYSIS:")
        print(f"{results['analysis']}")

        print(f"\n⏱️  Processing time: {results['processing_time_seconds']}s")

    def save_analysis_report(self, results: Dict[str, Any], filename: str = "video_analysis_report.json") -> None:
        """Save analysis results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Analysis report saved to {filename}")

    def get_recording_status(self) -> Dict[str, Any]:
        """Get current recording status and statistics"""
        status = {
            'is_previewing': self.is_previewing,
            'is_recording': self.is_recording,
            'is_processing': self.is_processing,
            'detection_history_size': len(self.detection_history),
            'frame_buffer_size': len(self.frame_buffer)
        }

        if self.is_recording:
            status['recording_elapsed'] = time.time() - self.recording_start_time
            status['recording_duration'] = self.recording_duration

        return status


def main():
    """Main function demonstrating the refined pipeline"""
    # Initialize the reasoner
    reasoner = UnifiedVideoReasoner(
        yolo_model="./models/yolo8.pt",
        ollama_base_url="http://localhost:11434",
        deepseek_model="deepseek-r1:1.5b"
    )

    print("🎯 UNIFIED VIDEO REASONING PIPELINE")
    print("=" * 50)
    print("Senior Software Engineer Edition")
    print("Features:")
    print("  • Real-time preview with YOLO detection")
    print("  • Manual recording control (press 's' to start)")
    print("  • Interactive analysis with DeepSeek")
    print("  • Professional status display")
    print("=" * 50)

    while True:
        print("\n📋 MAIN MENU:")
        print("1. Start Preview (Manual Recording Control)")
        print("3. Analyze Last Recording")
        print("4. Interactive Analysis Session")
        print("5. Show Recording Status")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == "1":
            # Manual recording control
            output_file = input("Enter output filename (default: recorded_video.mp4): ").strip()
            if not output_file:
                output_file = "recorded_video.mp4"

            print(f"\n🚀 Starting preview mode...")
            print("   Press 's' to start recording")
            print("   Press 'r' to stop recording")
            print("   Press 'q' to quit preview")

            reasoner.start_preview(
                video_source=0,  # Default camera
                output_path=output_file
            )

            # Show status
            status = reasoner.get_recording_status()
            print(f"\n📊 CURRENT STATUS:")
            print(f"   Preview Active: {status['is_previewing']}")
            print(f"   Recording Active: {status['is_recording']}")
            print(f"   Processing Active: {status['is_processing']}")
            print(f"   Objects Detected: {status['detection_history_size']} types")
            print(f"   Frames in Buffer: {status['frame_buffer_size']}")

            if status['is_recording']:
                print(f"   Recording Time: {status['recording_elapsed']:.1f}s")

        elif choice == "3":
            # Analyze last recording
            if not reasoner.detection_history:
                print("❌ No recording data available. Please record a video first.")
                continue

            question = input("Enter your analysis question (or press Enter for general analysis): ").strip()

            results = reasoner.analyze_video(question if question else None)

            if 'error' not in results:
                reasoner._print_results(results)

                save = input("\n💾 Save analysis report? (y/n): ").strip().lower()
                if save == 'y':
                    filename = input("Enter filename (default: analysis_report.json): ").strip()
                    reasoner.save_analysis_report(results, filename or "analysis_report.json")

        elif choice == "4":
            # Interactive analysis
            if not reasoner.detection_history:
                print("❌ No recording data available. Please record a video first.")
                continue
            reasoner.interactive_analysis()


        elif choice == "6":
            print("👋 Exiting...")
            break

        else:
            print("❌ Invalid choice! Please enter 1-6.")


if __name__ == "__main__":
    main()