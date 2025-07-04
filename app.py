# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import atexit

from realtime_code import (
    extract_keypoints_from_frame, 
    predict_pose, 
    get_pose_corrections,
    mp_drawing,  # ✅ Import drawing utils
    mp_pose      # ✅ Import pose model
)

import sys
print("🔧 Python version:", sys.version)

# Streamlit config
st.set_page_config(page_title="Yoga Pose Detection", layout="centered")
st.title("🧘‍♀️ Real-Time Yoga Pose Detection with GCN + LSTM")
st.markdown("Supports **Webcam**, **Image**, and **Video** input for yoga pose prediction and correction.")

# Select mode
mode = st.radio("Choose input mode:", ["Webcam", "Upload Image", "Upload Video"])

# 🟢 Webcam Mode
if mode == "Webcam":
    st.warning("Click 'Start' to use your webcam for live pose prediction.")
    run = st.button("Start Webcam")

    if run:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            keypoints, results, full_body_visible = extract_keypoints_from_frame(frame)

            if keypoints is not None:
                # ✅ Draw landmarks on frame
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

                pose_name, confidence = predict_pose(keypoints)
                corrections = get_pose_corrections(results.pose_landmarks, pose_name)

                text = f"Pose: {pose_name} ({confidence*100:.1f}%)"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                y_offset = 60
                if corrections:
                    for correction in corrections[:3]:
                        y_offset += 25
                        cv2.putText(frame, f"- {correction}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
                else:
                    cv2.putText(frame, "Perfect Pose!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if not full_body_visible:
                    cv2.putText(frame, "⚠️ Partial body detected", (10, frame.shape[0]-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()

# 🟡 Upload Image Mode
elif mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload a yoga pose image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        keypoints, results, full_body_visible = extract_keypoints_from_frame(frame)

        if keypoints is not None:
            # ✅ Draw landmarks on frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            pose_name, confidence = predict_pose(keypoints)
            corrections = get_pose_corrections(results.pose_landmarks, pose_name)

            st.success(f"Predicted Pose: **{pose_name}** with {confidence:.1%} confidence")

            if not full_body_visible:
                st.warning("⚠️ Partial body detected — prediction may be less accurate.")

            if corrections:
                st.subheader("🔧 Corrections:")
                for corr in corrections:
                    st.markdown(f"- {corr}")
            else:
                st.info("✅ Perfect Pose!")
        else:
            st.error("❌ Unable to extract keypoints from the image.")

        st.image(frame[..., ::-1], caption="Uploaded Image with Landmarks", use_container_width=True)

# 🔵 Upload Video Mode
elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a yoga video file", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        # Register cleanup at app exit
        @atexit.register
        def cleanup_temp():
            try:
                os.unlink(video_path)
            except Exception:
                pass

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()
        progress = st.progress(0, text="Processing video...")
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            keypoints, results, full_body_visible = extract_keypoints_from_frame(frame)

            if keypoints is not None:
                # ✅ Draw landmarks on frame
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

                pose_name, confidence = predict_pose(keypoints)
                corrections = get_pose_corrections(results.pose_landmarks, pose_name)

                text = f"{pose_name} ({confidence*100:.1f}%)"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                y_offset = 60
                if corrections:
                    for corr in corrections[:2]:
                        y_offset += 25
                        cv2.putText(frame, f"- {corr}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(frame, "Perfect Pose!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if not full_body_visible:
                    cv2.putText(frame, "⚠️ Partial body detected", (10, frame.shape[0]-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            out.write(frame)
            stframe.image(frame, channels="BGR", use_container_width=True)

            frame_idx += 1
            progress.progress(min(frame_idx / total_frames, 1.0), text=f"Processing... {frame_idx}/{total_frames}")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        time.sleep(1)
        try:
            os.unlink(video_path)
        except PermissionError:
            pass

        st.success("✅ Video processing complete!")
        with open(output_path, "rb") as f:
            st.download_button("📥 Download Annotated Video", f, file_name="yoga_pose_output.mp4")
