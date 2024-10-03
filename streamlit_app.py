import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

st.title('Markerless MediaPipe')

# Create two columns
col1, col2 = st.columns(2)

with col1:
    stframe = st.empty()

with col2:
    stangle = st.empty()

# Dropdowns for selecting landmarks
landmark_options = [landmark.name for landmark in mp_pose.PoseLandmark]
landmark_a = st.selectbox('Select first landmark', landmark_options, index=mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
landmark_b = st.selectbox('Select middle landmark', landmark_options, index=mp_pose.PoseLandmark.RIGHT_ELBOW.value)
landmark_c = st.selectbox('Select end landmark', landmark_options, index=mp_pose.PoseLandmark.RIGHT_WRIST.value)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

            # Extract selected landmarks
            point_a = landmarks[mp_pose.PoseLandmark[landmark_a].value]
            point_b = landmarks[mp_pose.PoseLandmark[landmark_b].value]
            point_c = landmarks[mp_pose.PoseLandmark[landmark_c].value]

            # Calculate the angle
            angle = calculate_angle(point_a, point_b, point_c)

            # Display the angle
            stangle.markdown(f"<h1 style='color: black;'>{angle:.2f}°</h1>", unsafe_allow_html=True)

            # Draw the pose annotations on the image.
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Using default webcam
webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
