import cv2
import mediapipe as mp
import numpy as np
import time
import os
import tkinter as tk
from tkinter import Scrollbar, ttk

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# --- Adjustable Parameters (Constants) ---
ANGLE_THRESHOLDS = {
    "severe_slouching": 85,
    "slouching": 90,
    "good_posture": 105,
}
SHOULDER_Y_DIFF_THRESHOLD = 0.05
HIP_Y_DIFF_THRESHOLD = 0.1
NOSE_TO_SHOULDER_DIST_THRESHOLD = -0.03
NECK_ANGLE_THRESHOLD = 80
SHOULDER_HIP_ANGLE_THRESHOLD = 85
HIP_KNEE_ANKLE_ANGLE_THRESHOLD = 170
SHOULDER_ELBOW_WRIST_ANGLE_THRESHOLD = 160
LEAN_Z_THRESHOLD = 0.1
ANGLE_HISTORY_LENGTH = 5
SPINE_CURVATURE_THRESHOLD = 10
SPINE_TILT_THRESHOLD = 5
THUMBS_UP_DELAY = 3
THUMBS_UP_COOLDOWN = 3
AUTO_CAPTURE_DELAY = 10

# --- Angles to Capture ---
REQUIRED_ANGLES = ["Front", "Side", "Back"]

# --- Global Variables ---
image_counter = 0
last_thumbs_up_time = 0
thumbs_up_detected = False
last_capture_time = 0
captured_images = []
current_angle_index = 0
analysis_complete = False

def midpoint(p1, p2):
    """Calculates the midpoint between two points."""
    if isinstance(p1, tuple):
        p1_x, p1_y = p1
    else:
        p1_x, p1_y = p1.x, p1.y

    if isinstance(p2, tuple):
        p2_x, p2_y = p2
    else:
        p2_x, p2_y = p2.x, p2.y

    return ((p1_x + p2_x) / 2, (p1_y + p2_y) / 2)

def angle_between(p1, p2, p3):
    """Calculates the angle between three points."""
    if isinstance(p1, tuple):
        a = np.array(p1)
    else:
        a = np.array((p1.x, p1.y))

    if isinstance(p2, tuple):
        b = np.array(p2)
    else:
        b = np.array((p2.x, p2.y))

    if isinstance(p3, tuple):
        c = np.array(p3)
    else:
        c = np.array((p3.x, p3.y))

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1, 1)
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def check_angle(landmarks, frame, required_angle):
    """Checks if the user is in the correct angle."""
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    frame_height, frame_width, _ = frame.shape
    center_x = frame_width // 2

    if required_angle == "Front":
        if abs(nose.x * frame_width - center_x) > frame_width * 0.1:
            return False
        if l_shoulder.visibility < 0.5 or r_shoulder.visibility < 0.5:
            return False
        return True
    elif required_angle == "Side":
        if l_shoulder.visibility < 0.5 and r_shoulder.visibility < 0.5:
            return False
        if abs(nose.x * frame_width - center_x) < frame_width * 0.3:
            return False
        return True
    elif required_angle == "Back":
        if nose.visibility > 0.5:
            return False
        if l_shoulder.visibility < 0.5 or r_shoulder.visibility < 0.5:
            return False
        return True
    return False

angle_history = []
def analyze_posture(landmarks, frame):
    """Analyzes posture based on landmarks."""
    global angle_history
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    r_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    hip_center = midpoint((l_hip.x, l_hip.y), (r_hip.x, r_hip.y))
    neck = midpoint((l_shoulder.x, l_shoulder.y), (r_shoulder.x, r_shoulder.y))
    neck_midpoint = midpoint((l_ear.x, l_ear.y), (r_ear.x, r_ear.y))

    # Enhanced Spine Points
    spine_top = neck
    spine_mid_upper = midpoint(neck, midpoint(neck, hip_center))
    spine_mid_lower = midpoint(midpoint(neck, hip_center), hip_center)
    spine_bottom = hip_center

    # Calculate spine angles
    spine_angle1 = angle_between(spine_top, spine_mid_upper, spine_mid_lower)
    spine_angle2 = angle_between(spine_mid_upper, spine_mid_lower, spine_bottom)

    # --- Posture Analysis ---
    shoulder_y_diff = abs(l_shoulder.y - r_shoulder.y)
    hip_y_diff = abs(l_hip.y - r_hip.y)

    frame_height, frame_width, _ = frame.shape
    vertical_reference = (frame_width // 2, 0)
    angle = angle_between(hip_center, neck, vertical_reference)
    angle_history.append(angle)
    if len(angle_history) > ANGLE_HISTORY_LENGTH:
        angle_history.pop(0)
    smoothed_angle = sum(angle_history) / len(angle_history)

    posture_issues = []

    if smoothed_angle < ANGLE_THRESHOLDS["severe_slouching"]:
        posture_issues.append("Severe Slouching")
    elif smoothed_angle < ANGLE_THRESHOLDS["slouching"]:
        posture_issues.append("Slouching")
    elif smoothed_angle < ANGLE_THRESHOLDS["good_posture"]:
        posture_issues.append("Good Posture")
    else:
        posture_issues.append("Overextended")

    if shoulder_y_diff > SHOULDER_Y_DIFF_THRESHOLD:
        posture_issues.append("Uneven Shoulders")

    if hip_y_diff > HIP_Y_DIFF_THRESHOLD:
        posture_issues.append("Uneven Hips")

    neck_x, neck_y = neck
    nose_to_shoulder_dist = nose.x - neck_x
    neck_angle = angle_between(neck_midpoint, neck, vertical_reference)
    if nose_to_shoulder_dist < NOSE_TO_SHOULDER_DIST_THRESHOLD or neck_angle < NECK_ANGLE_THRESHOLD:
        posture_issues.append("Possible Forward Head")

    left_shoulder_hip_angle = angle_between((l_shoulder.x, l_shoulder.y), (l_hip.x, l_hip.y), vertical_reference)
    right_shoulder_hip_angle = angle_between((r_shoulder.x, r_shoulder.y), (r_hip.x, r_hip.y), vertical_reference)
    if left_shoulder_hip_angle < SHOULDER_HIP_ANGLE_THRESHOLD or right_shoulder_hip_angle < SHOULDER_HIP_ANGLE_THRESHOLD:
        posture_issues.append("Possible Rounded Shoulders")

    left_hip_knee_ankle_angle = angle_between((l_hip.x, l_hip.y), (l_knee.x, l_knee.y), (l_ankle.x, l_ankle.y))
    right_hip_knee_ankle_angle = angle_between((r_hip.x, r_hip.y), (r_knee.x, r_knee.y), (r_ankle.x, r_ankle.y))
    if left_hip_knee_ankle_angle > HIP_KNEE_ANKLE_ANGLE_THRESHOLD or right_hip_knee_ankle_angle > HIP_KNEE_ANKLE_ANGLE_THRESHOLD:
        posture_issues.append("Possible Swayback")

    left_shoulder_elbow_wrist_angle = angle_between((l_shoulder.x, l_shoulder.y), (l_elbow.x, l_elbow.y),
                                                    (l_wrist.x, l_wrist.y))
    right_shoulder_elbow_wrist_angle = angle_between((r_shoulder.x, r_shoulder.y), (r_elbow.x, r_elbow.y),
                                                     (r_wrist.x, r_wrist.y))
    if left_shoulder_elbow_wrist_angle < SHOULDER_ELBOW_WRIST_ANGLE_THRESHOLD or right_shoulder_elbow_wrist_angle < SHOULDER_ELBOW_WRIST_ANGLE_THRESHOLD:
        posture_issues.append("Possible Kyphosis")

    nose_z = nose.z
    shoulder_z = (l_shoulder.z + r_shoulder.z) / 2
    if nose_z < shoulder_z - LEAN_Z_THRESHOLD:
        posture_issues.append("Leaning Forward")
    elif nose_z > shoulder_z + LEAN_Z_THRESHOLD:
        posture_issues.append("Leaning Backward")

    # Spine curvature check
    if spine_angle1 < 180 - SPINE_CURVATURE_THRESHOLD:
        posture_issues.append("Spine Curvature Detected")

    # Spine tilt check
    if abs(spine_angle2 - 90) > SPINE_TILT_THRESHOLD:
        posture_issues.append("Spine Tilt Detected")

    posture_text = ", ".join(posture_issues) if posture_issues else "No Issues Detected"

    return posture_text, posture_issues

def analyze_multiple_images(images):
    """Analyzes posture from multiple images."""
    all_posture_data = []
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7) as pose:
        for image in images:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)
            if pose_results.pose_landmarks:
                posture_text, posture_issues = analyze_posture(pose_results.pose_landmarks.landmark, image)
                all_posture_data.append({"posture_text": posture_text, "posture_issues": posture_issues})
    return all_posture_data

# Dictionary mapping posture issues to their causes and fixes
posture_fixes = {
    "Severe Slouching": {
        "causes": "Weak core muscles, Prolonged sitting, Poor ergonomic habits",
        "fixes": [
            "Bruegger's Postural Relief: Sit at the edge of a chair, pull shoulders back, and squeeze shoulder blades together for 30 seconds.",
            "Thoracic Extension Stretch: Sit in a chair with a backrest, place hands behind your head, lean back, and stretch your spine.",
            "Plank Variations: Standard, side, or reverse plank to strengthen core stability.",
            "Wall Angels: Stand against a wall, slide arms up and down while keeping the lower back against the wall.",
            "Ergonomic Fixes: Adjust chair height, sit with feet flat, use lumbar support."
        ]
    },
    "Slouching": {
        "causes": "Weak core muscles, Prolonged sitting, Poor ergonomic habits",
        "fixes": [
            "Bruegger's Postural Relief: Sit at the edge of a chair, pull shoulders back, and squeeze shoulder blades together for 30 seconds.",
            "Thoracic Extension Stretch: Sit in a chair with a backrest, place hands behind your head, lean back, and stretch your spine.",
            "Plank Variations: Standard, side, or reverse plank to strengthen core stability.",
            "Wall Angels: Stand against a wall, slide arms up and down while keeping the lower back against the wall.",
            "Ergonomic Fixes: Adjust chair height, sit with feet flat, use lumbar support."
        ]
    },
    "Uneven Shoulders": {
        "causes": "Muscle imbalances, Favoring one side, Weak upper back muscles",
        "fixes": [
            "Scapular Retraction: Squeeze shoulder blades together and hold for 5 seconds.",
            "Dumbbell Shoulder Shrugs: Hold a dumbbell in each hand and shrug shoulders up, hold, then slowly lower.",
            "Single-Arm Farmers Carry: Carry a weight in one hand while walking to correct imbalances.",
            "Ergonomic Fixes: Avoid carrying heavy bags on one side, adjust sitting posture."
        ]
    },
    "Possible Forward Head": {
        "causes": "Excessive phone or screen time, Weak neck muscles, Poor sitting habits",
        "fixes": [
            "Chin Tucks: Gently tuck the chin toward the chest while keeping the back of the head aligned.",
            "Neck Stretches: Tilt head side to side and forward/backward to stretch neck muscles.",
            "Wall Posture Check: Stand against a wall, ensure head, shoulders, and glutes touch the wall.",
            "Ergonomic Fixes: Raise monitor to eye level, avoid looking down at devices."
        ]
    },
    "Possible Rounded Shoulders": {
        "causes": "Weak upper back, Overactive chest muscles, Excessive desk work",
        "fixes": [
            "Reverse Flys: Hold light weights, bend forward slightly, and raise arms outward.",
            "Chest Stretches: Stand in a doorway, place arms at 90 degrees, and lean forward.",
            "Face Pulls: Use resistance bands to pull towards your face, engaging rear delts.",
            "Ergonomic Fixes: Open chest, sit upright, avoid hunching."
        ]
    },
    "Possible Kyphosis": {
        "causes": "Poor posture over time, Weak spinal erectors, Excessive sitting",
        "fixes": [
            "Cobra Stretch: Lie on stomach, push up with hands, and extend the spine.",
            "Superman Exercise: Lie on the stomach, lift arms and legs off the ground for core engagement.",
            "Deadlifts (Light Weights): Strengthen posterior chain and spinal muscles.",
            "Ergonomic Fixes: Use a standing desk, adjust seat posture."
        ]
    },
    "Possible Swayback": {
        "causes": "Weak core, Overly tight hip flexors, Prolonged standing in poor posture",
        "fixes": [
            "Glute Bridges: Lie on your back, lift hips while squeezing glutes.",
            "Pelvic Tilts: Lay flat, press the lower back into the floor, hold, and release.",
            "Cat-Cow Stretch: Perform spinal flexion/extension while on hands and knees.",
            "Ergonomic Fixes: Strengthen the core and avoid prolonged arching posture."
        ]
    },
    "Leaning Forward": {
        "causes": "Muscle imbalances, Incorrect standing or sitting habits, Weak lower back or core",
        "fixes": [
            "Dead Bug Exercise: Lie on back, move opposite arm and leg down slowly.",
            "Standing Balance Training: Stand on one leg, shift weight forward/backward to train balance.",
            "Heel-to-Toe Walking: Walk in a straight line placing heel in front of the toe to improve posture control.",
            "Ergonomic Fixes: Ensure weight is evenly distributed when standing."
        ]
    },
    "Leaning Backward": {
        "causes": "Muscle imbalances, Incorrect standing or sitting habits, Weak lower back or core",
        "fixes": [
            "Dead Bug Exercise: Lie on back, move opposite arm and leg down slowly.",
            "Standing Balance Training: Stand on one leg, shift weight forward/backward to train balance.",
            "Heel-to-Toe Walking: Walk in a straight line placing heel in front of the toe to improve posture control.",
            "Ergonomic Fixes: Ensure weight is evenly distributed when standing."
        ]
    },
    "Spine Curvature": {
        "causes": "Genetics (for scoliosis), Poor posture habits, Muscle imbalances",
        "fixes": [
            "Side Planks (on the weaker side): Strengthens muscles and evens out spinal imbalances.",
            "Seated Spinal Twists: Sit cross-legged and twist the upper body to stretch and align the spine.",
            "Lat Stretch with Foam Roller: Stretch tight lat muscles which may pull on the spine.",
            "Medical Fixes: Severe cases require physiotherapy or a brace."
        ]
    },
    "Spine Tilt": {
        "causes": "Weak obliques/core, Asymmetrical muscle tightness, Improper weight distribution",
        "fixes": [
            "Side Planks: Strengthens obliques to correct imbalances.",
            "Side Stretching: Stand and reach one arm overhead to stretch the sides.",
            "Weighted Side Bends: Hold a weight in one hand and bend sideways.",
            "Ergonomic Fixes: Avoid sitting cross-legged too often, keep equal weight on both feet."
        ]
    }
}

def display_posture_defects(posture_issues):
    """Displays posture defects in a new window with exercises and treatments."""
    window = tk.Tk()
    window.title("Posture Defects and Fixes")

    style = ttk.Style()
    style.configure("TLabel", font=("Arial", 12), padding=10)
    style.configure("TFrame", background="#f0f0f0")

    # Create a Canvas to hold the content
    canvas = tk.Canvas(window)
    canvas.pack(side="left", fill="both", expand=True)

    # Add a Scrollbar
    scrollbar = Scrollbar(window, command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    # Configure the Canvas
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Create a Frame inside the Canvas
    main_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=main_frame, anchor="nw")

    if posture_issues:
        label = ttk.Label(main_frame, text="Posture Defects and Fixes:", font=("Arial", 14, "bold"))
        label.pack()

        for issue in posture_issues:
            issue_frame = ttk.Frame(main_frame)
            issue_frame.pack(fill="x", padx=10, pady=10)

            issue_label = ttk.Label(issue_frame, text=f"{issue}:", font=("Arial", 12, "bold"))
            issue_label.pack()

            causes_label = ttk.Label(issue_frame, text=f"Cause: {posture_fixes.get(issue, {}).get('causes', 'Unknown')}", wraplength=400)
            causes_label.pack()

            fixes_label = ttk.Label(issue_frame, text="Fixes:", font=("Arial", 12, "bold"))
            fixes_label.pack()

            fixes_text = "\n".join(posture_fixes.get(issue, {}).get('fixes', ['No fixes available.']))
            fixes_text_label = ttk.Label(issue_frame, text=fixes_text, wraplength=400, justify="left")
            fixes_text_label.pack()

    else:
        label = ttk.Label(main_frame, text="No Posture Defects Detected", font=("Arial", 14, "bold"))
        label.pack()

    # Update the scroll region after adding all content
    main_frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

    # Center the window on the screen
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    window.mainloop()



def analyze_hand_gesture(results, frame, pose_results):
    """Analyzes hand gestures."""
    global image_counter, last_thumbs_up_time, thumbs_up_detected, last_capture_time, captured_images, current_angle_index, analysis_complete
    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Thumbs-up gesture detection
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

            if thumb_tip.y < thumb_mcp.y and thumb_tip.y < thumb_ip.y:
                cv2.putText(frame, "Thumbs Up Detected", (20, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                if not thumbs_up_detected:
                    thumbs_up_detected = True
                    last_thumbs_up_time = current_time

                if thumbs_up_detected and current_time - last_thumbs_up_time >= THUMBS_UP_DELAY:
                    if current_time - last_capture_time >= THUMBS_UP_COOLDOWN:
                        if pose_results.pose_landmarks:
                            if not check_angle(pose_results.pose_landmarks.landmark, frame, REQUIRED_ANGLES[current_angle_index]):
                                cv2.putText(frame, f"Warning: Please adjust to {REQUIRED_ANGLES[current_angle_index]} View", (20, 180), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
                            # Capture image
                            captured_frame = frame.copy()
                            captured_images.append(captured_frame)
                            image_counter += 1
                            filename = f"captured_image_{image_counter}_{REQUIRED_ANGLES[current_angle_index]}.jpg"
                            cv2.imwrite(filename, captured_frame)  # Save the image
                            print(f"Image {image_counter} ({REQUIRED_ANGLES[current_angle_index]}) captured and saved as {filename}.")

                            # Save posture text to a file
                            posture_text, _ = analyze_posture(pose_results.pose_landmarks.landmark, frame)
                            text_filename = f"posture_log_{image_counter}_{REQUIRED_ANGLES[current_angle_index]}.txt"
                            with open(text_filename, "w") as log_file:
                                log_file.write(posture_text)
                            print(f"Posture text saved to {text_filename}")

                            last_capture_time = current_time
                            thumbs_up_detected = False
                            current_angle_index += 1

            else:
                thumbs_up_detected = False

    if current_angle_index >= len(REQUIRED_ANGLES):  # Analyze after capturing all angles
        posture_results = analyze_multiple_images(captured_images)
        print("Posture Analysis Results:")
        all_issues = []
        for i, data in enumerate(posture_results):
            print(f"  Image {i+1} ({REQUIRED_ANGLES[i]} View):")
            print(f"    Posture Text: {data['posture_text']}")
            print(f"    Posture Issues: {data['posture_issues']}")
            all_issues.extend(data['posture_issues'])
        display_posture_defects(all_issues)
        captured_images.clear()  # Reset captured images
        current_angle_index = 0  # Reset for next round
        
# Video capture loop (60 FPS)
def video_loop():
    """Captures video, processes frames, and analyzes posture."""
    global analysis_complete
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Pose Estimation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose, mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # flip the frame to correct the image.

            pose_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if pose_results.pose_landmarks:
                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                posture, _ = analyze_posture(pose_results.pose_landmarks.landmark, frame)
                cv2.putText(frame, posture, (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            else:
                cv2.putText(frame, "No Pose Detected", (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

            # --- New: Prompt User for Angle ---
            if current_angle_index < len(REQUIRED_ANGLES):
                cv2.putText(frame, f"Turn to {REQUIRED_ANGLES[current_angle_index]} View", (20, 150), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 255), 1)

            analyze_hand_gesture(hand_results, frame, pose_results)

            cv2.imshow('Pose Estimation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if analysis_complete:
                analysis_complete = False
                
    cap.release()
    cv2.destroyAllWindows()

# Start the video loop
video_loop()

