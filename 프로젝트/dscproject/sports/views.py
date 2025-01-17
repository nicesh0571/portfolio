from django.shortcuts import render
from django.http import JsonResponse
import os
import cv2
import numpy as np
import mediapipe as mp
from django.http import JsonResponse
from joblib import load as joblib_load

# Calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])  # Point A
    b = np.array([b.x, b.y])  # Point B (center point)
    c = np.array([c.x, c.y])  # Point C
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def test(request):
    # 여기에 필요한 로직 추가
    return JsonResponse({'message': '웹캠 실행 요청이 서버에서 처리되었습니다!'})

def testtest(request):
    if request.method == 'POST':
        # Fixed video path
        video_path = "dscproject/dscproject/sports/static/images/pullup.mp4"
        model_path = "dscproject/dscproject/sports/model/best_gbr_model.pkl"

        if not os.path.exists(video_path):
            return JsonResponse({"error": "Video file does not exist"}, status=404)

        if not os.path.exists(model_path):
            return JsonResponse({"error": "Model file does not exist"}, status=404)

        try:
            results = analyze_video(video_path, model_path)
        except Exception as e:
            return JsonResponse({"error": f"Analysis failed: {str(e)}"}, status=500)

        return JsonResponse({"results": results})

    return JsonResponse({"error": "Invalid request"}, status=400)

# Create your views here.
def index(request): 
    return render(request, 'v1/index.html')

def analyze_video(video_path, model_path):
    # Load the model
    linear_model = joblib_load(model_path)
    pose = mp.solutions.pose.Pose()

    # Store results
    results_text = []

    cam = cv2.VideoCapture(video_path)
    tolerance = 15  # Allowable deviation

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Calculate angles
            right_elbow_angle = calculate_angle(
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value],
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]
            )

            left_elbow_angle = calculate_angle(
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
            )

            right_shoulder_angle = calculate_angle(
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value]
            )

            left_shoulder_angle = calculate_angle(
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
            )

            # Predict ideal angles using the model
            input_data = np.array([[right_elbow_angle, left_elbow_angle]])
            predicted_output = linear_model.predict(input_data)

            ideal_posture = {
                "right_shoulder_angle": predicted_output[0][0],
                "left_shoulder_angle": predicted_output[0][1]
            }

            # Check for deviation
            right_diff = abs(right_shoulder_angle - ideal_posture['right_shoulder_angle'])
            left_diff = abs(left_shoulder_angle - ideal_posture['left_shoulder_angle'])

            if right_diff > tolerance or left_diff > tolerance:
                side = "오른쪽" if right_diff > left_diff else "왼쪽"
                results_text.append(f"{side} 부분에 불균형 존재.")

    cam.release()
    return results_text


