import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import copy
import itertools
import csv
import logging
from model import KeyPointClassifier
from spellchecker import SpellChecker

# Set up logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()
log.setLevel(logging.INFO)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load classifiers
keypoint_classifier = KeyPointClassifier()

# Load classifier labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

info = []
letters = []
result_string = ""
last_word = ""
last_hand_sign_text = ""

def append_bounded(lst, max_len, item):
    if len(lst) == 0 and item == " ":  # Ensure first appended item is not a space
        return
    lst.append(item)
    while len(lst) > max_len:
        lst.pop(0)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list))
    return [n / max_value for n in temp_landmark_list]

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    global last_hand_sign_text, result_string, last_word
    if handedness is not None:
        info_text = handedness.classification[0].label[0:]
    else:
        info_text = ""
    append_bounded(info, 30, hand_sign_text)
    info_text = info_text + ':' + hand_sign_text

    count = sum(1 for sign in info[-13:] if sign == hand_sign_text)
    if count >= 9 and (len(letters) == 0 or hand_sign_text != letters[-1]):
        letters.append(hand_sign_text)
    count1 = sum(1 for sign in info[-30:] if sign == hand_sign_text)
    if count1 >= 27 and hand_sign_text != " " and (len(letters) < 2 or hand_sign_text != letters[-2]):
        letters.append(hand_sign_text)
    
    current_string = ''.join(letters)
    if letters and letters[-1] == " ":
        words = current_string.strip().split()
        if words:
            temp_word = words[-1]
            if last_word != temp_word:
                last_word = temp_word
                corrected_word = SpellChecker().correction(last_word)
                result_string += corrected_word + " "
    
    cv.putText(image, "Current String: " + current_string, (10, 60), cv.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(image, "Result String: " + result_string, (10, 120), cv.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv.LINE_AA)
    return image

# Open webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    log.error("Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        log.error("Failed to grab frame.")
        break
    
    # Flip and convert to RGB
    frame = cv.flip(frame, 1)
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = calc_landmark_list(frame, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            hand_sign_text = keypoint_classifier_labels[hand_sign_id]
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        hand_sign_text = " "  # Set to space when no hand is detected
    
    frame = draw_info_text(frame, [10, 100, 300, 150], results.multi_handedness[0] if results.multi_handedness else None, hand_sign_text, "")
    
    cv.imshow('Hand Gesture Recognition', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()