# importing OpenCV
import cv2
import time
import mediapipe as mp

# defining the webcam
cap = cv2.VideoCapture(0)

# getting the hands module into the program
mpHands = mp.solutions.hands
# the hands object only uses RGB images
hands = mpHands.Hands()
# drawing the lines which denote the detected points of the hands
mpDraw = mp.solutions.drawing_utils

# previous time
pTime = 0
# current time
cTime = 0

# Function to count fingers
def count_fingers(hand_landmarks):
    finger_count = 0
    # Write Your code here
    return finger_count

while True:
    success, img = cap.read()
    # getting the hand image and converting it to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # calling the object hands
    # method=process (process the image and give the results)
    results = hands.process(imgRGB)
    # displays the hand detection on a x,y,z graph
    # print(results.multi_hand_landmarks)

    # if a hand is detected
    if results.multi_hand_landmarks:
        # for each hand that is detected
        for handLms in results.multi_hand_landmarks:
            # getting landmark info
            for id, lm in enumerate(handLms.landmark):
                # id = index of hand/id of each dot 1 - 21 , lm = landmarks x,y,z cordinates
                ## print(id, lm)

                ### img.shape give the details of the webcam height width etc.
                h, w, c = img.shape
                ### initializing pixel values for x and y cordinates, instead of decimal nums.
                cx, cy = int(lm.x * w), int(lm.y * h)
                ## marking a specific dot in the matrix
                # if id== 0:
                # cv2.circle(img, (cx,cy),12,(255,45,25),  cv2.FILLED)

            ### img, handLms = draw the detection dots
            ### mpHands.HAND_CONNECTIONS = draws the connecting lines
            ### we draw these in img because imgRGB is just a converted pic and not the actual output
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Count fingers
            num_fingers = count_fingers(handLms)
            cv2.putText(img, f'Fingers: {num_fingers}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    ### calculating the frames per seconds
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Turning on the webcam
    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Function to find the largest contour in the frame
# def get_largest_contour(contours):
#     max_contour = max(contours, key=cv2.contourArea)
#     return max_contour

# # Function to count the number of fingers
# def count_fingers(contour, hull):
#     defects = cv2.convexityDefects(contour, hull)
#     if defects is None:
#         return 0

#     finger_count = 0
#     for i in range(defects.shape[0]):
#         s, e, f, d = defects[i, 0]
#         start = tuple(contour[s][0])
#         end = tuple(contour[e][0])
#         far = tuple(contour[f][0])
        
#         # Convert tuples to numpy arrays
#         start = np.array(start)
#         end = np.array(end)
#         far = np.array(far)
        
#         a = np.linalg.norm(end - start)
#         b = np.linalg.norm(far - start)
#         c = np.linalg.norm(far - end)
        
#         angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * (180 / np.pi)

#         if angle <= 90:
#             finger_count += 1
    
#     return finger_count

# # Capture video from the webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame horizontally
#     frame = cv2.flip(frame, 1)
    
#     # Define the region of interest (ROI) for hand detection
#     roi = frame[100:400, 100:400]
#     cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

#     # Convert the ROI to grayscale
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur
#     blur = cv2.GaussianBlur(gray, (35, 35), 0)

#     # Threshold the image
#     _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_contour = get_largest_contour(contours)
#         hull = cv2.convexHull(largest_contour, returnPoints=False)

#         # Draw contours
#         cv2.drawContours(roi, [largest_contour], -1, (0, 255, 0), 3)
        
#         # Count fingers
#         if hull is not None:
#             finger_count = count_fingers(largest_contour, hull)
#             cv2.putText(frame, f'Fingers: {finger_count+1}', (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
#     # Display the resulting frame
#     cv2.imshow('Hand Gesture Recognition', frame)
    
#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
