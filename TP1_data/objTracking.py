import numpy as np
import cv2
from Detector import detect
from KalmanFilter import KalmanFilter

# Set parameters for Kalman filter
dt = 0.1
u_x = 1
u_y = 1
std_acc = 1
x_dt_meas = 0.1
y_dt_meas = 0.1

# Create KalmanFilter object
kf = KalmanFilter(dt, u_x, u_y, std_acc, x_dt_meas, y_dt_meas)

# Start video capture
cap = cv2.VideoCapture('TP1_data/randomball.avi')

# Initialize variables
center = None
trajectory = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect circles in the frame
    if frame is None:
        break
    centers = detect(frame)

    # If a circle is detected, track it using Kalman filter
    if len(centers) > 0:
        # Get the centroid of the detected circle
        center = centers[0].T

        # Predict the object's position using 
        pos_pred, _ = kf.predict()
        # Update the object's position using Kalman filter's update function
        kf.update(center)

        # Draw the detected circle in green color
        cv2.circle(frame, (int(center[0][0]), int(center[0][1])), 3, (0, 255, 0), -1)

        # Draw the predicted object position in blue color
        x_pred, y_pred = pos_pred[0], pos_pred[1]
        print(pos_pred)
        cv2.rectangle(frame, (int(x_pred - 5), int(y_pred - 5)), (int(x_pred + 5), int(y_pred + 5)), (255, 0, 0), 2)

        # Draw the estimated object position in red color
        x_est, y_est = kf.x[0], kf.x[1]
        cv2.rectangle(frame, (int(x_est - 5), int(y_est - 5)), (int(x_est + 5), int(y_est + 5)), (0, 0, 255), 2)

        # Add the current object position to the trajectory list
        trajectory.append([x_est, y_est])

    # Draw the trajectory
    for i in range(0, len(trajectory), 2):
        if i + 1 < len(trajectory):
            x_0, y_0 = trajectory[i]
            x_1, y_1 = trajectory[i + 1]
            cv2.line(frame, (int(x_0), int(y_0)), (int(x_1), int(y_1)), (255, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('Object Tracking', frame)

    # Check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
