import numpy as np
from sklearn.metrics import jaccard_score
import cv2
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter

# Load detections
def load_detections(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    frames, ids, bb_left, bb_top, bb_width, bb_height, conf, x, y, z = data.T
    detections = []
    max_frame = int(np.max(frames))
    for i in range(max_frame + 1):
        detections.append([])
    for i in range(len(frames)):
        frame = int(frames[i])
        detection = {'id': ids[i], 'bb_left': bb_left[i], 'bb_top': bb_top[i], 
                     'bb_width': bb_width[i], 'bb_height': bb_height[i], 'conf': conf[i], 
                     'x': x[i], 'y': y[i], 'z': z[i]}
        detections[frame].append(detection)
    return detections


# Compute IoU
def compute_iou(box1, box2):
    intersection = np.maximum(0, np.minimum(box1[2], box2[2]) - np.maximum(box1[0], box2[0])) * \
                   np.maximum(0, np.minimum(box1[3], box2[3]) - np.maximum(box1[1], box2[1]))
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
    return intersection / union

# Associate detections to tracks
def associate_detections_to_tracks(detections, tracks, sigma_iou):
    for track in tracks:
        ious = [compute_iou(track['last_known_position'], detection) for detection in detections]
        best_match = np.argmax(ious)
        if ious[best_match] >= sigma_iou:
            track['detections'].append(detections[best_match])
            track['last_known_position'] = detections[best_match]
            detections.pop(best_match)

# Track management
def manage_tracks(detections, tracks, sigma_iou):
    # Associate detections to existing tracks
    associate_detections_to_tracks(detections, tracks, sigma_iou)
    
    # Delete unmatched tracks
    tracks = [track for track in tracks if len(track['detections']) > 0]
    
    # Create new tracks from unmatched detections
    for detection in detections:
        new_track = {'last_known_position': detection, 'detections': [detection]}
        tracks.append(new_track)
    
    return tracks

def manage_tracks_hungarian(detections, tracks, sigma_iou):
    # Compute the cost matrix (negative IoU)
    cost = -np.array([[compute_iou(track['last_known_position'], detection) for detection in detections] for track in tracks])

    # Solve the linear sum assignment problem
    row_indices, col_indices = linear_sum_assignment(cost)

    # Associate detections to existing tracks
    for row_index, col_index in zip(row_indices, col_indices):
        if -cost[row_index, col_index] >= sigma_iou:
            tracks[row_index]['detections'].append(detections[col_index])
            tracks[row_index]['last_known_position'] = detections[col_index]
            detections.pop(col_index)

    # Delete unmatched tracks
    tracks = [track for track in tracks if len(track['detections']) > 0]
    
    # Create new tracks from unmatched detections
    for detection in detections:
        new_track = {'id': len(tracks), 'last_known_position': detection, 'detections': [detection]}
        tracks.append(new_track)
    
    return tracks

def manage_tracks_kalman(detections, tracks, sigma_iou, kalman_filters):
    # Compute the cost matrix (negative IoU)
    cost = -np.array([[compute_iou(kalman_filter.predict()[0][:2], detection) for detection in detections] for kalman_filter in kalman_filters])

    # Solve the linear sum assignment problem
    row_indices, col_indices = linear_sum_assignment(cost)

    # Associate detections to existing tracks and update Kalman filters
    for row_index, col_index in zip(row_indices, col_indices):
        if -cost[row_index, col_index] >= sigma_iou:
            tracks[row_index]['detections'].append(detections[col_index])
            kalman_filters[row_index].update(detections[col_index])

    # Delete unmatched tracks and corresponding Kalman filters
    tracks = [track for track in tracks if len(track['detections']) > 0]
    kalman_filters = [kalman_filter for kalman_filter in kalman_filters if len(kalman_filter.x) > 0]
    
    # Create new tracks and corresponding Kalman filters for unmatched detections
    for detection in detections:
        new_track = {'id': len(tracks), 'last_known_position': detection, 'detections': [detection]}
        tracks.append(new_track)
        new_kalman_filter = KalmanFilter(dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas)
        new_kalman_filter.update(detection)
        kalman_filters.append(new_kalman_filter)
    
    return tracks, kalman_filters


detections = load_detections("ADL-Rundle-6\\det\\det.txt")
# Initialize an empty list of tracks
tracks = []
kalman_filters = []

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1920, 1080))

max_frame = 525
sigma_iou = 0.5

for frame in range(1, max_frame+1):
    # Load the corresponding image
    img = cv2.imread(f"ADL-Rundle-6\\img1\\{frame:06}.jpg")

    # Draw the detections and tracks
    if frame in detections:
        # Manage tracks
        tracks = manage_tracks_kalman(detections[frame], tracks, sigma_iou, kalman_filters)

    for track in tracks:
        # Draw bounding box for each detection in the track
        for detection in track['detections']:
            cv2.rectangle(img, (int(detection['bb_left']), int(detection['bb_top'])), 
                          (int(detection['bb_left'] + detection['bb_width']), 
                           int(detection['bb_top'] + detection['bb_height'])), (0, 255, 0), 2)
            
        # Draw trajectory
        for i in range(1, len(track['detections'])):
            if track['detections'][i - 1]['frame'] == frame - 1 and track['detections'][i]['frame'] == frame:
                cv2.line(img, (int(track['detections'][i - 1]['bb_left']), int(track['detections'][i - 1]['bb_top'])), 
                         (int(track['detections'][i]['bb_left']), int(track['detections'][i]['bb_top'])), (0, 0, 255), 2)

    # Write the frame
    out.write(img)
    with open('output.txt', 'w') as f:
        for track in tracks:
            for detection in track['detections']:
                f.write(f"{detection['frame']},{track['id']},{detection['bb_left']},{detection['bb_top']},{detection['bb_width']},{detection['bb_height']},{detection['conf']},{detection['x']},{detection['y']},{detection['z']}\n")

# Release everything
out.release()
cv2.destroyAllWindows()
