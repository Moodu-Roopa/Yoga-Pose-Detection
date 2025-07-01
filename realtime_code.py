import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Module, Dropout, BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
import mediapipe as mp
import time
from torch.nn import LSTM

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DeepGCN_ResMLP_LSTM(Module):
    def __init__(self, in_channels, hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.gcn4 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.bn4 = BatchNorm1d(hidden_dim)

        self.dropout = Dropout(0.3)

        # 🔁 Replaced BiLSTM with LSTM (not bidirectional)
        self.lstm = LSTM(hidden_dim, hidden_dim, num_layers=2,
                         bidirectional=False, batch_first=True, dropout=0.25)

        self.classifier = torch.nn.Sequential(
            Linear(hidden_dim * 3 + hidden_dim, hidden_dim * 4),
            BatchNorm1d(hidden_dim * 4),
            torch.nn.ReLU(),
            Dropout(0.5),
            Linear(hidden_dim * 4, hidden_dim * 2),
            BatchNorm1d(hidden_dim * 2),
            torch.nn.ReLU(),
            Dropout(0.4),
            Linear(hidden_dim * 2, hidden_dim),
            BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            Dropout(0.3),
            Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # Residual GCN layers
        res1 = F.relu(self.bn1(self.gcn1(x, edge_index)))
        res2 = F.relu(self.bn2(self.gcn2(res1, edge_index))) + res1
        res3 = F.relu(self.bn3(self.gcn3(res2, edge_index))) + res2
        res4 = F.relu(self.bn4(self.gcn4(res3, edge_index))) + res3

        # Prepare sequence input for LSTM
        x_split = torch.split(res4, torch.bincount(batch).tolist())
        x_padded = torch.nn.utils.rnn.pad_sequence(x_split, batch_first=True)
        lengths = [x.shape[0] for x in x_split]

        packed = torch.nn.utils.rnn.pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Last valid timestep output per sequence
        final_outputs = [unpacked[i, length - 1, :] for i, length in enumerate(lengths)]
        lstm_features = torch.stack(final_outputs)

        # Global Pooling
        pooled = torch.cat([
            global_mean_pool(res4, batch),
            global_max_pool(res4, batch),
            global_add_pool(res4, batch)
        ], dim=1)

        fused = torch.cat([pooled, lstm_features], dim=1)
        return self.classifier(fused)
    
# ✅ STEP 4: Class names (same order as training)
class_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']    
# class_names = ['Boat', 'Bridge', 'Cat', 'Chair', 'Cobra', 'Downdog', 'Goddess', 'Headstand', 'Plank', 'Triangle','Tree', 'Warrior I', 'Warrior II']
# class_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior2', 'warrior1', 'headstand', 'cat', 'bridge', 'boat', 'triangle', 'cobra', 'chair', 'standing']


# Model definition with LSTM
model = DeepGCN_ResMLP_LSTM(in_channels=3, hidden_dim=128, num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("best_gcn_residual_mlp_lstm_model.pth", map_location=device))
model.eval()
print("✅ Advanced model loaded successfully!")

# ✅ STEP 3: Enhanced anatomical edge connections (same as training)
def create_anatomical_edge_index(num_nodes=33):
    # MediaPipe pose connections (anatomically correct)
    connections = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # Arms
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        # Body
        (11, 23), (12, 24), (23, 24),
        # Legs  
        (23, 25), (25, 27), (27, 29), (29, 31), (24, 26), (26, 28), (28, 30), (30, 32)
    ]
    
    edge_index = []
    # Add anatomical connections (bidirectional)
    for i, j in connections:
        if i < num_nodes and j < num_nodes:
            edge_index.extend([[i, j], [j, i]])
    
    # Add some long-range connections for better information flow
    important_joints = [0, 11, 12, 23, 24, 15, 16, 27, 28]  # head, shoulders, hips, wrists, ankles
    for i in important_joints:
        for j in important_joints:
            if i != j and i < num_nodes and j < num_nodes:
                edge_index.append([i, j])
    
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

edge_index = create_anatomical_edge_index().to(device)

# # ✅ STEP 4: Class names (same order as training)
# class_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

# ✅ STEP 5: MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ✅ STEP 6: Enhanced keypoint processing functions (same as training)
def normalize_keypoints(keypoints):
    """Enhanced normalize function with better stability (same as training)"""
    keypoints = np.array(keypoints)
    x, y, visibility = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
    
    # Center around hip midpoint for better pose normalization
    left_hip, right_hip = 23, 24
    if len(keypoints) > max(left_hip, right_hip):
        center_x = (keypoints[left_hip, 0] + keypoints[right_hip, 0]) / 2
        center_y = (keypoints[left_hip, 1] + keypoints[right_hip, 1]) / 2
        x = x - center_x
        y = y - center_y
    
    # Robust normalization
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    scale = max(x_range, y_range) + 1e-6
    
    x = x / scale
    y = y / scale
    
    # Filter low-visibility keypoints
    visibility = np.where(visibility < 0.5, 0.0, visibility)
    
    return np.stack([x, y, visibility], axis=1)

def is_full_body_visible(landmarks):
    """Check if full body is visible in the frame"""
    # Key points to check for full body visibility
    # Head (nose), shoulders, hips, knees, ankles
    key_points = [0, 11, 12, 23, 24, 25, 26, 27, 28]  # MediaPipe landmark indices
    
    visible_count = 0
    for idx in key_points:
        if landmarks.landmark[idx].visibility > 0.5:  # Visibility threshold
            visible_count += 1
    
    # Require at least 80% of key points to be visible
    return visible_count >= len(key_points) * 0.8

def extract_keypoints_from_frame(frame):
    """Extract keypoints from a frame using enhanced normalization"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(frame_rgb)
    
    if not results.pose_landmarks:
        return None, results, False
    
    # Check if full body is visible
    if not is_full_body_visible(results.pose_landmarks):
        return None, results, False
    
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.append([lm.x, lm.y, lm.visibility])
    
    normalized_kpts = normalize_keypoints(keypoints)
    return normalized_kpts, results, True


def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    # Convert to numpy arrays
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    
    # Calculate vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate angle
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def get_pose_corrections(landmarks, predicted_pose):
    """Get personalized pose corrections with specific measurements - ENHANCED WITH MORE POSES"""
    corrections = []
    
    # Extract key landmarks (x, y, z coordinates)
    lm = landmarks.landmark
    
    if predicted_pose == "Standing":
        # For standing pose, just return empty corrections (no specific corrections needed)
        return []
    
    elif predicted_pose == "tree":
        # Tree pose corrections with specific measurements
        # 1. Check if standing leg is straight
        left_knee_angle = calculate_angle(
            [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]  # hip-knee-ankle
        )
        right_knee_angle = calculate_angle(
            [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
        )
        
        # Determine standing leg (the straighter one)
        if left_knee_angle > right_knee_angle:
            standing_angle = left_knee_angle
            lifted_angle = right_knee_angle
            standing_side = "left"
            lifted_side = "right"
        else:
            standing_angle = right_knee_angle
            lifted_angle = left_knee_angle
            standing_side = "right"
            lifted_side = "left"
        
        # Standing leg should be straight (>165 degrees)
        target_standing = 175
        if standing_angle < 165:
            needed_degrees = target_standing - standing_angle
            corrections.append(f"Straighten {standing_side} leg by {needed_degrees:.0f} degrees more")
        
        # Lifted leg should be bent (<90 degrees)
        target_lifted = 80
        if lifted_angle > 90:
            excess_degrees = lifted_angle - target_lifted
            corrections.append(f"Bend {lifted_side} knee {excess_degrees:.0f} degrees more")
        
        # Check arm elevation angle
        left_shoulder_elbow_y = lm[11].y - lm[13].y  # negative means elbow is above shoulder
        right_shoulder_elbow_y = lm[12].y - lm[14].y
        
        if left_shoulder_elbow_y > -0.05:  # arms not raised enough
            corrections.append("Raise left arm higher above head")
        if right_shoulder_elbow_y > -0.05:
            corrections.append("Raise right arm higher above head")
        
        # Check balance - hip alignment
        hip_tilt = abs(lm[23].y - lm[24].y) * 100  # Convert to percentage
        if hip_tilt > 3:
            if lm[23].y > lm[24].y:
                corrections.append(f"Level hips - lift left hip higher")
            else:
                corrections.append(f"Level hips - lift right hip higher")
    
    elif predicted_pose == "Warrior I":
        # Warrior 1 pose corrections
        left_knee_angle = calculate_angle(
            [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
        )
        right_knee_angle = calculate_angle(
            [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
        )
        
        # Determine front leg (more bent) and back leg
        if left_knee_angle < right_knee_angle:
            front_angle = left_knee_angle
            back_angle = right_knee_angle
            front_side = "left"
            back_side = "right"
        else:
            front_angle = right_knee_angle
            back_angle = left_knee_angle
            front_side = "right"
            back_side = "left"
        
        # Front leg should be around 90 degrees
        if front_angle > 110:
            excess_degrees = front_angle - 90
            corrections.append(f"Bend {front_side} knee {excess_degrees:.0f} degrees deeper")
        elif front_angle < 70:
            needed_degrees = 90 - front_angle
            corrections.append(f"Raise {front_side} knee {needed_degrees:.0f} degrees higher")
        
        # Back leg should be straight
        if back_angle < 165:
            needed_degrees = 175 - back_angle
            corrections.append(f"Straighten {back_side} leg more")
        
        # Arms should be raised above head
        left_arm_above = lm[11].y - lm[13].y  # Should be negative (elbow above shoulder)
        right_arm_above = lm[12].y - lm[14].y
        
        if left_arm_above > -0.1:
            corrections.append("Raise left arm higher above head")
        if right_arm_above > -0.1:
            corrections.append("Raise right arm higher above head")
        
        # Check torso alignment (should face forward)
        shoulder_alignment = abs(lm[11].x - lm[12].x) * 100
        if shoulder_alignment < 15:  # Shoulders too narrow (twisted)
            corrections.append("Square shoulders to face forward")
    
    elif predicted_pose == "warrior2":
        # Warrior 2 pose corrections with specific measurements
        left_knee_angle = calculate_angle(
            [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
        )
        right_knee_angle = calculate_angle(
            [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
        )
        
        # Check which leg is more bent (front leg)
        if left_knee_angle < right_knee_angle:
            front_angle = left_knee_angle
            back_angle = right_knee_angle
            front_side = "left"
            back_side = "right"
        else:
            front_angle = right_knee_angle
            back_angle = left_knee_angle
            front_side = "right"
            back_side = "left"
        
        # Front leg should be around 90 degrees
        target_front = 90
        if front_angle > 110:
            excess_degrees = front_angle - target_front
            corrections.append(f"Bend {front_side} knee {excess_degrees:.0f} degrees deeper")
        elif front_angle < 70:
            needed_degrees = target_front - front_angle
            corrections.append(f"Raise {front_side} knee {needed_degrees:.0f} degrees higher")
        
        # Back leg should be straight (>165 degrees)
        if back_angle < 165:
            needed_degrees = 175 - back_angle
            corrections.append(f"Straighten {back_side} leg more")
        
        # Check arm alignment - should be parallel to ground
        left_arm_slope = (lm[13].y - lm[11].y) * 100  # shoulder to elbow slope
        right_arm_slope = (lm[14].y - lm[12].y) * 100
        
        if abs(left_arm_slope) > 5:
            if left_arm_slope > 0:
                corrections.append(f"Raise left arm - keep parallel")
            else:
                corrections.append(f"Lower left arm - keep parallel")
        
        if abs(right_arm_slope) > 5:
            if right_arm_slope > 0:
                corrections.append(f"Raise right arm - keep parallel")
            else:
                corrections.append(f"Lower right arm - keep parallel")
    
    elif predicted_pose == "Triangle":
        # Triangle pose corrections
        left_knee_angle = calculate_angle(
            [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
        )
        right_knee_angle = calculate_angle(
            [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
        )
        
        # Both legs should be straight
        if left_knee_angle < 165:
            needed_degrees = 175 - left_knee_angle
            corrections.append(f"Straighten left leg {needed_degrees:.0f} degrees more")
        if right_knee_angle < 165:
            needed_degrees = 175 - right_knee_angle
            corrections.append(f"Straighten right leg {needed_degrees:.0f} degrees more")
        
        # Check if one arm is reaching down and other up
        left_hand_low = lm[15].y > lm[11].y  # Hand below shoulder
        right_hand_low = lm[16].y > lm[12].y
        
        if not (left_hand_low ^ right_hand_low):  # Both high or both low
            corrections.append("One hand should reach down, other up")
        
        # Check side bend - torso should be tilted
        torso_tilt = abs((lm[11].y + lm[12].y) / 2 - (lm[23].y + lm[24].y) / 2) * 100
        if torso_tilt < 10:
            corrections.append("Tilt torso more to the side")
        
        # Check stance width
        foot_width = abs(lm[27].x - lm[28].x) * 100
        if foot_width < 30:
            corrections.append("Widen stance - feet wider apart")
    
    elif predicted_pose == "Chair":
        # Chair pose corrections
        left_knee_angle = calculate_angle(
            [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
        )
        right_knee_angle = calculate_angle(
            [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
        )
        
        # Both knees should be bent (around 90-120 degrees)
        target_knee = 105
        if left_knee_angle > 140:
            needed_degrees = left_knee_angle - target_knee
            corrections.append(f"Bend left knee {needed_degrees:.0f} degrees deeper")
        if right_knee_angle > 140:
            needed_degrees = right_knee_angle - target_knee
            corrections.append(f"Bend right knee {needed_degrees:.0f} degrees deeper")
        
        # Arms should be raised above head
        left_arm_raised = lm[11].y - lm[15].y > 0.2  # Wrist well above shoulder
        right_arm_raised = lm[12].y - lm[16].y > 0.2
        
        if not left_arm_raised:
            corrections.append("Raise left arm higher above head")
        if not right_arm_raised:
            corrections.append("Raise right arm higher above head")
        
        # Check hip hinge - hips should be pushed back
        hip_knee_alignment = abs((lm[23].x + lm[24].x) / 2 - (lm[25].x + lm[26].x) / 2) * 100
        if hip_knee_alignment < 5:
            corrections.append("Push hips back more")
        
        # Check knee alignment - knees shouldn't cave in
        knee_width = abs(lm[25].x - lm[26].x) * 100
        ankle_width = abs(lm[27].x - lm[28].x) * 100
        if knee_width < ankle_width * 0.8:
            corrections.append("Push knees out - don't let them cave in")
    
    elif predicted_pose == "Cobra":
        # Cobra pose corrections
        # Check if person is lying down (hips should be low)
        hip_height = (lm[23].y + lm[24].y) / 2
        shoulder_height = (lm[11].y + lm[12].y) / 2
        
        if hip_height < shoulder_height:  # Hips above shoulders (wrong orientation)
            corrections.append("Lie face down with hips on ground")
        
        # Arms should be supporting upper body
        left_arm_angle = calculate_angle(
            [lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y]
        )
        right_arm_angle = calculate_angle(
            [lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]
        )
        
        # Arms should be moderately bent (not fully straight)
        if left_arm_angle > 150:
            corrections.append("Bend left elbow slightly - don't lock arm")
        if right_arm_angle > 150:
            corrections.append("Bend right elbow slightly - don't lock arm")
        
        # Check back arch - shoulders should be above elbows
        left_shoulder_elbow = lm[11].y - lm[13].y
        right_shoulder_elbow = lm[12].y - lm[13].y
        
        if left_shoulder_elbow > 0:  # Shoulder below elbow
            corrections.append("Lift chest higher")
        if right_shoulder_elbow > 0:
            corrections.append("Lift chest higher")
        
        # Legs should be straight and together
        leg_separation = abs(lm[27].x - lm[28].x) * 100  # Ankle separation
        if leg_separation > 15:
            corrections.append("Keep legs together")
    
    elif predicted_pose == "Bridge":
        # Bridge pose corrections
        # Check if hips are lifted (bridge position)
        hip_height = (lm[23].y + lm[24].y) / 2
        knee_height = (lm[25].y + lm[26].y) / 2
        
        if hip_height > knee_height:  # Hips below knees
            corrections.append("Lift hips higher - create bridge shape")
        
        # Knees should be bent around 90 degrees
        left_knee_angle = calculate_angle(
            [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
        )
        right_knee_angle = calculate_angle(
            [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
        )
        
        target_knee = 90
        if abs(left_knee_angle - target_knee) > 20:
            if left_knee_angle > target_knee:
                corrections.append("Bring left foot closer to hip")
            else:
                corrections.append("Move left foot away from hip")
        
        if abs(right_knee_angle - target_knee) > 20:
            if right_knee_angle > target_knee:
                corrections.append("Bring right foot closer to hip")
            else:
                corrections.append("Move right foot away from hip")
        
        # Knees should be parallel (not caving in or out)
        knee_alignment = abs(lm[25].x - lm[26].x) * 100
        ankle_alignment = abs(lm[27].x - lm[28].x) * 100
        
        if abs(knee_alignment - ankle_alignment) > 10:
            corrections.append("Keep knees parallel to ankles")
        
        # Arms should be on ground for support
        arm_support_height = (lm[15].y + lm[16].y) / 2  # Average wrist height
        if arm_support_height < hip_height:  # Arms above hips (not supporting)
            corrections.append("Press arms into ground for support")
    
    elif predicted_pose == "Boat":
        # Boat pose corrections
        # Check V-shape - torso and legs should form a V
        hip_height = (lm[23].y + lm[24].y) / 2
        shoulder_height = (lm[11].y + lm[12].y) / 2
        knee_height = (lm[25].y + lm[26].y) / 2
        
        # Hips should be the lowest point
        if hip_height < shoulder_height or hip_height < knee_height:
            corrections.append("Sit on tailbone - hips should be lowest point")
        
        # Legs should be lifted
        ankle_height = (lm[27].y + lm[28].y) / 2
        if ankle_height > knee_height:  # Ankles below knees
            corrections.append("Lift legs higher")
        
        # Arms should be reaching forward
        left_arm_forward = lm[15].x - lm[11].x  # Wrist relative to shoulder
        right_arm_forward = lm[16].x - lm[12].x
        
        if abs(left_arm_forward) < 0.1:  # Arm not extended forward
            corrections.append("Reach left arm forward")
        if abs(right_arm_forward) < 0.1:
            corrections.append("Reach right arm forward")
        
        # Check balance - body should be stable
        torso_lean = abs((lm[11].y + lm[12].y) / 2 - (lm[23].y + lm[24].y) / 2) * 100
        if torso_lean > 20:
            corrections.append("Balance on sitting bones")
        
        # Legs can be bent or straight - check if consistent
        left_knee_angle = calculate_angle(
            [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
        )
        right_knee_angle = calculate_angle(
            [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
        )
        
        if abs(left_knee_angle - right_knee_angle) > 30:
            corrections.append("Keep both legs at same angle")
    
    elif predicted_pose == "Cat":
        # Cat pose corrections
        # Check if on hands and knees
        hand_support = (lm[15].y + lm[16].y) / 2  # Average wrist height
        knee_support = (lm[25].y + lm[26].y) / 2  # Average knee height
        
        if abs(hand_support - knee_support) > 0.3:  # Not on same level
            corrections.append("Get on hands and knees")
        
        # Check spine arch - back should be rounded (cat stretch)
        shoulder_height = (lm[11].y + lm[12].y) / 2
        hip_height = (lm[23].y + lm[24].y) / 2
        
        # In cat pose, spine should be arched up
        if shoulder_height > hip_height:  # Shoulders below hips (wrong arch)
            corrections.append("Round spine upward - tuck chin to chest")
        
        # Arms should be straight
        left_arm_angle = calculate_angle(
            [lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y]
        )
        right_arm_angle = calculate_angle(
            [lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]
        )
        
        if left_arm_angle < 160:
            corrections.append("Straighten left arm")
        if right_arm_angle < 160:
            corrections.append("Straighten right arm")
        
        # Hands should be under shoulders
        left_hand_shoulder_align = abs(lm[15].x - lm[11].x) * 100
        right_hand_shoulder_align = abs(lm[16].x - lm[12].x) * 100
        
        if left_hand_shoulder_align > 15:
            corrections.append("Place left hand under left shoulder")
        if right_hand_shoulder_align > 15:
            corrections.append("Place right hand under right shoulder")
    
    elif predicted_pose == "Headstand":
        # Headstand corrections
        # Check if person is inverted (head should be lowest point)
        head_height = lm[0].y  # Nose landmark
        hip_height = (lm[23].y + lm[24].y) / 2
        
        if head_height < hip_height:  # Head above hips (not inverted)
            corrections.append("Invert body - head should be at bottom")
        
        # Check arm support - arms should be supporting body weight
        left_elbow_height = lm[13].y
        right_elbow_height = lm[14].y
        shoulder_height = (lm[11].y + lm[12].y) / 2
        
        # Elbows should be supporting (near ground level with head)
        if abs(left_elbow_height - head_height) > 0.1:
            corrections.append("Place left elbow on ground for support")
        if abs(right_elbow_height - head_height) > 0.1:
            corrections.append("Place right elbow on ground for support")
        
        # Check leg alignment - legs should be straight up
        left_leg_angle = calculate_angle(
            [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
        )
        right_leg_angle = calculate_angle(
            [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
        )
        
        # Legs should be straight (>165 degrees)
        if left_leg_angle < 165:
            corrections.append("Straighten left leg")
        if right_leg_angle < 165:
            corrections.append("Straighten right leg")
        
        # Check body alignment - should be in straight line
        head_to_hip_x = abs(lm[0].x - (lm[23].x + lm[24].x) / 2) * 100
        if head_to_hip_x > 10:
            corrections.append("Align body in straight line")
        
        # Check balance - feet should be together
        foot_separation = abs(lm[27].x - lm[28].x) * 100
        if foot_separation > 10:
            corrections.append("Keep feet together")
        
        # Check core engagement - torso should be stable
        shoulder_hip_alignment = abs(shoulder_height - hip_height) * 100
        expected_alignment = abs(head_height - hip_height) * 80  # Should be proportional
        if abs(shoulder_hip_alignment - expected_alignment) > 15:
            corrections.append("Engage core for stability")
    
    elif predicted_pose == "plank":
        # Plank pose corrections (if not already implemented)
        # Check if body is in straight line
        head_height = lm[0].y
        shoulder_height = (lm[11].y + lm[12].y) / 2
        hip_height = (lm[23].y + lm[24].y) / 2
        ankle_height = (lm[27].y + lm[28].y) / 2
        
        # All points should be roughly in line
        body_line_variance = np.var([head_height, shoulder_height, hip_height, ankle_height]) * 1000
        if body_line_variance > 50:
            if hip_height > shoulder_height:
                corrections.append("Lower hips - create straight line")
            elif hip_height < shoulder_height:
                corrections.append("Lift hips slightly")
        
        # Arms should be straight
        left_arm_angle = calculate_angle(
            [lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y]
        )
        right_arm_angle = calculate_angle(
            [lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]
        )
        
        if left_arm_angle < 160:
            corrections.append("Straighten left arm")
        if right_arm_angle < 160:
            corrections.append("Straighten right arm")
        
        # Hands should be under shoulders
        left_hand_shoulder_align = abs(lm[15].x - lm[11].x) * 100
        right_hand_shoulder_align = abs(lm[16].x - lm[12].x) * 100
        
        if left_hand_shoulder_align > 15:
            corrections.append("Place left hand under left shoulder")
        if right_hand_shoulder_align > 15:
            corrections.append("Place right hand under right shoulder")
        
        # Legs should be straight
        left_leg_angle = calculate_angle(
            [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
        )
        right_leg_angle = calculate_angle(
            [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
        )
        
        if left_leg_angle < 165:
            corrections.append("Straighten left leg")
        if right_leg_angle < 165:
            corrections.append("Straighten right leg")
    
    elif predicted_pose == "downdog":
        # Downward Dog corrections (if not already implemented)
        # Check if in inverted V shape
        hand_height = (lm[15].y + lm[16].y) / 2
        hip_height = (lm[23].y + lm[24].y) / 2
        foot_height = (lm[27].y + lm[28].y) / 2
        
        # Hips should be highest point
        if hip_height > hand_height or hip_height > foot_height:
            corrections.append("Lift hips higher - create inverted V")
        
        # Arms should be straight
        left_arm_angle = calculate_angle(
            [lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y]
        )
        right_arm_angle = calculate_angle(
            [lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]
        )
        
        if left_arm_angle < 160:
            corrections.append("Straighten left arm")
        if right_arm_angle < 160:
            corrections.append("Straighten right arm")
        
        # Legs should be straight
        left_leg_angle = calculate_angle(
            [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
        )
        right_leg_angle = calculate_angle(
            [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
        )
        
        if left_leg_angle < 160:
            corrections.append("Straighten left leg")
        if right_leg_angle < 160:
            corrections.append("Straighten right leg")
        
        # Check hand-foot distance
        hand_foot_distance = abs(((lm[15].x + lm[16].x) / 2) - ((lm[27].x + lm[28].x) / 2)) * 100
        if hand_foot_distance < 30:
            corrections.append("Step feet back for better stretch")
        elif hand_foot_distance > 60:
            corrections.append("Step feet closer to hands")
        
        # Hands should be shoulder-width apart
        hand_width = abs(lm[15].x - lm[16].x) * 100
        shoulder_width = abs(lm[11].x - lm[12].x) * 100
        if abs(hand_width - shoulder_width) > 10:
            corrections.append("Adjust hand width to match shoulders")
    
    elif predicted_pose == "goddess":
        # Goddess pose corrections (if not already implemented)
        # Both legs should be bent in squat position
        left_knee_angle = calculate_angle(
            [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
        )
        right_knee_angle = calculate_angle(
            [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
        )
        
        # Both knees should be bent (90-120 degrees)
        target_knee = 105
        if left_knee_angle > 130:
            needed_degrees = left_knee_angle - target_knee
            corrections.append(f"Bend left knee {needed_degrees:.0f} degrees deeper")
        if right_knee_angle > 130:
            needed_degrees = right_knee_angle - target_knee
            corrections.append(f"Bend right knee {needed_degrees:.0f} degrees deeper")
        
        # Knees should be wide apart (turned out)
        knee_width = abs(lm[25].x - lm[26].x) * 100
        if knee_width < 25:
            corrections.append("Turn knees out wider")
        
        # Arms should be in cactus shape or raised
        left_elbow_height = lm[13].y
        right_elbow_height = lm[14].y
        shoulder_height = (lm[11].y + lm[12].y) / 2
        
        # Elbows should be at or above shoulder level
        if left_elbow_height > shoulder_height:
            corrections.append("Lift left elbow to shoulder height")
        if right_elbow_height > shoulder_height:
            corrections.append("Lift right elbow to shoulder height")
        
        # Check stance width
        foot_width = abs(lm[27].x - lm[28].x) * 100
        if foot_width < 35:
            corrections.append("Widen stance - feet further apart")
        
        # Check hip level
        hip_level = abs(lm[23].y - lm[24].y) * 100
        if hip_level > 5:
            corrections.append("Keep hips level")
        
        # Torso should be upright
        torso_lean = abs((lm[11].y + lm[12].y) / 2 - (lm[23].y + lm[24].y) / 2) * 100
        if torso_lean > 15:
            corrections.append("Keep torso upright")
    
    # Return empty list for poses that don't need specific corrections or aren't implemented
    return corrections


def predict_pose(keypoints):
    """Predict yoga pose from keypoints using advanced model"""
    if keypoints is None:
        return None, None
    
    # Create graph data
    data = Data(
        x=torch.tensor(keypoints, dtype=torch.float).to(device),
        edge_index=edge_index
    )
    
    # Predict
    with torch.no_grad():
        batch = torch.zeros(data.x.size(0), dtype=torch.long).to(device)
        out = model(data.x, data.edge_index, batch)
        probabilities = F.softmax(out, dim=1)
        pred_idx = out.argmax().item()
        confidence = probabilities[0][pred_idx].item()
    
    return class_names[pred_idx], confidence

# ✅ STEP 7: Enhanced real-time detection function
def run_realtime_detection():
    """Run real-time yoga pose detection with advanced model"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("✅ Starting real-time yoga pose detection with advanced model...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Variables for FPS calculation
    fps_counter = 0
    start_time = time.time()
    
    # Variables for confidence smoothing
    confidence_history = []
    pose_history = []
    history_size = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Extract keypoints and get pose landmarks
        keypoints, results, full_body_visible = extract_keypoints_from_frame(frame)
        
        # Draw pose landmarks on frame only if full body is visible
        if results.pose_landmarks and full_body_visible:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Predict pose
            pose_name, confidence = predict_pose(keypoints)
            
            if pose_name and confidence > 0.65:  # Threshold for advanced model
                # Smooth predictions with history
                confidence_history.append(confidence)
                pose_history.append(pose_name)
                
                if len(confidence_history) > history_size:
                    confidence_history.pop(0)
                    pose_history.pop(0)
                
                # Get most frequent pose and average confidence
                if len(pose_history) >= 3:
                    most_common_pose = max(set(pose_history), key=pose_history.count)
                    avg_confidence = np.mean(confidence_history)
                    
                    # Get pose corrections
                    corrections = get_pose_corrections(results.pose_landmarks, most_common_pose)
                    
                    # Create text with outline for better visibility
                    text = f"{most_common_pose.upper()}"
                    conf_text = f"{avg_confidence:.1%}"
                    
                    # Display pose name and confidence
                    text_x, text_y = 20, 40
                    conf_x, conf_y = 20, 70
                    
                    # Draw text with black outline for visibility
                    # Pose name
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)  # Black outline
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)  # Green text
                    
                    # Confidence
                    cv2.putText(frame, conf_text, (conf_x, conf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Black outline
                    cv2.putText(frame, conf_text, (conf_x, conf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text
                    
                    # Display corrections if any
                    if corrections:
                        correction_y = 100
                        cv2.putText(frame, "Corrections:", (20, correction_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                        cv2.putText(frame, "Corrections:", (20, correction_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)  # Orange
                        
                        for i, correction in enumerate(corrections[:3]):  # Show max 3 corrections
                            corr_y = correction_y + 25 + (i * 25)
                            corr_text = f"- {correction}"
                            cv2.putText(frame, corr_text, (25, corr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                            cv2.putText(frame, corr_text, (25, corr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # Yellow
                    else:
                        # Perfect pose
                        perfect_text = "Perfect pose!"
                        cv2.putText(frame, perfect_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                        cv2.putText(frame, perfect_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Green
                        
                    # Model info
                    model_text = "Advanced GCN+Residual+LSTM+MLP Model"
                    cv2.putText(frame, model_text, (20, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                    cv2.putText(frame, model_text, (20, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                else:
                    # Warming up
                    warmup_text = "Warming up..."
                    cv2.putText(frame, warmup_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                    cv2.putText(frame, warmup_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                # Low confidence - show "No pose detected"
                no_pose_text = "No pose detected"
                cv2.putText(frame, no_pose_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)  # Black outline
                cv2.putText(frame, no_pose_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red text
                
                # Clear history on low confidence
                confidence_history.clear()
                pose_history.clear()
        
        elif results.pose_landmarks and not full_body_visible:
            # Partial body detected - show warning
            warning_text = "Show full body for detection"
            cv2.putText(frame, warning_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Black outline
            cv2.putText(frame, warning_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)  # Orange text
            
            # Clear history on partial body
            confidence_history.clear()
            pose_history.clear()
        
        # Calculate and display FPS
        fps_counter += 1
        if fps_counter % 30 == 0:  # Update FPS every 30 frames
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = end_time
        
        # Display FPS (top-right corner with outline)
        fps_text = f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --"
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        fps_x = frame.shape[1] - fps_size[0] - 10
        cv2.putText(frame, fps_text, (fps_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # Black outline
        cv2.putText(frame, fps_text, (fps_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)  # White text
        
        # Display instructions (bottom with outline)
        instruction_text = "Press 'q' to quit, 's' to save"
        cv2.putText(frame, instruction_text, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black outline
        cv2.putText(frame, instruction_text, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
        
        # Show frame
        cv2.imshow('Real-time Yoga Pose Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = int(time.time())
            filename = f"yoga_pose_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Real-time detection stopped")

# ✅ STEP 8: Main execution
if __name__ == "__main__":
    try:
        run_realtime_detection()
    except KeyboardInterrupt:
        print("\n⚠️ Detection interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

