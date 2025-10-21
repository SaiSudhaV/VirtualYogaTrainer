import cv2
import numpy as np
import mediapipe as mp
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

class YogaDataPreprocessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.scaler = StandardScaler()
        
        # Pose mapping for dataset
        self.pose_mapping = {
            '1_pranamasana': 0,
            '2_hastauttanasana': 1, 
            '3_hastapadasana': 2,
            '4_ashwa_sanchalanasana': 3,
            '5_dandasana': 4,
            '6_ashtanga_namaskara': 5,
            '7_bhujangasana': 6,
            '8_adho_mukha_svanasana': 7,
            'padmasana': 8,
            'tadasana': 9
        }
        
        self.pose_names = [
            'Pranamasana', 'Hastauttanasana', 'Hastapadasana', 
            'Ashwa Sanchalanasana', 'Dandasana', 'Ashtanga Namaskara',
            'Bhujangasana', 'Adho Mukha Svanasana', 'Padmasana', 'Tadasana'
        ]
    
    def extract_features_from_image(self, image_path):
        """Extract pose features from image"""
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
            
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        return self._extract_pose_features(landmarks)
    
    def _extract_pose_features(self, landmarks):
        """Extract 30 comprehensive features from landmarks"""
        if len(landmarks) < 33:
            return None
            
        # Key body points
        key_points = {
            'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        points = {}
        for name, idx in key_points.items():
            if landmarks[idx]['visibility'] > 0.5:
                points[name] = [landmarks[idx]['x'], landmarks[idx]['y'], landmarks[idx]['z']]
            else:
                points[name] = [0, 0, 0]
        
        # 1. Joint angles (12 features)
        angles = [
            self._calculate_angle_3d(points['left_shoulder'], points['left_elbow'], points['left_wrist']),
            self._calculate_angle_3d(points['right_shoulder'], points['right_elbow'], points['right_wrist']),
            self._calculate_angle_3d(points['left_hip'], points['left_knee'], points['left_ankle']),
            self._calculate_angle_3d(points['right_hip'], points['right_knee'], points['right_ankle']),
            self._calculate_angle_3d(points['left_shoulder'], points['left_hip'], points['left_knee']),
            self._calculate_angle_3d(points['right_shoulder'], points['right_hip'], points['right_knee']),
            self._calculate_angle_3d(points['left_elbow'], points['left_shoulder'], points['right_shoulder']),
            self._calculate_angle_3d(points['right_elbow'], points['right_shoulder'], points['left_shoulder']),
            self._calculate_angle_3d(points['left_knee'], points['left_hip'], points['right_hip']),
            self._calculate_angle_3d(points['right_knee'], points['right_hip'], points['left_hip']),
            self._calculate_angle_3d(points['nose'], points['left_shoulder'], points['left_hip']),
            self._calculate_angle_3d(points['nose'], points['right_shoulder'], points['right_hip'])
        ]
        
        # 2. Distances (8 features)
        distances = [
            self._calculate_distance_3d(points['left_wrist'], points['right_wrist']),
            self._calculate_distance_3d(points['left_ankle'], points['right_ankle']),
            self._calculate_distance_3d(points['left_shoulder'], points['right_shoulder']),
            self._calculate_distance_3d(points['left_hip'], points['right_hip']),
            self._calculate_distance_3d(points['nose'], points['left_hip']),
            self._calculate_distance_3d(points['nose'], points['right_hip']),
            self._calculate_distance_3d(points['left_wrist'], points['left_hip']),
            self._calculate_distance_3d(points['right_wrist'], points['right_hip'])
        ]
        
        # 3. Body ratios (6 features)
        ratios = [
            abs(points['left_shoulder'][1] - points['right_shoulder'][1]),
            abs(points['left_hip'][1] - points['right_hip'][1]),
            abs(points['left_ankle'][1] - points['right_ankle'][1]),
            self._safe_divide(distances[2], distances[3]),
            self._safe_divide(distances[0], distances[1]),
            abs(points['nose'][1] - (points['left_hip'][1] + points['right_hip'][1])/2)
        ]
        
        # 4. Pose orientation (4 features)
        orientation = [
            points['nose'][0] - (points['left_shoulder'][0] + points['right_shoulder'][0])/2,
            (points['left_shoulder'][1] + points['right_shoulder'][1])/2 - (points['left_hip'][1] + points['right_hip'][1])/2,
            abs(points['left_wrist'][0] - points['right_wrist'][0]),
            abs(points['left_ankle'][0] - points['right_ankle'][0])
        ]
        
        all_features = angles + distances + ratios + orientation
        all_features = [0 if np.isnan(f) or np.isinf(f) else f for f in all_features]
        
        return np.array(all_features)
    
    def _calculate_angle_3d(self, a, b, c):
        """Calculate angle between three 3D points"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def _calculate_distance_3d(self, a, b):
        """Calculate 3D Euclidean distance"""
        return np.sqrt(sum((a[i] - b[i])**2 for i in range(3)))
    
    def _safe_divide(self, a, b):
        """Safe division to avoid division by zero"""
        return a / b if b != 0 else 0
    
    def load_dataset_from_images(self, dataset_path='datasets'):
        """Load dataset from image files"""
        X, y = [], []
        
        print("Loading dataset from images...")
        
        for pose_dir in os.listdir(dataset_path):
            pose_path = os.path.join(dataset_path, pose_dir)
            
            if not os.path.isdir(pose_path) or pose_dir.startswith('.'):
                continue
                
            if pose_dir not in self.pose_mapping:
                continue
            
            print(f"Processing {pose_dir}...")
            
            images = [f for f in os.listdir(pose_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            valid_count = 0
            for img_file in images:
                img_path = os.path.join(pose_path, img_file)
                features = self.extract_features_from_image(img_path)
                
                if features is not None:
                    X.append(features)
                    y.append(self.pose_mapping[pose_dir])
                    valid_count += 1
            
            print(f"  Loaded {valid_count}/{len(images)} images")
        
        return np.array(X), np.array(y)
    
    def preprocess_data(self, X, y, test_size=0.2, validation_size=0.1):
        """Preprocess data with train/validation/test split"""
        # Remove invalid samples
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_clean, y_clean, test_size=test_size, random_state=42, stratify=y_clean
        )
        
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def save_preprocessor(self, filename='preprocessor.pkl'):
        """Save preprocessor components"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'pose_mapping': self.pose_mapping,
                'pose_names': self.pose_names
            }, f)