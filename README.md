# 🧘 Yoga Pose Detection - Dataset Preparation & Model Training

A machine learning project focused on **dataset preparation, preprocessing, and model training & tuning** for yoga pose detection using MediaPipe and scikit-learn.

## 📋 Project Overview

This project implements:
- **Dataset preparation** from yoga pose images
- **Feature extraction** using MediaPipe pose landmarks
- **Data preprocessing** with normalization and train/validation/test splits
- **Model training & tuning** with multiple ML algorithms and hyperparameter optimization

### 🎯 Supported Yoga Poses (10 Classes)

1. **Pranamasana** (Prayer Pose)
2. **Hastauttanasana** (Raised Arms Pose)
3. **Hastapadasana** (Standing Forward Bend)
4. **Ashwa Sanchalanasana** (Low Lunge)
5. **Dandasana** (Staff Pose)
6. **Ashtanga Namaskara** (Eight-Limbed Pose)
7. **Bhujangasana** (Cobra Pose)
8. **Adho Mukha Svanasana** (Downward Dog)
9. **Padmasana** (Lotus Pose)
10. **Tadasana** (Mountain Pose)

## 🔬 Technical Architecture

### MediaPipe PoseNet Library

**MediaPipe** is Google's open-source framework for pose estimation:

- **33 Body Landmarks**: Detects key points including joints and extremities
- **3D Coordinates**: Provides x, y, z coordinates with visibility confidence
- **Real-time Performance**: Efficient pose detection from images
- **Robust Detection**: Works across different body types and lighting conditions

#### How PoseNet Works:
1. **Input Processing**: Takes RGB images
2. **Feature Extraction**: Uses deep neural networks to identify body parts
3. **Landmark Detection**: Outputs 33 key body points with confidence scores
4. **Pose Estimation**: Provides accurate human pose estimation

### 📊 Feature Engineering (30 Features)

Our system extracts **30 comprehensive features** from pose landmarks:

#### 1. Joint Angles (12 features)
- **Arm Angles**: Shoulder-Elbow-Wrist angles for both arms
- **Leg Angles**: Hip-Knee-Ankle angles for both legs  
- **Torso Angles**: Body alignment and posture measurements
- **Cross-body Angles**: Diagonal body measurements

#### 2. Distance Measurements (8 features)
- **Hand-to-Hand Distance**: Arm span measurements
- **Foot-to-Foot Distance**: Leg positioning
- **Body Proportions**: Shoulder width, hip width
- **Limb Distances**: Hand-to-hip, body center distances

#### 3. Body Ratios (6 features)
- **Symmetry Measurements**: Left-right body balance
- **Alignment Ratios**: Shoulder/hip alignment
- **Pose Orientation**: Body tilt and positioning

#### 4. Pose Characteristics (4 features)
- **Head Position**: Relative to body center
- **Limb Spread**: Arm and leg extension measurements

## 🛠️ Main Libraries and Technologies

### Core Libraries
- **OpenCV (4.12.0)**: Image processing and computer vision
- **MediaPipe (0.10.21)**: Pose detection and landmark extraction
- **NumPy (1.26.4)**: Numerical computations and array operations
- **Scikit-learn (1.6.1)**: Machine learning algorithms and preprocessing

### Machine Learning Models & Tuning
- **Random Forest**: Ensemble learning with hyperparameter tuning
- **Gradient Boosting**: Advanced boosting with learning rate optimization
- **Support Vector Machine**: Kernel and regularization parameter tuning
- **Neural Networks**: Multi-layer perceptron with architecture optimization
- **Logistic Regression**: Regularization parameter tuning

## 🚀 How to Run the Project

### 1. Environment Setup
```bash
# Install required packages
pip install -r updated_requirements.txt
```

### 2. Dataset Structure
Ensure your dataset follows this structure:
```
datasets/
├── 1_pranamasana/          # Prayer Pose images
├── 2_hastauttanasana/      # Raised Arms Pose images
├── 3_hastapadasana/        # Standing Forward Bend images
├── 4_ashwa_sanchalanasana/ # Low Lunge images
├── 5_dandasana/            # Staff Pose images
├── 6_ashtanga_namaskara/   # Eight-Limbed Pose images
├── 7_bhujangasana/         # Cobra Pose images
├── 8_adho_mukha_svanasana/ # Downward Dog images
├── padmasana/              # Lotus Pose images
└── tadasana/               # Mountain Pose images
```

### 3. Run Training Pipeline
```bash
# Complete dataset preparation, preprocessing, and model training
python train_model.py
```

**Expected Output**:
- Dataset loading and preprocessing statistics
- Model training progress for 5 different algorithms
- Hyperparameter tuning results
- Best model evaluation with accuracy metrics
- Saved model files

## 📊 Dataset Information

### Current Dataset Statistics
- **Total Images**: 1,015 yoga pose images
- **Valid Samples**: 972 (after pose detection filtering)
- **Image Formats**: JPG, JPEG, PNG
- **Feature Extraction**: 30 features per image

### Class Distribution
| Pose | Images | Percentage |
|------|--------|------------|
| Cobra Pose | 213 | 21.9% |
| Downward Dog | 175 | 18.0% |
| Mountain Pose | 149 | 15.3% |
| Others | 435 | 44.8% |

## 🎯 Advantages of the Project

### 1. **Comprehensive Dataset Preparation**
- Automated feature extraction from images
- MediaPipe integration for robust pose detection
- Handles missing landmarks and invalid poses

### 2. **Advanced Feature Engineering**
- 30 engineered features capture pose nuances
- 3D spatial analysis with angle and distance calculations
- Normalized features for optimal model performance

### 3. **Extensive Model Training & Tuning**
- 5 different ML algorithms comparison
- Grid search hyperparameter optimization
- Cross-validation for reliable performance estimation
- Automatic best model selection

### 4. **Robust Data Preprocessing**
- Train/Validation/Test split (70%/10%/20%)
- Feature standardization and normalization
- Invalid sample filtering and handling

### 5. **Modular Architecture**
- Clean separation of preprocessing and training
- Reusable components for different datasets
- Easy to extend with new algorithms

## 🎨 Pose Validation through Landmark Plotting

### Visual Validation System

The system uses **33 key body landmarks** for pose validation:

#### 1. **Landmark Categories**
- **Face**: Nose, eyes, ears (5 points)
- **Torso**: Shoulders, hips (4 points)
- **Arms**: Elbows, wrists (4 points)  
- **Legs**: Knees, ankles, heels, toes (20 points)

#### 2. **Validation Process**
1. **Landmark Detection**: MediaPipe identifies 33 body points
2. **Confidence Filtering**: Only landmarks with >50% confidence are used
3. **Feature Extraction**: 30 comprehensive features calculated
4. **Data Validation**: Invalid samples filtered out
5. **Normalization**: Features scaled for optimal training

## 📈 Expected Performance

### Model Training Results
- **Training Time**: 2-10 minutes (depending on dataset size)
- **Expected Accuracy**: 75-90% (varies by pose complexity)
- **Cross-validation**: 5-fold stratified validation
- **Hyperparameter Tuning**: Grid search optimization

### System Requirements
- **Processing**: Modern CPU (no GPU required)
- **Memory**: 4GB RAM minimum
- **Storage**: 100MB for models and dependencies

## 📁 Project Structure

```
yoga_pose_detection/
├── README.md                    # Project documentation
├── updated_requirements.txt     # Python dependencies
├── train_model.py              # Main training pipeline
├── data_preprocessor.py        # Dataset preparation & preprocessing
├── model_trainer.py            # Model training & tuning
├── datasets/                   # Training images (10 pose folders)
└── Generated Files:
    ├── best_yoga_model.pkl     # Best trained model
    └── preprocessor.pkl        # Data preprocessor
```

## 🔧 Troubleshooting

### Common Issues
1. **"No pose landmarks detected"**: Ensure images show clear human poses
2. **Low accuracy**: Check dataset balance and image quality
3. **Memory errors**: Reduce dataset size or use smaller models
4. **Import errors**: Verify all dependencies are installed correctly

### Performance Optimization
- Ensure good image quality for better landmark detection
- Balance dataset classes for optimal training
- Use sufficient training data (50+ images per pose recommended)

## 🎓 Educational Value

This project demonstrates:
- **Computer Vision**: Pose estimation and landmark detection
- **Feature Engineering**: Domain-specific feature extraction from spatial data
- **Machine Learning**: Multiple algorithm comparison and hyperparameter tuning
- **Data Science**: Complete ML pipeline from raw data to trained model
- **Software Engineering**: Modular, maintainable code structure

---

**Built for educational purposes focusing on dataset preparation and model training**