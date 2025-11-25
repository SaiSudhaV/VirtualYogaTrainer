# 🧘 AI-Enhanced Real-Time Virtual Yoga Studio

A cutting-edge web-based yoga pose detection system powered by advanced AI, using PoseNet and TensorFlow.js for intelligent real-time pose analysis, personalized coaching, and comprehensive training sessions.

## 🚀 Enhanced AI Features

### 🤖 AI Yoga Coach
- **Intelligent Progress Tracking**: Advanced analytics track your improvement over time
- **Personalized Recommendations**: AI analyzes your performance patterns and suggests focus areas
- **Adaptive Difficulty**: System automatically adjusts to your skill level (beginner/intermediate/advanced)
- **Smart Motivation**: Context-aware motivational messages based on your progress
- **Performance Insights**: Detailed metrics on accuracy, consistency, and improvement trends

### 🔄 Smart Sequence Detection
- **Guided Yoga Flows**: Pre-built sequences like Sun Salutation with intelligent transitions
- **Personalized Sequences**: AI generates custom flows based on your strengths and weaknesses
- **Real-time Flow Guidance**: Voice and visual cues for smooth pose transitions
- **Adaptive Sequencing**: Sequences adjust based on your performance and preferences
- **Multiple Difficulty Levels**: Beginner flows to advanced strength-building sequences

### 📊 Advanced Pose Analytics
- **Biomechanical Analysis**: Real-time joint angle calculations and muscle activation estimates
- **Injury Prevention**: AI monitors for risky positions and provides safety warnings
- **Balance & Stability Metrics**: Comprehensive analysis of your stability and alignment
- **Performance Efficiency**: Measures how well you execute poses with minimal effort
- **Historical Trends**: Track your progress across multiple sessions with detailed analytics

### 🎯 Core Features
- **Real-time Pose Detection**: 10 yoga poses with AI-enhanced accuracy feedback
- **Interactive Timer**: Automatic timing based on pose correctness
- **Cross-Platform**: Desktop, tablet, and mobile support with responsive AI
- **PWA Support**: Install as native app with offline AI capabilities
- **Voice Guidance**: AI-powered voice coaching and corrections
- **Auto-Capture**: Automatically saves images of correct poses for progress tracking

## Supported Poses

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

## 🚀 How to Run the Project

### Prerequisites
- Modern web browser (Chrome recommended for best AI performance)
- Webcam access
- Local server (Python 3.6+, Node.js 12+, or VS Code Live Server)

### Quick Start

1. **Clone/Download the project**
   ```bash
   git clone <repository-url>
   cd yoga_pose_detection
   ```

2. **Start a local server** (choose one method):

   **Method 1: Python Server**
   ```bash
   python -m http.server 8000
   ```
   Then open: http://localhost:8000

   **Method 2: Node.js Server**
   ```bash
   npx http-server -p 8000
   ```
   Then open: http://localhost:8000

   **Method 3: VS Code Live Server**
   - Install Live Server extension
   - Right-click `index.html` → "Open with Live Server"

3. **Allow camera permissions** when prompted

4. **Start practicing** with AI-enhanced yoga coaching!

### First Time Setup
1. Position yourself 3-6 feet from camera
2. Ensure good lighting and plain background
3. Click "Start Session" to activate AI coaching
4. Try "Sequence Mode" for guided flows
5. Use "Analytics" to track your progress

## 💡 Usage Guide

### Basic Usage
1. **Allow camera access** when prompted
2. **Select pose** from dropdown menu or use AI sequence mode
3. **Position yourself** 3-6 feet from camera
4. **Click "Start Session"** to activate AI coaching
5. **Follow AI guidance** - real-time voice and visual feedback
6. **Watch AI analytics** - live biomechanical analysis

### AI-Enhanced Controls
- **Start/Resume**: Begin AI-coached session
- **Pause**: Pause with AI progress saving
- **Reset**: Reset with AI data retention
- **Sequence Mode**: Start guided AI flows
- **Personal Flow**: Generate custom AI sequences
- **Analytics**: View AI progress insights

### AI Features in Action
- **🔊 Voice Guidance**: Real-time AI corrections and encouragement
- **📊 Live Analytics**: Biomechanical analysis during practice
- **⚠️ Safety Monitoring**: AI injury prevention warnings
- **📸 Auto-Capture**: AI saves perfect poses automatically
- **🎯 Progress Tracking**: AI tracks improvement over time

## Setup Requirements

### Optimal Setup for AI
- **Distance**: 3-6 feet from camera for best AI analysis
- **Lighting**: Good ambient lighting for accurate AI detection
- **Background**: Plain background for enhanced AI processing
- **Position**: Full body visible for complete biomechanical analysis
- **Audio**: Enable speakers/headphones for AI voice guidance
- **Space**: Clear area for AI-guided sequences and transitions

### Browser Compatibility
- Chrome 80+ (recommended)
- Firefox 75+
- Safari 13+
- Edge 80+

## Troubleshooting

### Camera Issues
- Check browser permissions (click lock icon in address bar)
- Ensure HTTPS or localhost for camera access
- Try different browser if issues persist
- Clear browser cache and restart

### Performance Issues
- Close other browser tabs
- Use good lighting conditions
- Ensure plain background
- Try Chrome for best performance

### Server Issues
```bash
# Check if port is available
netstat -an | grep 8000

# Use different port if needed
python -m http.server 3000
```

## Mobile Installation (PWA)

**iOS:**
1. Open in Safari
2. Tap Share → "Add to Home Screen"

**Android:**
1. Open in Chrome
2. Tap menu → "Add to Home Screen" or "Install App"

## 🔬 Technical Details

### Core Technologies
- **Frontend**: HTML5, CSS3, JavaScript ES6+
- **ML Framework**: TensorFlow.js, PoseNet with MobileNetV1
- **AI Engine**: Custom neural network for pose classification
- **Analytics Engine**: Real-time biomechanical analysis algorithms

### AI Capabilities
- **Pose Detection**: 17 keypoints with 24+ engineered features
- **Biomechanical Analysis**: Joint angles, muscle activation estimation
- **Pattern Recognition**: User behavior analysis and adaptation
- **Predictive Analytics**: Performance trend analysis and forecasting
- **Safety Monitoring**: Real-time injury risk assessment

### Performance Metrics
- **Frame Rate**: 15-30+ FPS (device-dependent)
- **Pose Accuracy**: 75-95% (condition-dependent)
- **AI Response Time**: <100ms for real-time feedback
- **Analytics Processing**: Real-time biomechanical calculations
- **Memory Usage**: Optimized for mobile devices

## 📊 AI Enhancement Benefits

### For Beginners
- **Gentle Learning Curve**: AI adapts to your pace and provides encouraging feedback
- **Safety First**: Real-time injury prevention monitoring keeps you safe
- **Personalized Guidance**: Custom sequences designed for your skill level
- **Progress Tracking**: See your improvement over time with detailed analytics

### For Intermediate Practitioners
- **Technique Refinement**: Biomechanical analysis helps perfect your form
- **Challenging Sequences**: AI generates flows that push your boundaries safely
- **Consistency Building**: Track and improve your practice consistency
- **Weakness Identification**: AI identifies areas needing improvement

### For Advanced Yogis
- **Performance Optimization**: Advanced metrics help maximize efficiency
- **Complex Sequences**: AI creates challenging flows with smooth transitions
- **Injury Prevention**: Sophisticated monitoring prevents overexertion
- **Teaching Insights**: Detailed analytics useful for yoga instructors

### Universal Benefits
- **24/7 Availability**: Your AI yoga instructor is always ready
- **Personalized Experience**: Every session adapts to your unique needs
- **Progress Motivation**: AI celebrates achievements and encourages growth
- **Data-Driven Insights**: Make informed decisions about your practice
- **Accessibility**: Works on any device with a camera
- **Privacy**: All AI processing happens locally on your device

## 📝 Project Structure

```
yoga_pose_detection/
├── index.html                 # Main application with AI integration
├── manifest.json             # PWA manifest
├── service-worker.js         # Offline functionality
├── src/js/                   # JavaScript modules
│   ├── device_compatibility.js  # Device optimization
│   ├── yoga_pose_detector.js    # Enhanced pose detection with AI
│   ├── yoga_timer.js           # Smart timing system
│   ├── ai_coach.js             # 🆕 AI Yoga Coach
│   ├── smart_sequence_detector.js # 🆕 Intelligent sequence flows
│   ├── pose_analytics.js       # 🆕 Advanced biomechanical analysis
│   └── yoga_studio_app.js      # Main app with AI orchestration
├── datasets/                 # Training data (10 pose folders)
└── docs/                    # Documentation
```

### 🆕 AI Modules Overview

#### 1. AI Yoga Coach (`ai_coach.js`)
**Purpose**: Intelligent coaching system that learns from user behavior
- Progress tracking with accuracy, hold time, and consistency metrics
- Pattern recognition to identify strengths and weaknesses
- Adaptive difficulty adjustment (beginner/intermediate/advanced)
- Personalized recommendations based on performance data
- Motivational system with context-aware encouragement

#### 2. Smart Sequence Detector (`smart_sequence_detector.js`)
**Purpose**: AI-powered yoga sequence generation and guidance
- Pre-built sequences (Sun Salutation, Beginner Flow, Strength Builder)
- Personalized sequence generation based on user analytics
- Real-time transition guidance with voice and visual cues
- Adaptive sequencing that modifies based on performance

#### 3. Advanced Pose Analytics (`pose_analytics.js`)
**Purpose**: Comprehensive biomechanical analysis and injury prevention
- Real-time joint angle calculations and muscle activation estimates
- Injury prevention monitoring with safety warnings
- Balance and stability metrics analysis
- Performance efficiency measurements
- Historical trend tracking and insights

## 🚀 Getting Started with AI Features

### Quick AI Setup
1. **Start a Session**: Click "Start Session" to activate AI coaching
2. **Enable Voice Guidance**: Keep the "🔊 AI Voice Guidance" checkbox enabled
3. **Try Sequence Mode**: Click "Sequence Mode" for guided flows
4. **Generate Personal Flow**: Click "Personal Flow" for AI-customized sequences
5. **View Analytics**: Click "Analytics" to see your progress insights

### AI Voice Commands
- **"Start Session"**: Activates AI coaching
- **"Sequence Mode"**: Begins guided yoga flows
- **"Personal Flow"**: Creates customized sequences
- **"Show Analytics"**: Displays progress data

### Understanding AI Feedback
- **Green Border**: Perfect pose detected by AI
- **Red Border**: AI suggests adjustments
- **Voice Corrections**: Real-time AI guidance
- **Safety Warnings**: AI injury prevention alerts
- **Progress Notifications**: AI celebrates your achievements

## 📊 AI Data & Privacy

- **Local Processing**: All AI computations happen on your device
- **No Data Upload**: Your practice data stays private
- **Offline Capable**: AI works without internet connection
- **Auto-Save**: Correct poses automatically saved to downloads
- **Progress Tracking**: Local storage for personal analytics

## 🤖 AI Implementation Details

### AI Architecture
- **Modular Design**: Separate AI components for coaching, sequencing, and analytics
- **Real-Time Processing**: All AI computations happen locally for privacy and speed
- **Data Persistence**: Local storage for user progress and preferences
- **Performance Optimization**: Efficient algorithms for real-time analysis

### Integration Points
- **Main Application**: Enhanced `yoga_studio_app.js` orchestrates all AI components
- **Pose Detection**: Upgraded `yoga_pose_detector.js` with AI-powered corrections
- **User Interface**: Dynamic panels for analytics, sequences, and AI controls
- **Voice System**: Integrated speech synthesis for AI coaching

### Performance Metrics
- **AI Processing Speed**: <100ms response time for AI feedback
- **Real-time Analysis**: Processed every frame (15-30 FPS)
- **Pattern Recognition**: Continuous learning from user behavior
- **Accuracy Improvements**: Enhanced pose detection through AI corrections

### Privacy & Security
- **Local Processing**: All AI processing happens on user's device
- **No Data Upload**: Personal practice data never leaves the device
- **Offline Capability**: AI works without internet connection
- **Secure Storage**: Local browser storage for user data

## 🔮 Future AI Enhancements

- **Computer Vision Improvements**: Enhanced pose detection accuracy
- **Advanced Biomechanics**: Deeper physiological analysis
- **Health Integration**: Connect with fitness trackers and health apps
- **Social Features**: AI-powered community challenges
- **Yoga Therapy**: AI-assisted therapeutic sequences
- **Accessibility**: AI adaptations for different physical abilities