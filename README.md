# 🧘 AI-Enhanced Real-Time Virtual Yoga Studio

A cutting-edge web-based yoga pose detection system powered by advanced AI, using PoseNet and TensorFlow.js for intelligent real-time pose analysis, personalized coaching, and comprehensive training sessions with **12 Sun Salutation poses**.

## 🚀 Enhanced Features

### 🎯 12-Pose Sun Salutation Sequence
Complete traditional Sun Salutation (Surya Namaskara) with:
1. **Pranamasana** (Prayer Pose)
2. **Hasta Uttanasana** (Raised Arms Pose) 
3. **Pada Hastasana** (Standing Forward Bend)
4. **Ashwa Sanchalanasana** (Low Lunge)
5. **Parvatasana** (Mountain Pose/Downward Dog)
6. **Ashtanga Namaskara** (Eight-Limbed Pose)
7. **Bhujangasana** (Cobra Pose)
8. **Parvatasana** (Mountain Pose/Downward Dog)
9. **Ashwa Sanchalanasana** (Low Lunge)
10. **Pada Hastasana** (Standing Forward Bend)
11. **Hasta Uttanasana** (Raised Arms Pose)
12. **Pranamasana** (Prayer Pose)

### 🤖 AI Yoga Coach
- **Real-time Pose Correction**: Intelligent voice guidance for proper alignment
- **Accuracy Tracking**: Live percentage feedback on pose correctness
- **Personalized Instructions**: AI analyzes your form and provides specific corrections
- **Progress Monitoring**: Track improvement across multiple sessions
- **Voice Guidance**: Optional audio coaching with pose instructions

### 📚 Professional Split-Screen Interface
- **Left Panel**: Full-screen live video with zoom controls (50%-300%)
- **Right Panel**: Comprehensive information display
- **Learn Mode**: 30-second timer per pose for beginners
- **Practice Mode**: 60-second timer per pose for advanced practitioners
- **Interactive Pose Gallery**: Visual reference with SVG illustrations

### 📊 Advanced Analytics
- **Biomechanical Analysis**: Real-time joint angle calculations
- **Pose Accuracy Metrics**: Detailed scoring system (75%+ threshold for correct poses)
- **Training Data Collection**: Automatic capture of correct poses for dataset building
- **Performance Insights**: Session completion tracking and progress analytics

### 🎯 Core Features
- **Real-time Pose Detection**: 12 yoga poses with AI-enhanced accuracy feedback
- **Complete Body Coverage**: Zoom controls for full posture visibility
- **Split-Screen Layout**: Live video on left, comprehensive info on right
- **Pose Benefits Display**: Health advantages and therapeutic benefits
- **Real-time Corrections**: Specific alignment guidance and feedback
- **Webcam Integration**: Secure camera access with permission handling
- **Cross-Platform**: Desktop, tablet, and mobile support with responsive design
- **PWA Support**: Install as native app with offline capabilities
- **Auto-Capture**: Automatically saves correct poses for training purposes
- **Manual Capture**: Button to manually capture training images

## 🛠️ Technical Stack

### Frontend Technologies
- **HTML5/CSS3**: Modern responsive design with gradient backgrounds
- **JavaScript ES6+**: Modular architecture with class-based components
- **p5.js**: Canvas rendering and video capture
- **ml5.js v0.12.2**: PoseNet integration for pose detection

### AI/ML Components
- **TensorFlow.js**: Client-side machine learning
- **PoseNet MobileNetV1**: Lightweight pose estimation model
- **Real-time Analysis**: 15-30 FPS pose detection and correction
- **Voice Synthesis**: Browser-based text-to-speech for guidance

### Performance Optimizations
- **Device-Adaptive Settings**: Automatic optimization for mobile/desktop
- **Efficient Rendering**: Optimized canvas operations
- **Memory Management**: Proper cleanup and resource management
- **Error Handling**: Graceful fallbacks for camera/AI failures

## 🚀 Quick Start Guide

### Prerequisites
- Modern web browser (Chrome 80+ recommended for best performance)
- Webcam access
- Local server (Python 3.6+, Node.js 12+, or VS Code Live Server)
- Stable internet connection for initial AI model loading

### Installation & Setup

1. **Clone/Download the project**
   ```bash
   git clone <repository-url>
   cd yoga_pose_detection
   ```

2. **Start a local server** (choose one method):

   **Method 1: Python Server**
   ```bash
   python3 -m http.server 8000
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
1. **Camera Setup**: Position yourself 3-6 feet from camera
2. **Lighting**: Ensure good ambient lighting and plain background
3. **Adjust View**: Use zoom controls (50%-300%) to ensure complete body visibility
4. **Choose Mode**: Select "Learn" (30s) or "Practice" (60s) mode
5. **Select Pose**: Choose from 12 Sun Salutation poses
6. **Start Session**: Click "Start Practice" to begin AI coaching with split-screen interface

## 💡 Usage Guide

### Learn Tab
- **Pose Gallery**: Visual grid of all 12 poses with SVG illustrations
- **Pose Selection**: Click any pose card to switch to Practice mode
- **Reference Images**: High-quality SVG graphics for proper form guidance
- **Pose Information**: Benefits and descriptions for each pose

### Practice Tab
- **Split-Screen Layout**: Professional dual-panel interface
- **Left Panel**: Full-screen live video with zoom controls
  - Zoom range: 50% to 300% for complete body visibility
  - Clean pose overlay without masking original video
  - Reset zoom functionality for optimal positioning
- **Right Panel**: Comprehensive information display
  - Current pose reference image with SVG illustration
  - Real-time timer with color-coded countdown
  - Live pose accuracy percentage with progress bar
  - **Pose Benefits Section**: Health advantages and therapeutic benefits
  - **Real-time Corrections**: Specific alignment guidance (e.g., "Move your left arm closer")
  - Practice mode selector and session controls
- **Voice Guidance**: Optional audio coaching and corrections
- **Progress Tracking**: Session completion and pose accuracy metrics

### Controls & Features
- **🎯 Start Practice**: Begin AI-coached session with selected pose
- **⏸️ Pause/Resume**: Pause session while maintaining progress
- **🔄 Reset**: Reset session and return to starting state
- **🔍 Zoom Controls**: Adjust video size (50%-300%) for complete body coverage
- **🔊 Voice Toggle**: Enable/disable AI voice guidance
- **📸 Manual Capture**: Capture current pose for training dataset
- **📊 Progress Bar**: Visual session completion tracking
- **🌟 Benefits Display**: Real-time pose benefits and health advantages
- **🎯 Corrections Panel**: Live alignment feedback and specific guidance

## 🎯 AI Features in Action

### Real-time Corrections
- **Pose Analysis**: "Move your left arm closer to your body"
- **Alignment Feedback**: "Keep your shoulders level and aligned"
- **Accuracy Scoring**: Live percentage display (75%+ for correct poses)
- **Visual Indicators**: Green overlay for correct, red for incorrect poses

### Training Data Collection
- **Auto-Capture**: Automatically saves images when pose is held correctly for 1 second
- **Manual Capture**: Button to manually capture training images
- **Organized Storage**: Images saved with pose name and timestamp
- **Dataset Building**: Contributes to improving pose detection accuracy

### Voice Coaching
- **Pose Instructions**: "Now practicing Bhujangasana. Lie on stomach, lift chest with arms support."
- **Real-time Corrections**: Spoken feedback for pose adjustments
- **Session Guidance**: Welcome messages and completion celebrations
- **Customizable**: Toggle voice guidance on/off as needed

## 📱 Device Compatibility

### Optimal Setup
- **Distance**: 3-6 feet from camera for best AI analysis
- **Lighting**: Good ambient lighting for accurate detection
- **Background**: Plain background for enhanced AI processing
- **Position**: Full body visible - use zoom controls to adjust view
- **Zoom**: Adjust from 50%-300% to ensure complete body coverage
- **Audio**: Enable speakers/headphones for voice guidance

### Browser Support
- **Chrome 80+** (recommended for best performance)
- **Firefox 75+** (good compatibility)
- **Safari 13+** (iOS/macOS support)
- **Edge 80+** (Windows support)

### Mobile Installation (PWA)
**iOS:**
1. Open in Safari
2. Tap Share → "Add to Home Screen"

**Android:**
1. Open in Chrome
2. Tap menu → "Add to Home Screen" or "Install App"

## 🔧 Troubleshooting

### Camera Issues
- **Permission Denied**: Click camera icon in browser address bar → Allow
- **No Video**: Check if camera is being used by another application
- **Poor Quality**: Ensure good lighting and stable internet connection
- **Browser Issues**: Try Chrome for best compatibility

### Performance Issues
- **Slow Detection**: Close other browser tabs and applications
- **Low Accuracy**: Ensure proper lighting and plain background
- **Lag**: Check internet connection for AI model loading
- **Memory Issues**: Refresh page if performance degrades

### AI Model Loading
```bash
# Check if port is available
netstat -an | grep 8000

# Use different port if needed
python3 -m http.server 3000
```

### Common Solutions
1. **Refresh Page**: Solves most initialization issues
2. **Clear Cache**: Browser cache may interfere with updates
3. **Check Console**: F12 → Console for detailed error messages
4. **Camera Permissions**: Ensure browser has camera access

## 📊 Technical Architecture

### File Structure
```
yoga_pose_detection/
├── index.html                 # Main application with dual-tab UI
├── manifest.json             # PWA manifest for app installation
├── service-worker.js         # Offline functionality
├── src/js/                   # JavaScript modules
│   ├── device_compatibility.js  # Device optimization
│   ├── yoga_pose_detector.js    # Enhanced pose detection with 12 poses
│   ├── yoga_timer.js           # Smart timing system (30s/60s modes)
│   ├── ai_coach.js             # AI Yoga Coach
│   ├── smart_sequence_detector.js # Intelligent sequence flows
│   ├── pose_analytics.js       # Advanced biomechanical analysis
│   └── yoga_studio_app.js      # Main app orchestration
├── datasets/                 # Training data and references
│   ├── pose_images/          # 12 SVG pose illustrations (1.svg - 12.svg)
│   └── [pose_folders]/       # Individual pose training images
└── docs/                    # Documentation and presentations
```

### Core Components

#### 1. Yoga Pose Detector (`yoga_pose_detector.js`)
- **12-Pose Support**: Complete Sun Salutation sequence
- **Real-time Analysis**: Joint angle calculations and pose scoring
- **Voice Feedback**: Intelligent correction system
- **Training Capture**: Automatic and manual image capture
- **Error Handling**: Graceful camera permission and initialization

#### 2. Main Application (`yoga_studio_app.js`)
- **Split-Screen Interface**: Professional dual-panel layout
- **Zoom Controls**: Video scaling from 50% to 300%
- **Benefits Display**: Real-time pose health advantages
- **Corrections Engine**: Live alignment feedback system
- **Timer Management**: 30s (Learn) / 60s (Practice) modes
- **Session Tracking**: Progress monitoring and completion
- **Pose Selection**: Grid-based pose selection and sequential practice

#### 3. AI Enhancement Modules
- **AI Coach**: Progress tracking and personalized feedback
- **Sequence Detector**: Guided yoga flows and transitions
- **Pose Analytics**: Biomechanical analysis and injury prevention

## 🎯 Performance Metrics

### AI Capabilities
- **Detection Speed**: 15-30+ FPS (device-dependent)
- **Pose Accuracy**: 75-95% (condition-dependent)
- **Response Time**: <100ms for real-time feedback
- **Model Size**: Optimized MobileNetV1 for web deployment
- **Memory Usage**: Efficient for mobile devices

### User Experience
- **Setup Time**: <30 seconds for first-time users
- **Learning Curve**: Intuitive split-screen interface for all skill levels
- **Body Coverage**: Complete posture visibility with zoom controls
- **Real-time Feedback**: Instant benefits and corrections display
- **Session Length**: Flexible 30s or 60s per pose
- **Completion Rate**: Visual progress tracking encourages full sessions

## 🔮 Future Enhancements

### Planned Features
- **Advanced Sequences**: Additional yoga flows beyond Sun Salutation
- **Enhanced Zoom**: Pan and tilt controls for video positioning
- **Progress Analytics**: Detailed improvement tracking over time
- **Social Features**: Community challenges and sharing
- **Offline Mode**: Full functionality without internet connection
- **Advanced AI**: Enhanced pose detection and correction algorithms
- **Pose Variations**: Multiple difficulty levels for each pose

### Technical Improvements
- **WebRTC Integration**: Better camera handling and streaming
- **Machine Learning**: Custom pose classification models
- **Cloud Storage**: Synchronized progress across devices
- **Accessibility**: Enhanced support for different physical abilities

## 📝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and test thoroughly
4. Commit changes: `git commit -m 'Add new feature'`
5. Push to branch: `git push origin feature/new-feature`
6. Submit pull request

### Code Standards
- **ES6+ JavaScript**: Modern syntax and features
- **Modular Architecture**: Separate concerns and reusable components
- **Error Handling**: Comprehensive try-catch and graceful fallbacks
- **Documentation**: Clear comments and function descriptions
- **Testing**: Manual testing across different devices and browsers

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TensorFlow.js Team**: For the amazing machine learning framework
- **ml5.js Community**: For making AI accessible to web developers
- **p5.js Foundation**: For the creative coding platform
- **Yoga Community**: For inspiration and pose guidance
- **Open Source Contributors**: For continuous improvements and feedback

---

**Built with ❤️ for the yoga community - Namaste! 🙏**