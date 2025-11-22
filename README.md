# 🧘 Real-Time Virtual Yoga Studio

A web-based yoga pose detection system using PoseNet and TensorFlow.js for real-time pose analysis and interactive training sessions.

## Features

- **Real-time Pose Detection**: 10 yoga poses with accuracy feedback
- **Interactive Timer**: Automatic timing based on pose correctness
- **Cross-Platform**: Desktop, tablet, and mobile support
- **PWA Support**: Install as native app
- **Offline Capability**: Works without internet after initial load

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

## Quick Start

### Prerequisites
- Modern web browser (Chrome recommended)
- Webcam access
- Python 3.6+ or Node.js 12+

### Installation

**Python Server:**
```bash
cd yoga_pose_detection
python -m http.server 8000
# Open: http://localhost:8000
```

**Node.js Server:**
```bash
cd yoga_pose_detection
npx http-server -p 8000
# Open: http://localhost:8000
```

**VS Code Live Server:**
- Install Live Server extension
- Right-click `index.html` → "Open with Live Server"

## Usage

1. **Allow camera access** when prompted
2. **Select pose** from dropdown menu
3. **Position yourself** 3-6 feet from camera
4. **Click "Start Session"** to begin
5. **Hold pose** - timer runs when pose is correct (green border)
6. **Adjust position** if border turns red

### Controls
- **Start/Resume**: Begin or continue session
- **Pause**: Pause timer
- **Reset**: Reset timer and start over

## Setup Requirements

### Camera Setup
- Distance: 3-6 feet from camera
- Lighting: Good ambient lighting, avoid backlighting
- Background: Plain, uncluttered background
- Position: Full body visible in frame

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

## Technical Details

- **Frontend**: HTML5, CSS3, JavaScript
- **ML Framework**: TensorFlow.js, PoseNet
- **Pose Detection**: 17 keypoints, 24 engineered features
- **Performance**: 15-30+ FPS depending on device
- **Accuracy**: 75-95% depending on conditions

## Project Structure

```
yoga_pose_detection/
├── index.html                 # Main application
├── manifest.json             # PWA manifest
├── service-worker.js         # Offline functionality
├── src/js/                   # JavaScript modules
│   ├── device_compatibility.js
│   ├── yoga_pose_detector.js
│   ├── yoga_timer.js
│   └── yoga_studio_app.js
├── datasets/                 # Training data (10 pose folders)
└── docs/                    # Documentation
```

## License

MIT License - Open source project for the yoga community.