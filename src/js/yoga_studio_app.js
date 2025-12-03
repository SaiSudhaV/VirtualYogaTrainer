// Enhanced Yoga Studio Application with 12 Poses
let poseDetector;
let yogaTimer;
let canvas;
let isSessionActive = false;

// Practice modes and timers
let practiceMode = 'learn'; // 'learn' (30s) or 'practice' (60s)
let currentPoseIndex = 0;
let completedPoses = [];
let sessionStartTime = null;

// Video zoom controls
let zoomLevel = 1.0;
let videoOffsetX = 0;
let videoOffsetY = 0;

// 12 Pose Sun Salutation Sequence
const poseSequence = [
    'Pranamasana', 'Hasta Uttanasana', 'Pada Hastasana', 'Ashwa Sanchalanasana',
    'Parvatasana', 'Ashtanga Namaskara', 'Bhujangasana', 'Parvatasana',
    'Ashwa Sanchalanasana', 'Pada Hastasana', 'Hasta Uttanasana', 'Pranamasana'
];

const poseDescriptions = {
    'Pranamasana': 'Stand with palms together in prayer position at chest level.',
    'Hasta Uttanasana': 'Raise both arms overhead, palms facing each other.',
    'Pada Hastasana': 'Bend forward, hands reaching toward feet.',
    'Ashwa Sanchalanasana': 'Step back into low lunge, hands on ground.',
    'Parvatasana': 'Form inverted V-shape, hands and feet on ground.',
    'Ashtanga Namaskara': 'Lower knees, chest, and chin to ground.',
    'Bhujangasana': 'Lie on stomach, lift chest with arms support.',
};

const poseBenefits = {
    'Pranamasana': 'Calms the mind, improves focus and concentration. Helps center the body and prepare for practice.',
    'Hasta Uttanasana': 'Stretches the chest and abdomen. Improves digestion and energizes the body.',
    'Pada Hastasana': 'Stretches hamstrings and calves. Improves blood circulation to the brain.',
    'Ashwa Sanchalanasana': 'Strengthens leg muscles. Improves balance and hip flexibility.',
    'Parvatasana': 'Strengthens arms and shoulders. Stretches the entire back body and calves.',
    'Ashtanga Namaskara': 'Strengthens arms and chest. Develops upper body strength and stability.',
    'Bhujangasana': 'Strengthens the spine. Opens the chest and improves lung capacity.',
};

// Add manual capture function for training
function captureTrainingImage() {
    if (poseDetector && poseDetector.video) {
        poseDetector.captureCorrectPose();
    } else {
        alert('Please start the session first to capture training images.');
    }
}

// AI Enhancement Components
let aiCoach, smartSequenceDetector, poseAnalytics;
let currentSessionData = {
    startTime: null,
    poses: [],
    corrections: [],
    analytics: []
};

function setup() {
    // Wait for DOM to be ready
    setTimeout(() => {
        const container = document.getElementById('canvas-container');
        if (!container) {
            console.error('Canvas container not found');
            return;
        }
        
        const containerRect = container.getBoundingClientRect();
        let canvasWidth = Math.floor(containerRect.width) || 640;
        let canvasHeight = Math.floor(containerRect.height) || 480;
        
        // Ensure proper aspect ratio for video
        const aspectRatio = 4/3;
        if (canvasWidth / canvasHeight > aspectRatio) {
            canvasWidth = canvasHeight * aspectRatio;
        } else {
            canvasHeight = canvasWidth / aspectRatio;
        }
        
        // Create canvas to fill left half
        canvas = createCanvas(canvasWidth, canvasHeight);
        canvas.parent('canvas-container');
        
        // Initialize components
        poseDetector = new YogaPoseDetector();
        yogaTimer = new YogaTimer();
        
        // Initialize AI components with error handling
        try {
            aiCoach = new AIYogaCoach();
            smartSequenceDetector = new SmartSequenceDetector();
            poseAnalytics = new PoseAnalytics();
            console.log('AI components initialized successfully');
        } catch (error) {
            console.warn('AI components failed to initialize:', error);
            aiCoach = null;
            smartSequenceDetector = null;
            poseAnalytics = null;
        }
        
        // Initialize pose detection
        poseDetector.initialize();
        
        // Setup UI
        setupUI();
        
        console.log('Enhanced Yoga Studio initialized with split-screen layout');
    }, 100);
}

function draw() {
    if (poseDetector.video) {
        // Clear canvas
        clear();
        
        // Calculate zoomed video dimensions
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        const videoWidth = poseDetector.video.width * zoomLevel;
        const videoHeight = poseDetector.video.height * zoomLevel;
        
        // Center the zoomed video
        const x = (canvasWidth - videoWidth) / 2 + videoOffsetX;
        const y = (canvasHeight - videoHeight) / 2 + videoOffsetY;
        
        // Draw video with zoom and offset
        image(poseDetector.video, x, y, videoWidth, videoHeight);
        
        // Draw pose landmarks (adjust for zoom)
        push();
        translate(x, y);
        scale(zoomLevel);
        poseDetector.drawPose();
        pop();
        
        // Update session
        if (isSessionActive) {
            const isCorrect = poseDetector.isCurrentPoseCorrect();
            const accuracy = poseDetector.getPoseAccuracy();
            
            // Update timer based on mode
            const targetTime = practiceMode === 'learn' ? 30 : 60;
            yogaTimer.targetHoldTime = targetTime;
            yogaTimer.updateWithPoseStatus(isCorrect, accuracy);
            
            // Update UI
            updatePracticeUI(isCorrect, accuracy);
            
            // Update corrections display
            updateCorrectionsDisplay();
            
            // Check for pose completion
            if (yogaTimer.checkPoseCompletion()) {
                handlePoseCompletion(accuracy);
            }
            
            // AI Analytics Integration
            try {
                if (poseAnalytics && poseDetector.poses && poseDetector.poses.length > 0) {
                    const keypoints = poseDetector.poses[0]?.pose?.keypoints || [];
                    if (keypoints.length > 0) {
                        const biomechanics = poseAnalytics.analyzeBiomechanics(
                            keypoints,
                            poseDetector.currentTargetPose
                        );
                        updateAnalyticsDisplay(biomechanics);
                    }
                }
            } catch (error) {
                console.warn('AI processing error:', error);
            }
        }
    }
}

function setupUI() {
    // Populate poses grid
    populatePosesGrid();
    
    // Populate pose selector
    populatePoseSelector();
    
    // Setup event listeners
    document.getElementById('startBtn').onclick = startSession;
    document.getElementById('pauseBtn').onclick = pauseSession;
    document.getElementById('resetBtn').onclick = resetSession;
    
    document.getElementById('poseSelect').onchange = (e) => {
        selectPose(parseInt(e.target.value));
    };
    
    document.getElementById('practiceMode').onchange = (e) => {
        practiceMode = e.target.value;
        updateTimerDisplay();
    };
    
    // Voice guidance toggle
    document.getElementById('voiceToggle').onchange = (e) => {
        if (poseDetector) {
            poseDetector.voiceEnabled = e.target.checked;
            if (e.target.checked) {
                poseDetector.speak('Voice guidance enabled');
            }
        }
    };
    
    // Add manual capture button
    addManualCaptureButton();
    
    // Initialize zoom display
    document.getElementById('zoomLevel').textContent = '100%';
    
    // Initial setup
    selectPose(0);
    updateSessionProgress();
}

function populatePosesGrid() {
    const posesGrid = document.getElementById('posesGrid');
    if (!posesGrid) return;
    
    poseSequence.forEach((poseName, index) => {
        const poseCard = document.createElement('div');
        poseCard.className = 'pose-card';
        poseCard.onclick = () => {
            selectPoseFromGrid(index);
            showTab('practice');
        };
        
        // Use correct SVG path
        const svgPath = `datasets/pose_images/${index + 1}.svg`;
        
        poseCard.innerHTML = `
            <img class="pose-image" src="${svgPath}" alt="${poseName}" onerror="this.style.display='none'">
            <div class="pose-name">${poseName}</div>
            <div class="pose-description">${poseDescriptions[poseName] || 'Practice this yoga pose with proper alignment.'}</div>
            <div style="margin-top: 10px; font-size: 12px; opacity: 0.7;">Pose ${index + 1} of 12</div>
        `;
        
        posesGrid.appendChild(poseCard);
    });
    
    console.log('Populated poses grid with 12 poses');
}

function populatePoseSelector() {
    const poseSelect = document.getElementById('poseSelect');
    
    poseSequence.forEach((poseName, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `${index + 1}. ${poseName}`;
        poseSelect.appendChild(option);
    });
}

function selectPoseFromGrid(poseIndex) {
    selectPose(poseIndex);
    
    // Update grid selection
    const poseCards = document.querySelectorAll('.pose-card');
    poseCards.forEach((card, index) => {
        if (index === poseIndex) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });
}

function selectPose(poseIndex) {
    currentPoseIndex = poseIndex;
    
    // Update pose detector
    poseDetector.setTargetPose(poseIndex);
    
    // Update UI
    const poseName = poseSequence[poseIndex];
    const currentPoseNameEl = document.getElementById('currentPoseName');
    const currentPoseImageEl = document.getElementById('currentPoseImage');
    const benefitsTextEl = document.getElementById('benefitsText');
    
    if (currentPoseNameEl) currentPoseNameEl.textContent = poseName;
    if (currentPoseImageEl) {
        currentPoseImageEl.src = `datasets/pose_images/${poseIndex + 1}.svg`;
        currentPoseImageEl.onerror = () => {
            currentPoseImageEl.style.display = 'none';
            console.warn(`SVG not found: datasets/pose_images/${poseIndex + 1}.svg`);
        };
    }
    if (benefitsTextEl) {
        benefitsTextEl.textContent = poseBenefits[poseName] || 'Practice this pose with proper alignment.';
    }
    
    // Update selector
    const poseSelectEl = document.getElementById('poseSelect');
    if (poseSelectEl) poseSelectEl.value = poseIndex;
    
    // Reset corrections
    updateCorrectionsDisplay();
    
    // Reset timer if session is active
    if (isSessionActive) {
        yogaTimer.reset();
        yogaTimer.start();
    }
    
    console.log(`Selected pose: ${poseName}`);
}

function startSession() {
    if (!isSessionActive) {
        isSessionActive = true;
        sessionStartTime = Date.now();
        
        // Initialize session data
        currentSessionData = {
            startTime: sessionStartTime,
            poses: [],
            corrections: [],
            analytics: []
        };
        
        yogaTimer.start();
        
        // Update UI
        document.getElementById('startBtn').textContent = 'Resume';
        document.getElementById('pauseBtn').disabled = false;
        document.getElementById('statusDisplay').textContent = 'Practice started! Hold the pose correctly.';
        
        // Voice welcome
        if (poseDetector && poseDetector.voiceEnabled) {
            const mode = practiceMode === 'learn' ? '30 second' : '60 second';
            const message = `Starting ${mode} practice session. Hold each pose correctly to build strength and flexibility.`;
            poseDetector.speak(message);
        }
        
        console.log(`${practiceMode} session started`);
    } else {
        yogaTimer.start();
        document.getElementById('startBtn').textContent = 'Resume';
        document.getElementById('statusDisplay').textContent = 'Session resumed';
    }
}

function pauseSession() {
    if (isSessionActive) {
        yogaTimer.pause();
        
        if (yogaTimer.isPaused) {
            document.getElementById('startBtn').textContent = 'Resume';
            document.getElementById('statusDisplay').textContent = 'Session paused';
        } else {
            document.getElementById('startBtn').textContent = 'Pause';
            document.getElementById('statusDisplay').textContent = 'Session resumed';
        }
    }
}

function resetSession() {
    isSessionActive = false;
    completedPoses = [];
    yogaTimer.reset();
    
    // Reset UI
    document.getElementById('startBtn').textContent = 'Start Practice';
    document.getElementById('pauseBtn').disabled = true;
    document.getElementById('statusDisplay').textContent = 'Ready to start practice';
    document.getElementById('statusDisplay').className = 'status-display';
    document.getElementById('videoContainer').className = 'video-container';
    document.getElementById('accuracyValue').textContent = '0%';
    document.getElementById('accuracyBar').style.width = '0%';
    
    updateTimerDisplay();
    updateSessionProgress();
    
    console.log('Session reset');
}

function updatePracticeUI(isCorrect, accuracy) {
    // Update accuracy display
    document.getElementById('accuracyValue').textContent = `${Math.round(accuracy)}%`;
    document.getElementById('accuracyBar').style.width = `${accuracy}%`;
    
    // Update status and video container
    const statusDisplay = document.getElementById('statusDisplay');
    const videoContainer = document.getElementById('videoContainer');
    
    if (isCorrect && accuracy > 75) {
        statusDisplay.textContent = `✅ Perfect pose! ${Math.round(accuracy)}% accuracy`;
        statusDisplay.className = 'status-display status-correct';
        videoContainer.className = 'video-container correct';
    } else {
        statusDisplay.textContent = `❌ Adjust your pose - ${Math.round(accuracy)}% accuracy`;
        statusDisplay.className = 'status-display status-incorrect';
        videoContainer.className = 'video-container incorrect';
    }
    
    // Update timer display
    updateTimerDisplay();
}

function updateTimerDisplay() {
    const timerDisplay = document.getElementById('timerDisplay');
    if (isSessionActive && yogaTimer) {
        const elapsed = Math.floor(yogaTimer.getElapsedTime());
        const target = practiceMode === 'learn' ? 30 : 60;
        const remaining = Math.max(0, target - elapsed);
        
        const minutes = Math.floor(remaining / 60);
        const seconds = remaining % 60;
        timerDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        
        // Change color based on time remaining
        if (remaining <= 10) {
            timerDisplay.style.color = '#f44336';
        } else if (remaining <= 30) {
            timerDisplay.style.color = '#FF9800';
        } else {
            timerDisplay.style.color = '#4CAF50';
        }
    } else {
        const target = practiceMode === 'learn' ? 30 : 60;
        const minutes = Math.floor(target / 60);
        const seconds = target % 60;
        timerDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        timerDisplay.style.color = '#4CAF50';
    }
}

function handlePoseCompletion(accuracy) {
    // Mark pose as completed
    if (!completedPoses.includes(currentPoseIndex)) {
        completedPoses.push(currentPoseIndex);
        
        // Store session data
        currentSessionData.poses.push({
            pose: currentPoseIndex,
            poseName: poseSequence[currentPoseIndex],
            accuracy: accuracy,
            holdTime: yogaTimer.getElapsedTime(),
            timestamp: Date.now()
        });
        
        // AI Coach feedback
        if (aiCoach) {
            const feedback = aiCoach.trackProgress(
                currentPoseIndex,
                accuracy,
                yogaTimer.getElapsedTime(),
                []
            );
            showAIFeedback(feedback);
        }
        
        // Auto-capture correct pose
        if (accuracy > 85) {
            captureCorrectPose();
        }
    }
    
    // Show completion feedback
    showPoseCompletionFeedback();
    updateSessionProgress();
    
    // Auto-advance to next pose in sequence (optional)
    // advanceToNextPose();
}

function showPoseCompletionFeedback() {
    const statusDisplay = document.getElementById('statusDisplay');
    const originalText = statusDisplay.textContent;
    const originalClass = statusDisplay.className;
    
    statusDisplay.textContent = '🎉 Pose completed! Excellent work!';
    statusDisplay.className = 'status-display status-correct';
    
    // Voice feedback
    if (poseDetector.voiceEnabled) {
        const poseName = poseSequence[currentPoseIndex];
        speakMessage(`Excellent! You've completed ${poseName}. Great form and alignment.`);
    }
    
    // Reset after 3 seconds
    setTimeout(() => {
        statusDisplay.textContent = originalText;
        statusDisplay.className = originalClass;
    }, 3000);
    
    // Show completion animation
    showCompletionAnimation();
}

function showCompletionAnimation() {
    const videoContainer = document.getElementById('videoContainer');
    const effect = document.createElement('div');
    effect.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 48px;
        color: #4CAF50;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        animation: fadeInOut 2s ease-in-out;
        pointer-events: none;
        z-index: 1000;
    `;
    effect.textContent = '✅ COMPLETED!';
    
    // Add CSS animation
    if (!document.getElementById('completionStyles')) {
        const style = document.createElement('style');
        style.id = 'completionStyles';
        style.textContent = `
            @keyframes fadeInOut {
                0% { opacity: 0; transform: translate(-50%, -50%) scale(0.5); }
                50% { opacity: 1; transform: translate(-50%, -50%) scale(1.2); }
                100% { opacity: 0; transform: translate(-50%, -50%) scale(1); }
            }
        `;
        document.head.appendChild(style);
    }
    
    videoContainer.style.position = 'relative';
    videoContainer.appendChild(effect);
    
    setTimeout(() => {
        if (effect.parentNode) {
            effect.parentNode.removeChild(effect);
        }
    }, 2000);
}

function updateSessionProgress() {
    const progressText = document.getElementById('progressText');
    const sessionProgress = document.getElementById('sessionProgress');
    
    const completed = completedPoses.length;
    const total = poseSequence.length;
    const percentage = (completed / total) * 100;
    
    progressText.textContent = `${completed} / ${total} poses`;
    sessionProgress.style.width = `${percentage}%`;
    
    // Check if all poses completed
    if (completed === total) {
        showSessionCompletion();
    }
}

function showSessionCompletion() {
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        z-index: 1000;
        border: 3px solid #4CAF50;
        max-width: 400px;
    `;
    
    const sessionTime = Math.floor((Date.now() - sessionStartTime) / 1000);
    const minutes = Math.floor(sessionTime / 60);
    const seconds = sessionTime % 60;
    
    modal.innerHTML = `
        <h2 style="color: #4CAF50; margin-bottom: 20px;">🎉 Session Complete!</h2>
        <p style="margin-bottom: 15px;">Congratulations! You've completed all 12 yoga poses.</p>
        <p style="margin-bottom: 20px; opacity: 0.8;">Session time: ${minutes}m ${seconds}s</p>
        <button onclick="this.parentElement.remove(); resetSession();" style="
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        ">Start New Session</button>
    `;
    
    document.body.appendChild(modal);
    
    // Voice congratulations
    if (poseDetector.voiceEnabled) {
        speakMessage('Congratulations! You have completed the full Sun Salutation sequence. Excellent dedication to your yoga practice!');
    }
    
    setTimeout(() => {
        if (modal.parentElement) {
            modal.remove();
            resetSession();
        }
    }, 15000);
}

function captureCorrectPose() {
    try {
        // Create a temporary canvas to capture the current frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Draw current video frame
        tempCtx.drawImage(poseDetector.video, 0, 0, tempCanvas.width, tempCanvas.height);
        
        // Convert to blob and download
        tempCanvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `yoga_pose_${poseSequence[currentPoseIndex]}_${Date.now()}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 'image/png');
        
        console.log(`Captured correct pose: ${poseSequence[currentPoseIndex]}`);
    } catch (error) {
        console.error('Error capturing pose:', error);
    }
}

function speakMessage(message) {
    if (poseDetector && poseDetector.voiceEnabled) {
        poseDetector.speak(message);
    }
}

function showTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    
    // Update nav buttons
    const navTabs = document.querySelectorAll('.nav-tab');
    navTabs.forEach(tab => tab.classList.remove('active'));
    event.target.classList.add('active');
}

// AI Integration Functions
function updateAnalyticsDisplay(biomechanics) {
    if (!biomechanics) return;
    
    // Create or update analytics display in sidebar
    let analyticsDiv = document.getElementById('aiAnalytics');
    if (!analyticsDiv) {
        analyticsDiv = document.createElement('div');
        analyticsDiv.id = 'aiAnalytics';
        analyticsDiv.style.cssText = `
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid #2196F3;
        `;
        document.querySelector('.controls-panel').appendChild(analyticsDiv);
    }
    
    const stability = biomechanics.stabilityIndex || 0;
    const balance = biomechanics.balanceMetrics?.stability || 'Good';
    
    analyticsDiv.innerHTML = `
        <div style="margin-bottom: 10px; font-weight: bold;">🧠 AI Analytics</div>
        <div style="font-size: 12px; margin-bottom: 5px;">
            Stability: <span style="color: #4CAF50;">${stability}%</span>
        </div>
        <div style="font-size: 12px; margin-bottom: 5px;">
            Balance: <span style="color: #4CAF50;">${balance}</span>
        </div>
        <div style="font-size: 12px; opacity: 0.8;">
            Real-time biomechanical analysis
        </div>
    `;
}

function showAIFeedback(feedback) {
    if (!feedback || !poseDetector.voiceEnabled) return;
    
    if (feedback.motivation) {
        speakMessage(feedback.motivation);
    }
}

// Keyboard shortcuts
function keyPressed() {
    if (key === ' ') {
        if (!isSessionActive) {
            startSession();
        } else {
            pauseSession();
        }
    } else if (key === 'r' || key === 'R') {
        resetSession();
    } else if (key >= '1' && key <= '9') {
        const poseIndex = parseInt(key) - 1;
        if (poseIndex < poseSequence.length) {
            selectPose(poseIndex);
        }
    }
}

// Window resize handler
function windowResized() {
    const container = document.getElementById('canvas-container');
    if (container && canvas) {
        const containerRect = container.getBoundingClientRect();
        let newWidth = Math.floor(containerRect.width) || 640;
        let newHeight = Math.floor(containerRect.height) || 480;
        
        // Maintain aspect ratio
        const aspectRatio = 4/3;
        if (newWidth / newHeight > aspectRatio) {
            newWidth = newHeight * aspectRatio;
        } else {
            newHeight = newWidth / aspectRatio;
        }
        
        resizeCanvas(Math.max(newWidth, 320), Math.max(newHeight, 240));
    }
}

// Error handling
window.addEventListener('error', (e) => {
    console.error('Application error:', e.error);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
    e.preventDefault();
});

function addManualCaptureButton() {
    const controlsPanel = document.querySelector('.controls');
    if (controlsPanel) {
        const captureBtn = document.createElement('button');
        captureBtn.className = 'btn btn-secondary';
        captureBtn.textContent = '📸 Capture Training Image';
        captureBtn.onclick = captureTrainingImage;
        controlsPanel.appendChild(captureBtn);
    }
}

// Zoom control functions
function adjustZoom(delta) {
    zoomLevel = Math.max(0.5, Math.min(3.0, zoomLevel + delta));
    document.getElementById('zoomLevel').textContent = Math.round(zoomLevel * 100) + '%';
}

function resetZoom() {
    zoomLevel = 1.0;
    videoOffsetX = 0;
    videoOffsetY = 0;
    document.getElementById('zoomLevel').textContent = '100%';
}

function updateCorrectionsDisplay() {
    const correctionsTextEl = document.getElementById('correctionsText');
    if (correctionsTextEl && poseDetector && poseDetector.poseCorrections) {
        if (poseDetector.poseCorrections.length > 0) {
            correctionsTextEl.innerHTML = poseDetector.poseCorrections.map(correction => 
                `• ${correction}`
            ).join('<br>');
        } else if (poseDetector.isCurrentPoseCorrect()) {
            correctionsTextEl.textContent = '✅ Perfect alignment! Hold this position.';
        } else {
            correctionsTextEl.textContent = 'Adjust your posture for better alignment.';
        }
    }
}

// Global function for tab switching
function showTab(tabName) {
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    document.getElementById(tabName).classList.add('active');
    
    const navTabs = document.querySelectorAll('.nav-tab');
    navTabs.forEach(tab => tab.classList.remove('active'));
    
    // Find and activate the correct nav tab
    navTabs.forEach(tab => {
        if (tab.textContent.toLowerCase().includes(tabName)) {
            tab.classList.add('active');
        }
    });
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing Enhanced Yoga Studio with 12 poses...');
    
    // Register service worker
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/service-worker.js')
            .then(registration => console.log('Service Worker registered:', registration))
            .catch(error => console.log('Service Worker registration failed:', error));
    }
});