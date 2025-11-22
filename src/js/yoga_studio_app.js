// Main Yoga Studio Application
let poseDetector;
let yogaTimer;
let canvas;
let isSessionActive = false;

// Pose descriptions for UI
const poseDescriptions = {
    0: "Stand with palms together in prayer position at chest level.",
    1: "Raise both arms overhead, palms facing each other.",
    2: "Bend forward, hands reaching toward feet.",
    3: "Step back into low lunge, hands on ground.",
    4: "Sit with legs extended, spine straight, hands beside hips.",
    5: "Lower knees, chest, and chin to ground.",
    6: "Lie on stomach, lift chest with arms support.",
    7: "Form inverted V-shape, hands and feet on ground.",
    8: "Sit cross-legged, feet on opposite thighs.",
    9: "Stand tall, arms at sides, body aligned."
};

const poseNames = [
    "Pranamasana", "Hastauttanasana", "Hastapadasana", 
    "Ashwa Sanchalanasana", "Dandasana", "Ashtanga Namaskara",
    "Bhujangasana", "Adho Mukha Svanasana", "Padmasana", "Tadasana"
];

function setup() {
    // Get optimal canvas size for device
    let canvasWidth = 640;
    let canvasHeight = 480;
    
    if (deviceCompatibility) {
        if (deviceCompatibility.isMobile) {
            canvasWidth = Math.min(window.innerWidth - 40, 480);
            canvasHeight = (canvasWidth * 3) / 4; // 4:3 aspect ratio
        } else if (deviceCompatibility.isTablet) {
            canvasWidth = Math.min(window.innerWidth - 300, 640);
            canvasHeight = (canvasWidth * 3) / 4;
        }
    }
    
    // Create responsive canvas
    canvas = createCanvas(canvasWidth, canvasHeight);
    canvas.parent('canvas-container');
    
    // Initialize components
    poseDetector = new YogaPoseDetector();
    yogaTimer = new YogaTimer();
    
    // Initialize pose detection
    poseDetector.initialize();
    
    // Setup UI
    setupUI();
    
    console.log('Yoga Studio initialized');
}

function draw() {
    if (poseDetector.video) {
        // Display video with responsive sizing
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        image(poseDetector.video, 0, 0, canvasWidth, canvasHeight);
        
        // Draw pose landmarks and skeleton
        poseDetector.drawPose();
        
        // Update timer with pose status
        if (isSessionActive) {
            yogaTimer.updateWithPoseStatus(
                poseDetector.isCurrentPoseCorrect(),
                poseDetector.getPoseAccuracy()
            );
            
            // Check for pose completion
            if (yogaTimer.checkPoseCompletion()) {
                showPoseCompletionFeedback();
            }
        }
        
        // Display pose hold progress
        displayPoseHoldProgress();
    }
}

function setupUI() {
    // Populate pose list
    const poseList = document.getElementById('poseList');
    poseNames.forEach((name, index) => {
        const listItem = document.createElement('li');
        listItem.className = 'pose-item';
        listItem.textContent = `${index + 1}. ${name}`;
        listItem.onclick = () => selectPose(index);
        poseList.appendChild(listItem);
    });
    
    // Setup event listeners
    document.getElementById('startBtn').onclick = startSession;
    document.getElementById('pauseBtn').onclick = pauseSession;
    document.getElementById('resetBtn').onclick = resetSession;
    
    document.getElementById('poseSelect').onchange = (e) => {
        selectPose(parseInt(e.target.value));
    };
    
    // Add pose hold progress bar to sidebar
    addPoseHoldProgressBar();
    
    // Initial pose selection
    selectPose(0);
}

function addPoseHoldProgressBar() {
    const sidebar = document.querySelector('.sidebar');
    const progressContainer = document.createElement('div');
    progressContainer.innerHTML = `
        <div style="margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Pose Hold Progress</span>
                <span id="holdTimeDisplay">0s / 30s</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="poseHoldProgress"></div>
            </div>
        </div>
    `;
    
    // Insert after accuracy display
    const accuracyDisplay = document.querySelector('.accuracy-display');
    accuracyDisplay.parentNode.insertBefore(progressContainer, accuracyDisplay.nextSibling);
}

function selectPose(poseIndex) {
    // Update pose detector
    poseDetector.setTargetPose(poseIndex);
    
    // Update UI
    document.getElementById('currentPoseName').textContent = poseNames[poseIndex];
    document.getElementById('poseDescription').textContent = poseDescriptions[poseIndex];
    
    // Update pose selector
    document.getElementById('poseSelect').value = poseIndex;
    
    // Update pose list highlighting
    const poseItems = document.querySelectorAll('.pose-item');
    poseItems.forEach((item, index) => {
        if (index === poseIndex) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    // Reset timer when changing poses
    if (isSessionActive) {
        yogaTimer.reset();
        yogaTimer.start();
    }
    
    console.log(`Selected pose: ${poseNames[poseIndex]}`);
}

function startSession() {
    if (!isSessionActive) {
        isSessionActive = true;
        yogaTimer.start();
        
        // Update button states
        document.getElementById('startBtn').textContent = 'Resume';
        document.getElementById('pauseBtn').disabled = false;
        
        // Update status
        document.getElementById('statusDisplay').textContent = 'Session Started - Position Yourself';
        
        console.log('Yoga session started');
    } else {
        yogaTimer.start();
        document.getElementById('startBtn').textContent = 'Resume';
    }
}

function pauseSession() {
    if (isSessionActive) {
        yogaTimer.pause();
        
        // Update button states
        if (yogaTimer.isPaused) {
            document.getElementById('startBtn').textContent = 'Resume';
            document.getElementById('statusDisplay').textContent = 'Session Paused';
        } else {
            document.getElementById('startBtn').textContent = 'Pause';
            document.getElementById('statusDisplay').textContent = 'Session Resumed';
        }
        
        console.log('Yoga session paused/resumed');
    }
}

function resetSession() {
    isSessionActive = false;
    yogaTimer.reset();
    
    // Reset button states
    document.getElementById('startBtn').textContent = 'Start Session';
    document.getElementById('pauseBtn').disabled = true;
    
    // Reset UI
    document.getElementById('statusDisplay').textContent = 'Position yourself in front of camera';
    document.getElementById('statusDisplay').className = 'status-display';
    document.getElementById('videoContainer').className = 'video-container';
    document.getElementById('accuracyValue').textContent = '0%';
    document.getElementById('accuracyBar').style.width = '0%';
    
    // Reset pose hold progress
    const poseHoldProgress = document.getElementById('poseHoldProgress');
    if (poseHoldProgress) {
        poseHoldProgress.style.width = '0%';
    }
    
    const holdTimeDisplay = document.getElementById('holdTimeDisplay');
    if (holdTimeDisplay) {
        holdTimeDisplay.textContent = '0s / 30s';
    }
    
    console.log('Yoga session reset');
}

function displayPoseHoldProgress() {
    const holdTimeDisplay = document.getElementById('holdTimeDisplay');
    if (holdTimeDisplay && isSessionActive) {
        const currentHoldTime = Math.floor(yogaTimer.getPoseHoldTime());
        const targetTime = yogaTimer.targetHoldTime;
        holdTimeDisplay.textContent = `${currentHoldTime}s / ${targetTime}s`;
    }
}

function showPoseCompletionFeedback() {
    // Visual feedback for pose completion
    const statusDisplay = document.getElementById('statusDisplay');
    const originalText = statusDisplay.textContent;
    const originalClass = statusDisplay.className;
    
    statusDisplay.textContent = '🎉 Pose Completed! Great Job!';
    statusDisplay.className = 'status-display status-correct';
    statusDisplay.style.animation = 'pulse 0.5s ease-in-out';
    
    // Reset after 2 seconds
    setTimeout(() => {
        statusDisplay.textContent = originalText;
        statusDisplay.className = originalClass;
        statusDisplay.style.animation = '';
    }, 2000);
    
    // Show completion animation
    showCompletionAnimation();
}

function showCompletionAnimation() {
    // Create completion effect
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
    effect.textContent = '✅ POSE COMPLETED!';
    
    // Add CSS animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes fadeInOut {
            0% { opacity: 0; transform: translate(-50%, -50%) scale(0.5); }
            50% { opacity: 1; transform: translate(-50%, -50%) scale(1.2); }
            100% { opacity: 0; transform: translate(-50%, -50%) scale(1); }
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
    `;
    document.head.appendChild(style);
    
    videoContainer.style.position = 'relative';
    videoContainer.appendChild(effect);
    
    // Remove effect after animation
    setTimeout(() => {
        if (effect.parentNode) {
            effect.parentNode.removeChild(effect);
        }
    }, 2000);
}

// Keyboard shortcuts
function keyPressed() {
    if (key === ' ') { // Spacebar to start/pause
        if (!isSessionActive) {
            startSession();
        } else {
            pauseSession();
        }
    } else if (key === 'r' || key === 'R') { // R to reset
        resetSession();
    } else if (key >= '1' && key <= '9') { // Number keys to select poses
        const poseIndex = parseInt(key) - 1;
        if (poseIndex < poseNames.length) {
            selectPose(poseIndex);
        }
    } else if (key === '0') { // 0 for 10th pose
        selectPose(9);
    }
}

// Window resize handler
function windowResized() {
    // Maintain aspect ratio
    const container = document.getElementById('canvas-container');
    if (container) {
        const containerWidth = container.offsetWidth;
        const aspectRatio = 640 / 480;
        const newHeight = containerWidth / aspectRatio;
        resizeCanvas(containerWidth, newHeight);
    }
}

// Error handling
window.addEventListener('error', (e) => {
    console.error('Application error:', e.error);
    const statusDisplay = document.getElementById('statusDisplay');
    if (statusDisplay) {
        statusDisplay.textContent = 'Error: Please refresh the page';
        statusDisplay.className = 'status-display status-incorrect';
    }
});

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing Yoga Studio...');
    
    // Register service worker for offline functionality
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/service-worker.js')
            .then((registration) => {
                console.log('Service Worker registered:', registration);
            })
            .catch((error) => {
                console.log('Service Worker registration failed:', error);
            });
    }
    
    // Add to home screen prompt for mobile
    let deferredPrompt;
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        
        // Show install button
        const installBtn = document.createElement('button');
        installBtn.textContent = '📱 Install App';
        installBtn.className = 'btn btn-secondary';
        installBtn.style.position = 'fixed';
        installBtn.style.bottom = '20px';
        installBtn.style.right = '20px';
        installBtn.style.zIndex = '1000';
        
        installBtn.addEventListener('click', () => {
            deferredPrompt.prompt();
            deferredPrompt.userChoice.then((choiceResult) => {
                if (choiceResult.outcome === 'accepted') {
                    console.log('User accepted the install prompt');
                }
                deferredPrompt = null;
                installBtn.remove();
            });
        });
        
        document.body.appendChild(installBtn);
    });
});