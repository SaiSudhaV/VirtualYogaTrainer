// Main Yoga Studio Application with Enhanced AI Integration
let poseDetector;
let yogaTimer;
let canvas;
let isSessionActive = false;

// AI Enhancement Components
let aiCoach;
let smartSequenceDetector;
let poseAnalytics;
let currentSessionData = {
    startTime: null,
    poses: [],
    corrections: [],
    analytics: []
};

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
    
    // Initialize AI Enhancement Components
    aiCoach = new AIYogaCoach();
    smartSequenceDetector = new SmartSequenceDetector();
    poseAnalytics = new PoseAnalytics();
    
    // Initialize pose detection
    poseDetector.initialize();
    
    console.log('AI enhancements loaded: Coach, Sequence Detector, Analytics');
    
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
            const isCorrect = poseDetector.isCurrentPoseCorrect();
            const accuracy = poseDetector.getPoseAccuracy();
            
            yogaTimer.updateWithPoseStatus(isCorrect, accuracy);
            
            // AI Analytics Integration
            if (poseAnalytics) {
                const biomechanics = poseAnalytics.analyzeBiomechanics(
                    poseDetector.poses[0]?.pose?.keypoints || [],
                    poseDetector.currentTargetPose
                );
                
                const injuryRisk = poseAnalytics.analyzeInjuryRisk(
                    poseDetector.poses[0]?.pose?.keypoints || [],
                    poseDetector.currentTargetPose,
                    yogaTimer.getPoseHoldTime()
                );
                
                updateAnalyticsDisplay(biomechanics, injuryRisk);
            }
            
            // Smart Sequence Integration
            if (smartSequenceDetector && smartSequenceDetector.isActive()) {
                const sequenceGuidance = smartSequenceDetector.updateSequence(
                    poseDetector.currentTargetPose,
                    accuracy,
                    isCorrect
                );
                updateSequenceDisplay(sequenceGuidance);
            }
            
            // Check for pose completion
            if (yogaTimer.checkPoseCompletion()) {
                handlePoseCompletion(accuracy, poseDetector.poseCorrections);
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
    
    // Voice guidance toggle
    document.getElementById('voiceToggle').onchange = (e) => {
        if (poseDetector) {
            poseDetector.voiceEnabled = e.target.checked;
            if (e.target.checked) {
                console.log('AI Voice Guidance enabled');
            } else {
                console.log('AI Voice Guidance disabled');
                // Stop any current speech
                if (poseDetector.speechSynthesis) {
                    poseDetector.speechSynthesis.cancel();
                }
            }
        }
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
        
        // Initialize session data
        currentSessionData = {
            startTime: Date.now(),
            poses: [],
            corrections: [],
            analytics: []
        };
        
        // Update button states
        document.getElementById('startBtn').textContent = 'Resume';
        document.getElementById('pauseBtn').disabled = false;
        
        // Update status
        document.getElementById('statusDisplay').textContent = 'AI Session Started - Position Yourself';
        
        // AI Coach welcome
        if (aiCoach && poseDetector.voiceEnabled) {
            const welcomeMessage = "AI Yoga Coach activated. I'll provide personalized guidance throughout your session.";
            const utterance = new SpeechSynthesisUtterance(welcomeMessage);
            utterance.rate = 0.8;
            window.speechSynthesis.speak(utterance);
        }
        
        console.log('Enhanced AI Yoga session started');
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

function handlePoseCompletion(accuracy, corrections) {
    // AI Coach feedback
    if (aiCoach) {
        const feedback = aiCoach.trackProgress(
            poseDetector.currentTargetPose,
            accuracy,
            yogaTimer.getPoseHoldTime(),
            corrections
        );
        
        // Store session data
        currentSessionData.poses.push({
            pose: poseDetector.currentTargetPose,
            accuracy: accuracy,
            holdTime: yogaTimer.getPoseHoldTime(),
            timestamp: Date.now()
        });
        
        showAIFeedback(feedback);
    }
    
    showPoseCompletionFeedback();
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

// AI Integration Functions
function updateAnalyticsDisplay(biomechanics, injuryRisk) {
    if (!biomechanics || !injuryRisk) return;
    
    // Update safety indicators
    const safetyScore = injuryRisk.safetyScore;
    let safetyColor = '#4CAF50';
    if (safetyScore < 70) safetyColor = '#f44336';
    else if (safetyScore < 85) safetyColor = '#FF9800';
    
    // Create or update analytics panel
    let analyticsPanel = document.getElementById('analyticsPanel');
    if (!analyticsPanel) {
        analyticsPanel = createAnalyticsPanel();
    }
    
    // Update analytics content
    analyticsPanel.innerHTML = `
        <div style="margin-bottom: 10px;">
            <strong>🧠 AI Analytics</strong>
        </div>
        <div style="font-size: 12px; margin-bottom: 5px;">
            Safety Score: <span style="color: ${safetyColor}; font-weight: bold;">${safetyScore}%</span>
        </div>
        <div style="font-size: 12px; margin-bottom: 5px;">
            Balance: ${biomechanics.balanceMetrics?.stability || 'Analyzing...'}
        </div>
        <div style="font-size: 12px; margin-bottom: 5px;">
            Stability: ${biomechanics.stabilityIndex || 0}%
        </div>
        ${injuryRisk.warnings.length > 0 ? 
            `<div style="font-size: 11px; color: #FF9800; margin-top: 5px;">
                ⚠️ ${injuryRisk.warnings[0]}
            </div>` : ''}
    `;
}

function createAnalyticsPanel() {
    const panel = document.createElement('div');
    panel.id = 'analyticsPanel';
    panel.style.cssText = `
        background: rgba(0, 0, 0, 0.3);
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
        font-size: 12px;
    `;
    
    const sidebar = document.querySelector('.sidebar');
    const controls = document.querySelector('.controls');
    sidebar.insertBefore(panel, controls);
    
    return panel;
}

function updateSequenceDisplay(sequenceGuidance) {
    if (!sequenceGuidance) return;
    
    let sequencePanel = document.getElementById('sequencePanel');
    if (!sequencePanel) {
        sequencePanel = createSequencePanel();
    }
    
    const progressPercent = sequenceGuidance.progress || 0;
    const statusIcon = sequenceGuidance.status === 'correct' ? '✅' : 
                      sequenceGuidance.status === 'adjust' ? '🔄' : '❌';
    
    sequencePanel.innerHTML = `
        <div style="margin-bottom: 10px;">
            <strong>🔄 Sequence Flow</strong>
        </div>
        <div style="font-size: 12px; margin-bottom: 5px;">
            Step ${sequenceGuidance.currentStep}/${sequenceGuidance.totalSteps}
        </div>
        <div class="progress-bar" style="height: 8px; margin: 5px 0;">
            <div class="progress-fill" style="width: ${progressPercent}%;"></div>
        </div>
        <div style="font-size: 11px; margin-bottom: 5px;">
            ${statusIcon} ${sequenceGuidance.message}
        </div>
        ${sequenceGuidance.instruction ? 
            `<div style="font-size: 10px; color: #ccc; font-style: italic;">
                ${sequenceGuidance.instruction}
            </div>` : ''}
    `;
}

function createSequencePanel() {
    const panel = document.createElement('div');
    panel.id = 'sequencePanel';
    panel.style.cssText = `
        background: rgba(0, 0, 0, 0.3);
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #9C27B0;
        font-size: 12px;
        display: none;
    `;
    
    const sidebar = document.querySelector('.sidebar');
    const analyticsPanel = document.getElementById('analyticsPanel');
    if (analyticsPanel) {
        sidebar.insertBefore(panel, analyticsPanel.nextSibling);
    } else {
        const controls = document.querySelector('.controls');
        sidebar.insertBefore(panel, controls);
    }
    
    return panel;
}

function showAIFeedback(feedback) {
    if (!feedback || !poseDetector.voiceEnabled) return;
    
    // Show motivational message
    if (feedback.motivation && window.speechSynthesis) {
        const utterance = new SpeechSynthesisUtterance(feedback.motivation);
        utterance.rate = 0.8;
        utterance.pitch = 1.1;
        window.speechSynthesis.speak(utterance);
    }
    
    // Update UI with feedback
    const statusDisplay = document.getElementById('statusDisplay');
    if (statusDisplay && feedback.overall) {
        const originalText = statusDisplay.textContent;
        statusDisplay.textContent = `🤖 AI: ${feedback.overall}`;
        
        setTimeout(() => {
            statusDisplay.textContent = originalText;
        }, 3000);
    }
}

function startSequenceMode() {
    if (!smartSequenceDetector) return;
    
    // Show sequence selection
    const sequences = smartSequenceDetector.getAvailableSequences();
    const sequenceKeys = Object.keys(sequences);
    
    if (sequenceKeys.length === 0) return;
    
    // For demo, start with beginner flow
    const success = smartSequenceDetector.startSequence('beginner_flow', aiCoach);
    
    if (success) {
        document.getElementById('sequencePanel').style.display = 'block';
        
        // Update UI to show sequence mode
        const statusDisplay = document.getElementById('statusDisplay');
        statusDisplay.textContent = '🔄 Sequence Mode: Follow the guided flow';
        
        console.log('Sequence mode activated');
    }
}

function generatePersonalizedSequence() {
    if (!smartSequenceDetector || !aiCoach) return;
    
    const personalizedSeq = smartSequenceDetector.generatePersonalizedSequence(
        aiCoach, 240, 'balanced'
    );
    
    if (personalizedSeq) {
        console.log('Generated personalized sequence:', personalizedSeq.sequence.name);
        
        // Start the personalized sequence
        smartSequenceDetector.startSequence(personalizedSeq.id, aiCoach);
        document.getElementById('sequencePanel').style.display = 'block';
        
        // Announce the personalized sequence
        if (poseDetector.voiceEnabled && window.speechSynthesis) {
            const message = `I've created a personalized sequence for you: ${personalizedSeq.sequence.name}`;
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.rate = 0.8;
            window.speechSynthesis.speak(utterance);
        }
    }
}

function showProgressAnalytics() {
    if (!aiCoach) return;
    
    const analytics = aiCoach.getProgressAnalytics();
    
    // Create analytics modal or panel
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 20px;
        border-radius: 10px;
        max-width: 400px;
        z-index: 1000;
        border: 2px solid #4CAF50;
    `;
    
    modal.innerHTML = `
        <h3>📊 Your Progress Analytics</h3>
        <div style="margin: 15px 0;">
            <strong>Total Sessions:</strong> ${analytics.totalSessions}<br>
            <strong>Average Accuracy:</strong> ${analytics.averageAccuracy.toFixed(1)}%<br>
            <strong>Current Level:</strong> ${analytics.currentLevel}<br>
            <strong>Consistency Score:</strong> ${analytics.consistencyScore.toFixed(1)}%
        </div>
        ${analytics.strongestPoses.length > 0 ? `
            <div style="margin: 10px 0;">
                <strong>💪 Your Strengths:</strong><br>
                ${analytics.strongestPoses.map(p => `• ${p.name} (${p.accuracy.toFixed(1)}%)`).join('<br>')}
            </div>
        ` : ''}
        ${analytics.improvementAreas.length > 0 ? `
            <div style="margin: 10px 0;">
                <strong>🎯 Focus Areas:</strong><br>
                ${analytics.improvementAreas.map(p => `• ${p.name} (${p.accuracy.toFixed(1)}%)`).join('<br>')}
            </div>
        ` : ''}
        <button onclick="this.parentElement.remove()" style="
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 15px;
        ">Close</button>
    `;
    
    document.body.appendChild(modal);
    
    // Auto-remove after 10 seconds
    setTimeout(() => {
        if (modal.parentElement) {
            modal.remove();
        }
    }, 10000);
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing Enhanced AI Yoga Studio...');
    
    // Add AI control buttons
    setTimeout(() => {
        addAIControlButtons();
    }, 1000);
    
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

function addAIControlButtons() {
    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;
    
    const aiControlsDiv = document.createElement('div');
    aiControlsDiv.innerHTML = `
        <div style="margin: 15px 0; padding: 15px; background: rgba(0,0,0,0.3); border-radius: 8px; border-left: 4px solid #FF5722;">
            <div style="margin-bottom: 10px; font-weight: bold;">🤖 AI Features</div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                <button onclick="startSequenceMode()" class="btn" style="background: #9C27B0; color: white; padding: 6px 12px; font-size: 12px;">Sequence Mode</button>
                <button onclick="generatePersonalizedSequence()" class="btn" style="background: #FF5722; color: white; padding: 6px 12px; font-size: 12px;">Personal Flow</button>
                <button onclick="showProgressAnalytics()" class="btn" style="background: #607D8B; color: white; padding: 6px 12px; font-size: 12px;">Analytics</button>
            </div>
        </div>
    `;
    
    const controls = document.querySelector('.controls');
    sidebar.insertBefore(aiControlsDiv, controls);
}