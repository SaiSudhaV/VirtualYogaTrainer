// Enhanced Yoga Pose Detector with 12 Poses and Webcam Fix
class YogaPoseDetector {
    constructor() {
        this.poseNet = null;
        this.video = null;
        this.poses = [];
        this.isModelReady = false;
        this.canvas = null;
        this.ctx = null;
        
        // AI Correction System
        this.speechSynthesis = window.speechSynthesis;
        this.lastCorrectionTime = 0;
        this.correctionCooldown = 3000;
        this.correctPoseFrames = 0;
        this.requiredCorrectFrames = 30;
        this.captureCanvas = null;
        this.captureContext = null;
        
        // 12 Pose definitions for Sun Salutation
        this.poseDefinitions = {
            0: { // Pranamasana
                name: "Pranamasana",
                description: "Stand with palms together in prayer position at chest level.",
                targetAngles: { leftArm: 90, rightArm: 90, leftLeg: 180, rightLeg: 180 },
                threshold: 25
            },
            1: { // Hasta Uttanasana
                name: "Hasta Uttanasana", 
                description: "Raise both arms overhead, palms facing each other.",
                targetAngles: { leftArm: 180, rightArm: 180, leftLeg: 180, rightLeg: 180 },
                threshold: 30
            },
            2: { // Pada Hastasana
                name: "Pada Hastasana",
                description: "Bend forward, hands reaching toward feet.",
                targetAngles: { leftArm: 45, rightArm: 45, torso: 45 },
                threshold: 35
            },
            3: { // Ashwa Sanchalanasana
                name: "Ashwa Sanchalanasana",
                description: "Step back into low lunge, hands on ground.",
                targetAngles: { leftLeg: 90, rightLeg: 160 },
                threshold: 40
            },
            4: { // Parvatasana
                name: "Parvatasana",
                description: "Form inverted V-shape, hands and feet on ground.",
                targetAngles: { leftArm: 45, rightArm: 45, leftLeg: 45, rightLeg: 45 },
                threshold: 35
            },
            5: { // Ashtanga Namaskara
                name: "Ashtanga Namaskara",
                description: "Lower knees, chest, and chin to ground.",
                targetAngles: { leftArm: 45, rightArm: 45 },
                threshold: 35
            },
            6: { // Bhujangasana
                name: "Bhujangasana",
                description: "Lie on stomach, lift chest with arms support.",
                targetAngles: { leftArm: 120, rightArm: 120, torso: 45 },
                threshold: 30
            },
            7: { // Parvatasana (repeat)
                name: "Parvatasana",
                description: "Form inverted V-shape, hands and feet on ground.",
                targetAngles: { leftArm: 45, rightArm: 45, leftLeg: 45, rightLeg: 45 },
                threshold: 35
            },
            8: { // Ashwa Sanchalanasana (repeat)
                name: "Ashwa Sanchalanasana",
                description: "Step back into low lunge, hands on ground.",
                targetAngles: { leftLeg: 90, rightLeg: 160 },
                threshold: 40
            },
            9: { // Pada Hastasana (repeat)
                name: "Pada Hastasana",
                description: "Bend forward, hands reaching toward feet.",
                targetAngles: { leftArm: 45, rightArm: 45, torso: 45 },
                threshold: 35
            },
            10: { // Hasta Uttanasana (repeat)
                name: "Hasta Uttanasana",
                description: "Raise both arms overhead, palms facing each other.",
                targetAngles: { leftArm: 180, rightArm: 180, leftLeg: 180, rightLeg: 180 },
                threshold: 30
            },
            11: { // Pranamasana (repeat)
                name: "Pranamasana",
                description: "Stand with palms together in prayer position at chest level.",
                targetAngles: { leftArm: 90, rightArm: 90, leftLeg: 180, rightLeg: 180 },
                threshold: 25
            }
        };
        
        this.currentTargetPose = 0;
        this.poseAccuracy = 0;
        this.isCorrectPose = false;
        this.poseCorrections = [];
        this.voiceEnabled = true;
        
        this.initializeCaptureCanvas();
    }
    
    initializeCaptureCanvas() {
        this.captureCanvas = document.createElement('canvas');
        this.captureCanvas.width = 640;
        this.captureCanvas.height = 480;
        this.captureContext = this.captureCanvas.getContext('2d');
    }

    async initialize() {
        try {
            console.log('Initializing webcam and pose detection...');
            
            // Request camera permissions first
            await this.requestCameraPermission();
            
            // Create video capture
            this.video = createCapture(VIDEO, () => {
                console.log('Webcam initialized successfully');
            });
            
            this.video.size(640, 480);
            this.video.hide();
            
            // Wait for ml5 and initialize PoseNet
            await this.initializePoseNet();
            
        } catch (error) {
            console.error('Initialization failed:', error);
            this.handleInitializationError(error);
        }
    }
    
    async requestCameraPermission() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' },
                audio: false
            });
            
            // Stop the stream immediately as we just needed permission
            stream.getTracks().forEach(track => track.stop());
            console.log('Camera permission granted');
            
        } catch (error) {
            throw new Error('Camera access denied. Please allow camera permissions.');
        }
    }
    
    async initializePoseNet() {
        return new Promise((resolve, reject) => {
            if (typeof ml5 === 'undefined') {
                reject(new Error('ml5.js library not loaded'));
                return;
            }
            
            const options = {
                architecture: 'MobileNetV1',
                imageScaleFactor: 0.3,
                outputStride: 16,
                flipHorizontal: true,
                minConfidence: 0.5,
                maxPoseDetections: 1,
                scoreThreshold: 0.5,
                nmsRadius: 20,
                detectionType: 'single',
                inputResolution: 513,
                multiplier: 0.75,
                quantBytes: 2
            };
            
            this.poseNet = ml5.poseNet(this.video, options, () => {
                this.isModelReady = true;
                console.log('PoseNet model loaded successfully');
                
                if (this.voiceEnabled) {
                    this.speak('Yoga pose detection ready. Position yourself in front of the camera.');
                }
                
                resolve();
            });
            
            this.poseNet.on('pose', (results) => {
                this.poses = results;
                this.analyzePose();
            });
        });
    }
    
    handleInitializationError(error) {
        const statusDisplay = document.getElementById('statusDisplay');
        if (statusDisplay) {
            if (error.message.includes('Camera') || error.message.includes('denied')) {
                statusDisplay.innerHTML = `
                    📷 Camera access required<br>
                    <small>Please allow camera permissions and refresh</small>
                `;
            } else {
                statusDisplay.textContent = '⚠️ Initialization failed. Please refresh the page.';
            }
            statusDisplay.className = 'status-display status-incorrect';
        }
        
        this.showCameraInstructions();
    }
    
    showCameraInstructions() {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.9); color: white; padding: 30px;
            border-radius: 15px; text-align: center; z-index: 1000;
            border: 2px solid #f44336; max-width: 400px;
        `;
        
        modal.innerHTML = `
            <h3 style="color: #f44336; margin-bottom: 20px;">📷 Camera Access Required</h3>
            <p style="margin-bottom: 15px;">To use yoga pose detection:</p>
            <ol style="text-align: left; margin-bottom: 20px;">
                <li>Click the camera icon in your browser address bar</li>
                <li>Select "Allow" for camera access</li>
                <li>Refresh the page</li>
            </ol>
            <button onclick="location.reload()" style="
                background: #4CAF50; color: white; border: none;
                padding: 12px 24px; border-radius: 25px; cursor: pointer;
                font-size: 16px; font-weight: bold;
            ">Refresh Page</button>
        `;
        
        document.body.appendChild(modal);
    }

    analyzePose() {
        if (this.poses.length > 0) {
            const pose = this.poses[0].pose;
            const keypoints = pose.keypoints;
            
            const currentAngles = this.calculatePoseAngles(keypoints);
            const targetPose = this.poseDefinitions[this.currentTargetPose];
            this.poseAccuracy = this.calculatePoseAccuracy(currentAngles, targetPose);
            
            this.poseCorrections = this.analyzePostureCorrections(keypoints, currentAngles, targetPose);
            this.isCorrectPose = this.poseAccuracy >= 75;
            
            if (this.isCorrectPose) {
                this.correctPoseFrames++;
                if (this.correctPoseFrames >= this.requiredCorrectFrames) {
                    this.captureCorrectPose();
                    this.correctPoseFrames = 0;
                }
            } else {
                this.correctPoseFrames = 0;
                this.provideVoiceCorrection();
            }
            
            this.updatePoseStatus();
        }
    }

    calculatePoseAngles(keypoints) {
        const angles = {};
        
        const nose = keypoints[0];
        const leftShoulder = keypoints[5];
        const rightShoulder = keypoints[6];
        const leftElbow = keypoints[7];
        const rightElbow = keypoints[8];
        const leftWrist = keypoints[9];
        const rightWrist = keypoints[10];
        const leftHip = keypoints[11];
        const rightHip = keypoints[12];
        const leftKnee = keypoints[13];
        const rightKnee = keypoints[14];
        const leftAnkle = keypoints[15];
        const rightAnkle = keypoints[16];

        if (leftShoulder.score > 0.5 && leftElbow.score > 0.5 && leftWrist.score > 0.5) {
            angles.leftArm = this.calculateAngle(
                leftShoulder.position, leftElbow.position, leftWrist.position
            );
        }
        
        if (rightShoulder.score > 0.5 && rightElbow.score > 0.5 && rightWrist.score > 0.5) {
            angles.rightArm = this.calculateAngle(
                rightShoulder.position, rightElbow.position, rightWrist.position
            );
        }

        if (leftHip.score > 0.5 && leftKnee.score > 0.5 && leftAnkle.score > 0.5) {
            angles.leftLeg = this.calculateAngle(
                leftHip.position, leftKnee.position, leftAnkle.position
            );
        }
        
        if (rightHip.score > 0.5 && rightKnee.score > 0.5 && rightAnkle.score > 0.5) {
            angles.rightLeg = this.calculateAngle(
                rightHip.position, rightKnee.position, rightAnkle.position
            );
        }

        if (nose.score > 0.5 && leftHip.score > 0.5 && rightHip.score > 0.5) {
            const hipCenter = {
                x: (leftHip.position.x + rightHip.position.x) / 2,
                y: (leftHip.position.y + rightHip.position.y) / 2
            };
            angles.torso = Math.abs(Math.atan2(
                nose.position.y - hipCenter.y,
                nose.position.x - hipCenter.x
            ) * 180 / Math.PI);
        }

        return angles;
    }

    calculateAngle(a, b, c) {
        const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
        let angle = Math.abs(radians * 180.0 / Math.PI);
        if (angle > 180.0) {
            angle = 360 - angle;
        }
        return angle;
    }

    calculatePoseAccuracy(currentAngles, targetPose) {
        let totalScore = 0;
        let angleCount = 0;

        for (const [angleName, targetAngle] of Object.entries(targetPose.targetAngles)) {
            if (currentAngles[angleName] !== undefined) {
                const difference = Math.abs(currentAngles[angleName] - targetAngle);
                const score = Math.max(0, 100 - (difference / targetPose.threshold) * 100);
                totalScore += score;
                angleCount++;
            }
        }

        return angleCount > 0 ? totalScore / angleCount : 0;
    }

    updatePoseStatus() {
        const accuracyElement = document.getElementById('accuracyValue');
        const accuracyBar = document.getElementById('accuracyBar');
        
        if (accuracyElement && accuracyBar) {
            accuracyElement.textContent = Math.round(this.poseAccuracy) + '%';
            accuracyBar.style.width = this.poseAccuracy + '%';
            
            if (this.poseAccuracy >= 75) {
                accuracyBar.style.background = 'linear-gradient(90deg, #4CAF50, #8BC34A)';
            } else if (this.poseAccuracy >= 50) {
                accuracyBar.style.background = 'linear-gradient(90deg, #FF9800, #FFC107)';
            } else {
                accuracyBar.style.background = 'linear-gradient(90deg, #f44336, #FF5722)';
            }
        }

        const statusElement = document.getElementById('statusDisplay');
        const videoContainer = document.getElementById('videoContainer');
        
        if (statusElement && videoContainer) {
            if (this.isCorrectPose) {
                statusElement.textContent = '✅ Perfect Pose - Hold Position!';
                statusElement.className = 'status-display status-correct';
                videoContainer.className = 'video-container correct';
            } else {
                const correction = this.poseCorrections.length > 0 ? this.poseCorrections[0] : 'Adjust Your Position';
                statusElement.textContent = `❌ ${correction}`;
                statusElement.className = 'status-display status-incorrect';
                videoContainer.className = 'video-container incorrect';
            }
        }
    }

    drawPose() {
        if (this.poses.length > 0) {
            const pose = this.poses[0].pose;
            
            noStroke();
            for (let keypoint of pose.keypoints) {
                if (keypoint.score > 0.2) {
                    if (this.isCorrectPose) {
                        fill(76, 175, 80);
                    } else {
                        fill(244, 67, 54);
                    }
                    ellipse(keypoint.position.x, keypoint.position.y, 12, 12);
                }
            }
            
            strokeWeight(3);
            if (pose.skeleton) {
                for (let skeleton of pose.skeleton) {
                    const [pointA, pointB] = skeleton;
                    if (pointA.score > 0.2 && pointB.score > 0.2) {
                        if (this.isCorrectPose) {
                            stroke(76, 175, 80, 200);
                        } else {
                            stroke(244, 67, 54, 200);
                        }
                        line(pointA.position.x, pointA.position.y, 
                             pointB.position.x, pointB.position.y);
                    }
                }
            }
            
            this.drawAccuracyOverlay();
        }
    }
    
    drawAccuracyOverlay() {
        fill(255, 255, 255, 200);
        rect(10, 10, 120, 30, 5);
        
        if (this.isCorrectPose) {
            fill(76, 175, 80);
        } else {
            fill(244, 67, 54);
        }
        
        textSize(16);
        textAlign(LEFT, CENTER);
        text(`${Math.round(this.poseAccuracy)}% Accurate`, 20, 25);
    }

    setTargetPose(poseIndex) {
        this.currentTargetPose = poseIndex;
        const pose = this.poseDefinitions[poseIndex];
        
        this.correctPoseFrames = 0;
        this.lastCorrectionTime = 0;
        
        console.log(`Target pose set to: ${pose.name} (${poseIndex})`);
        
        if (this.voiceEnabled) {
            this.speak(`Now practicing ${pose.name}. ${pose.description}`);
        }
    }

    getPoseAccuracy() {
        return this.poseAccuracy;
    }

    isCurrentPoseCorrect() {
        return this.isCorrectPose;
    }
    
    analyzePostureCorrections(keypoints, currentAngles, targetPose) {
        const corrections = [];
        const threshold = targetPose.threshold;
        
        for (const [angleName, targetAngle] of Object.entries(targetPose.targetAngles)) {
            if (currentAngles[angleName] !== undefined) {
                const difference = currentAngles[angleName] - targetAngle;
                const absDiff = Math.abs(difference);
                
                if (absDiff > threshold) {
                    corrections.push(this.generateSpecificCorrection(angleName, difference));
                }
            }
        }
        
        corrections.push(...this.checkBodyAlignment(keypoints));
        return corrections.slice(0, 2);
    }
    
    generateSpecificCorrection(angleName, difference) {
        const corrections = {
            leftArm: {
                positive: "Move your left arm closer to your body",
                negative: "Extend your left arm further out"
            },
            rightArm: {
                positive: "Move your right arm closer to your body", 
                negative: "Extend your right arm further out"
            },
            leftLeg: {
                positive: "Bend your left leg more",
                negative: "Straighten your left leg more"
            },
            rightLeg: {
                positive: "Bend your right leg more",
                negative: "Straighten your right leg more"
            },
            torso: {
                positive: "Lean forward more",
                negative: "Stand up straighter"
            }
        };
        
        if (corrections[angleName]) {
            return difference > 0 ? corrections[angleName].positive : corrections[angleName].negative;
        }
        
        return "Adjust your position";
    }
    
    checkBodyAlignment(keypoints) {
        const corrections = [];
        
        const leftShoulder = keypoints[5];
        const rightShoulder = keypoints[6];
        
        if (leftShoulder.score > 0.5 && rightShoulder.score > 0.5) {
            const shoulderDiff = Math.abs(leftShoulder.position.y - rightShoulder.position.y);
            if (shoulderDiff > 30) {
                corrections.push("Keep your shoulders level and aligned");
            }
        }
        
        const leftHip = keypoints[11];
        const rightHip = keypoints[12];
        
        if (leftHip.score > 0.5 && rightHip.score > 0.5) {
            const hipDiff = Math.abs(leftHip.position.y - rightHip.position.y);
            if (hipDiff > 25) {
                corrections.push("Align your hips evenly");
            }
        }
        
        return corrections;
    }
    
    provideVoiceCorrection() {
        if (!this.voiceEnabled) return;
        
        const now = Date.now();
        if (now - this.lastCorrectionTime < this.correctionCooldown) {
            return;
        }
        
        if (this.poseCorrections.length > 0) {
            this.speak(this.poseCorrections[0]);
            this.lastCorrectionTime = now;
        }
    }
    
    speak(message) {
        if (this.speechSynthesis && this.voiceEnabled) {
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.rate = 0.8;
            utterance.pitch = 1.0;
            utterance.volume = 0.7;
            this.speechSynthesis.speak(utterance);
        }
    }
    
    captureCorrectPose() {
        if (!this.video || !this.captureCanvas) return;
        
        this.captureContext.drawImage(this.video.elt, 0, 0, 640, 480);
        
        this.captureCanvas.toBlob((blob) => {
            this.saveToDataset(blob);
        }, 'image/jpeg', 0.9);
        
        this.showCaptureEffect();
    }
    
    saveToDataset(blob) {
        const poseName = this.poseDefinitions[this.currentTargetPose].name.toLowerCase().replace(/\s+/g, '_');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `training_${poseName}_${timestamp}.jpg`;
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log(`Captured training image: ${filename}`);
        this.showCaptureNotification(poseName);
    }
    
    showCaptureEffect() {
        const videoContainer = document.getElementById('videoContainer');
        if (videoContainer) {
            videoContainer.style.boxShadow = '0 0 30px #4CAF50';
            setTimeout(() => {
                videoContainer.style.boxShadow = '';
            }, 500);
        }
    }
    
    showCaptureNotification(poseName) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed; top: 20px; right: 20px;
            background: #4CAF50; color: white; padding: 15px 20px;
            border-radius: 8px; font-weight: bold; z-index: 1000;
            animation: slideIn 0.3s ease-out;
        `;
        notification.textContent = `📸 Training image captured: ${poseName}`;
        
        if (!document.getElementById('captureStyles')) {
            const style = document.createElement('style');
            style.id = 'captureStyles';
            style.textContent = `
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 3000);
    }
}