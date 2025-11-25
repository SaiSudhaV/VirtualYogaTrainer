// Enhanced Yoga Pose Detector with AI Correction and Voice Guidance
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
        this.correctionCooldown = 3000; // 3 seconds between corrections
        this.correctPoseFrames = 0;
        this.requiredCorrectFrames = 30; // 1 second at 30fps
        this.captureCanvas = null;
        this.captureContext = null;
        
        // Pose definitions with target angles and thresholds
        this.poseDefinitions = {
            0: { // Pranamasana (Prayer)
                name: "Pranamasana",
                description: "Stand with palms together in prayer position at chest level.",
                targetAngles: {
                    leftArm: 90,
                    rightArm: 90,
                    leftLeg: 180,
                    rightLeg: 180
                },
                threshold: 25
            },
            1: { // Hastauttanasana (Raised Arms)
                name: "Hastauttanasana",
                description: "Raise both arms overhead, palms facing each other.",
                targetAngles: {
                    leftArm: 180,
                    rightArm: 180,
                    leftLeg: 180,
                    rightLeg: 180
                },
                threshold: 30
            },
            2: { // Hastapadasana (Forward Bend)
                name: "Hastapadasana",
                description: "Bend forward, hands reaching toward feet.",
                targetAngles: {
                    leftArm: 45,
                    rightArm: 45,
                    torso: 45
                },
                threshold: 35
            },
            3: { // Ashwa Sanchalanasana (Low Lunge)
                name: "Ashwa Sanchalanasana",
                description: "Step back into low lunge, hands on ground.",
                targetAngles: {
                    leftLeg: 90,
                    rightLeg: 160
                },
                threshold: 40
            },
            4: { // Dandasana (Staff Pose)
                name: "Dandasana",
                description: "Sit with legs extended, spine straight, hands beside hips.",
                targetAngles: {
                    leftLeg: 180,
                    rightLeg: 180,
                    torso: 90
                },
                threshold: 25
            },
            5: { // Ashtanga Namaskara (Eight-Limbed)
                name: "Ashtanga Namaskara",
                description: "Lower knees, chest, and chin to ground.",
                targetAngles: {
                    leftArm: 45,
                    rightArm: 45
                },
                threshold: 35
            },
            6: { // Bhujangasana (Cobra)
                name: "Bhujangasana",
                description: "Lie on stomach, lift chest with arms support.",
                targetAngles: {
                    leftArm: 120,
                    rightArm: 120,
                    torso: 45
                },
                threshold: 30
            },
            7: { // Adho Mukha Svanasana (Downward Dog)
                name: "Adho Mukha Svanasana",
                description: "Form inverted V-shape, hands and feet on ground.",
                targetAngles: {
                    leftArm: 45,
                    rightArm: 45,
                    leftLeg: 45,
                    rightLeg: 45
                },
                threshold: 35
            },
            8: { // Padmasana (Lotus)
                name: "Padmasana",
                description: "Sit cross-legged, feet on opposite thighs.",
                targetAngles: {
                    leftLeg: 90,
                    rightLeg: 90,
                    torso: 90
                },
                threshold: 30
            },
            9: { // Tadasana (Mountain)
                name: "Tadasana",
                description: "Stand tall, arms at sides, body aligned.",
                targetAngles: {
                    leftArm: 180,
                    rightArm: 180,
                    leftLeg: 180,
                    rightLeg: 180
                },
                threshold: 20
            }
        };
        
        this.currentTargetPose = 0;
        this.poseAccuracy = 0;
        this.isCorrectPose = false;
        this.poseCorrections = [];
        this.voiceEnabled = true;
        
        // Initialize capture canvas for dataset storage
        this.initializeCaptureCanvas();
    }
    
    initializeCaptureCanvas() {
        this.captureCanvas = document.createElement('canvas');
        this.captureCanvas.width = 640;
        this.captureCanvas.height = 480;
        this.captureContext = this.captureCanvas.getContext('2d');
    }

    async initialize() {
        // Get optimal constraints for device
        const constraints = deviceCompatibility ? deviceCompatibility.cameraConstraints : {
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false
        };

        // Create video capture with device-specific settings
        this.video = createCapture(constraints.video);
        
        // Adjust size based on device
        if (deviceCompatibility && deviceCompatibility.isMobile) {
            this.video.size(480, 360);
        } else {
            this.video.size(640, 480);
        }
        
        this.video.hide();

        // Initialize PoseNet with device-optimized settings
        let options = {
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

        // Optimize for device performance
        if (deviceCompatibility) {
            if (deviceCompatibility.performanceLevel === 'low' || deviceCompatibility.isMobile) {
                options.imageScaleFactor = 0.5;
                options.outputStride = 32;
                options.inputResolution = 257;
                options.multiplier = 0.5;
            } else if (deviceCompatibility.performanceLevel === 'high') {
                options.imageScaleFactor = 0.2;
                options.outputStride = 8;
                options.inputResolution = 641;
                options.multiplier = 1.0;
            }
        }

        this.poseNet = ml5.poseNet(this.video, options, () => {
            this.isModelReady = true;
            console.log('AI Yoga Instructor loaded successfully!');
            
            // Welcome message
            if (this.speechSynthesis) {
                const welcome = new SpeechSynthesisUtterance("Welcome to your AI Yoga Instructor. I will guide you through correct postures.");
                welcome.rate = 0.8;
                this.speechSynthesis.speak(welcome);
            }
        });

        this.poseNet.on('pose', (results) => {
            this.poses = results;
            this.analyzePose();
        });
    }

    analyzePose() {
        if (this.poses.length > 0) {
            const pose = this.poses[0].pose;
            const keypoints = pose.keypoints;
            
            // Calculate current pose angles
            const currentAngles = this.calculatePoseAngles(keypoints);
            
            // Compare with target pose
            const targetPose = this.poseDefinitions[this.currentTargetPose];
            this.poseAccuracy = this.calculatePoseAccuracy(currentAngles, targetPose);
            
            // AI Pose Correction Analysis
            this.poseCorrections = this.analyzePostureCorrections(keypoints, currentAngles, targetPose);
            
            // Determine if pose is correct
            this.isCorrectPose = this.poseAccuracy >= 75; // 75% threshold for correct pose
            
            // Handle correct pose detection and dataset storage
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
            
            // Update UI
            this.updatePoseStatus();
        }
    }

    calculatePoseAngles(keypoints) {
        const angles = {};
        
        // Get key points
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

        // Calculate arm angles
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

        // Calculate leg angles
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

        // Calculate torso angle
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
        // Update accuracy display
        const accuracyElement = document.getElementById('accuracyValue');
        const accuracyBar = document.getElementById('accuracyBar');
        
        if (accuracyElement && accuracyBar) {
            accuracyElement.textContent = Math.round(this.poseAccuracy) + '%';
            accuracyBar.style.width = this.poseAccuracy + '%';
            
            // Color code accuracy bar
            if (this.poseAccuracy >= 75) {
                accuracyBar.style.background = 'linear-gradient(90deg, #4CAF50, #8BC34A)';
            } else if (this.poseAccuracy >= 50) {
                accuracyBar.style.background = 'linear-gradient(90deg, #FF9800, #FFC107)';
            } else {
                accuracyBar.style.background = 'linear-gradient(90deg, #f44336, #FF5722)';
            }
        }

        // Update status display with specific corrections
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
            
            // Draw keypoints with color coding
            noStroke();
            for (let keypoint of pose.keypoints) {
                if (keypoint.score > 0.2) {
                    // Color code based on pose correctness
                    if (this.isCorrectPose) {
                        fill(76, 175, 80); // Green for correct
                    } else {
                        fill(244, 67, 54); // Red for incorrect
                    }
                    ellipse(keypoint.position.x, keypoint.position.y, 12, 12);
                }
            }
            
            // Draw skeleton with color coding
            strokeWeight(3);
            if (pose.skeleton) {
                for (let skeleton of pose.skeleton) {
                    const [pointA, pointB] = skeleton;
                    if (pointA.score > 0.2 && pointB.score > 0.2) {
                        // Color code skeleton based on pose correctness
                        if (this.isCorrectPose) {
                            stroke(76, 175, 80, 200); // Green for correct
                        } else {
                            stroke(244, 67, 54, 200); // Red for incorrect
                        }
                        line(pointA.position.x, pointA.position.y, 
                             pointB.position.x, pointB.position.y);
                    }
                }
            }
            
            // Draw pose accuracy overlay
            this.drawAccuracyOverlay();
        }
    }
    
    drawAccuracyOverlay() {
        // Draw accuracy percentage on video
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
        
        // Reset correction counters
        this.correctPoseFrames = 0;
        this.lastCorrectionTime = 0;
        
        // Update UI
        const poseNameElement = document.getElementById('currentPoseName');
        const poseDescElement = document.getElementById('poseDescription');
        
        if (poseNameElement && poseDescElement) {
            poseNameElement.textContent = pose.name;
            poseDescElement.textContent = pose.description;
        }
        
        // Voice announcement for new pose
        if (this.voiceEnabled && this.speechSynthesis) {
            const announcement = new SpeechSynthesisUtterance(`Now practicing ${pose.name}. ${pose.description}`);
            announcement.rate = 0.8;
            this.speechSynthesis.speak(announcement);
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
        
        // Analyze each body part for specific corrections
        for (const [angleName, targetAngle] of Object.entries(targetPose.targetAngles)) {
            if (currentAngles[angleName] !== undefined) {
                const difference = currentAngles[angleName] - targetAngle;
                const absDiff = Math.abs(difference);
                
                if (absDiff > threshold) {
                    corrections.push(this.generateSpecificCorrection(angleName, difference, absDiff));
                }
            }
        }
        
        // Additional posture checks
        corrections.push(...this.checkBodyAlignment(keypoints));
        
        return corrections.slice(0, 2); // Limit to 2 most important corrections
    }
    
    generateSpecificCorrection(angleName, difference, absDiff) {
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
        
        // Check shoulder alignment
        const leftShoulder = keypoints[5];
        const rightShoulder = keypoints[6];
        
        if (leftShoulder.score > 0.5 && rightShoulder.score > 0.5) {
            const shoulderDiff = Math.abs(leftShoulder.position.y - rightShoulder.position.y);
            if (shoulderDiff > 30) {
                corrections.push("Keep your shoulders level and aligned");
            }
        }
        
        // Check hip alignment
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
            return; // Too soon for another correction
        }
        
        if (this.poseCorrections.length > 0 && this.speechSynthesis) {
            const correction = this.poseCorrections[0];
            const utterance = new SpeechSynthesisUtterance(correction);
            utterance.rate = 0.8;
            utterance.pitch = 1.0;
            utterance.volume = 0.7;
            
            this.speechSynthesis.speak(utterance);
            this.lastCorrectionTime = now;
        }
    }
    
    captureCorrectPose() {
        if (!this.video || !this.captureCanvas) return;
        
        // Capture current frame
        this.captureContext.drawImage(this.video.elt, 0, 0, 640, 480);
        
        // Convert to blob and save
        this.captureCanvas.toBlob((blob) => {
            this.saveToDataset(blob);
        }, 'image/jpeg', 0.9);
        
        // Visual feedback for capture
        this.showCaptureEffect();
    }
    
    saveToDataset(blob) {
        const poseName = this.poseDefinitions[this.currentTargetPose].name.toLowerCase().replace(/\s+/g, '_');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `correct_${poseName}_${timestamp}.jpg`;
        
        // Create download link (browser-based storage)
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log(`Captured correct pose: ${filename}`);
        
        // Show notification
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
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            font-weight: bold;
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
        `;
        notification.textContent = `📸 Captured correct ${poseName} pose!`;
        
        // Add animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
            style.remove();
        }, 3000);
    }
}