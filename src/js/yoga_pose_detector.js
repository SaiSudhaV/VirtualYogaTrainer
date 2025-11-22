// Enhanced Yoga Pose Detector with Real-time Classification
class YogaPoseDetector {
    constructor() {
        this.poseNet = null;
        this.video = null;
        this.poses = [];
        this.isModelReady = false;
        this.canvas = null;
        this.ctx = null;
        
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
            console.log('PoseNet model loaded successfully!');
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
            
            // Determine if pose is correct
            this.isCorrectPose = this.poseAccuracy >= 70; // 70% threshold for correct pose
            
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
        }

        // Update status display
        const statusElement = document.getElementById('statusDisplay');
        const videoContainer = document.getElementById('videoContainer');
        
        if (statusElement && videoContainer) {
            if (this.isCorrectPose) {
                statusElement.textContent = '✅ Correct Pose - Hold Position!';
                statusElement.className = 'status-display status-correct';
                videoContainer.className = 'video-container correct';
            } else {
                statusElement.textContent = '❌ Adjust Your Position';
                statusElement.className = 'status-display status-incorrect';
                videoContainer.className = 'video-container incorrect';
            }
        }
    }

    drawPose() {
        if (this.poses.length > 0) {
            const pose = this.poses[0].pose;
            
            // Draw keypoints
            fill(255, 0, 0);
            noStroke();
            for (let keypoint of pose.keypoints) {
                if (keypoint.score > 0.2) {
                    ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
                }
            }
            
            // Draw skeleton
            stroke(255, 255, 0);
            strokeWeight(2);
            if (pose.skeleton) {
                for (let skeleton of pose.skeleton) {
                    const [pointA, pointB] = skeleton;
                    if (pointA.score > 0.2 && pointB.score > 0.2) {
                        line(pointA.position.x, pointA.position.y, 
                             pointB.position.x, pointB.position.y);
                    }
                }
            }
        }
    }

    setTargetPose(poseIndex) {
        this.currentTargetPose = poseIndex;
        const pose = this.poseDefinitions[poseIndex];
        
        // Update UI
        const poseNameElement = document.getElementById('currentPoseName');
        const poseDescElement = document.getElementById('poseDescription');
        
        if (poseNameElement && poseDescElement) {
            poseNameElement.textContent = pose.name;
            poseDescElement.textContent = pose.description;
        }
    }

    getPoseAccuracy() {
        return this.poseAccuracy;
    }

    isCurrentPoseCorrect() {
        return this.isCorrectPose;
    }
}