// Web-based pose detection using ML5.js PoseNet
class WebPoseDetector {
    constructor() {
        this.poseNet = null;
        this.video = null;
        this.poses = [];
        this.isModelReady = false;
    }

    async initialize() {
        // Create video capture
        this.video = createCapture(VIDEO);
        this.video.size(640, 480);
        this.video.hide();

        // Initialize PoseNet
        const options = {
            architecture: 'MobileNetV1',
            imageScaleFactor: 0.3,
            outputStride: 16,
            flipHorizontal: false,
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
            document.getElementById('status').textContent = 'PoseNet model loaded successfully!';
        });

        this.poseNet.on('pose', (results) => {
            this.poses = results;
        });
    }

    getPoses() {
        return this.poses;
    }

    drawKeypoints() {
        if (this.poses.length > 0) {
            const pose = this.poses[0].pose;
            
            // Draw keypoints
            for (let keypoint of pose.keypoints) {
                if (keypoint.score > 0.2) {
                    fill(255, 0, 0);
                    noStroke();
                    ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
                }
            }
        }
    }

    drawSkeleton() {
        if (this.poses.length > 0) {
            const skeleton = this.poses[0].skeleton;
            
            // Draw skeleton connections
            for (let connection of skeleton) {
                const [pointA, pointB] = connection;
                stroke(255, 255, 0);
                strokeWeight(2);
                line(pointA.position.x, pointA.position.y, 
                     pointB.position.x, pointB.position.y);
            }
        }
    }

    getKeypoints() {
        if (this.poses.length > 0) {
            return this.poses[0].pose.keypoints;
        }
        return null;
    }
}