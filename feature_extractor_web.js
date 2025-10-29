// Enhanced feature extraction for web-based pose detection
class WebFeatureExtractor {
    constructor() {
        this.poseClasses = [
            'Pranamasana', 'Hastauttanasana', 'Hastapadasana', 
            'Ashwa Sanchalanasana', 'Dandasana', 'Ashtanga Namaskara',
            'Bhujangasana', 'Adho Mukha Svanasana', 'Padmasana', 'Tadasana'
        ];
        
        // PoseNet keypoint indices
        this.keypointIndices = {
            nose: 0, leftEye: 1, rightEye: 2, leftEar: 3, rightEar: 4,
            leftShoulder: 5, rightShoulder: 6, leftElbow: 7, rightElbow: 8,
            leftWrist: 9, rightWrist: 10, leftHip: 11, rightHip: 12,
            leftKnee: 13, rightKnee: 14, leftAnkle: 15, rightAnkle: 16
        };
    }

    extractFeatures(keypoints) {
        if (!keypoints || keypoints.length < 17) return null;

        const points = this.getValidPoints(keypoints);
        if (Object.keys(points).length < 10) return null;

        const features = [];

        // 1. Joint angles (8 features) - adapted for 17 keypoints
        features.push(
            this.calculateAngle(points.leftShoulder, points.leftElbow, points.leftWrist),
            this.calculateAngle(points.rightShoulder, points.rightElbow, points.rightWrist),
            this.calculateAngle(points.leftHip, points.leftKnee, points.leftAnkle),
            this.calculateAngle(points.rightHip, points.rightKnee, points.rightAnkle),
            this.calculateAngle(points.leftElbow, points.leftShoulder, points.rightShoulder),
            this.calculateAngle(points.rightElbow, points.rightShoulder, points.leftShoulder),
            this.calculateAngle(points.leftKnee, points.leftHip, points.rightHip),
            this.calculateAngle(points.rightKnee, points.rightHip, points.leftHip)
        );

        // 2. Distance measurements (6 features)
        features.push(
            this.calculateDistance(points.leftWrist, points.rightWrist),
            this.calculateDistance(points.leftAnkle, points.rightAnkle),
            this.calculateDistance(points.leftShoulder, points.rightShoulder),
            this.calculateDistance(points.leftHip, points.rightHip),
            this.calculateDistance(points.nose, points.leftHip),
            this.calculateDistance(points.nose, points.rightHip)
        );

        // 3. Body ratios and symmetry (6 features)
        features.push(
            Math.abs(points.leftShoulder.y - points.rightShoulder.y),
            Math.abs(points.leftHip.y - points.rightHip.y),
            Math.abs(points.leftAnkle.y - points.rightAnkle.y),
            this.safeDiv(this.calculateDistance(points.leftShoulder, points.rightShoulder),
                        this.calculateDistance(points.leftHip, points.rightHip)),
            this.safeDiv(this.calculateDistance(points.leftWrist, points.rightWrist),
                        this.calculateDistance(points.leftAnkle, points.rightAnkle)),
            Math.abs(points.nose.y - (points.leftHip.y + points.rightHip.y) / 2)
        );

        // 4. Pose orientation (4 features)
        features.push(
            points.nose.x - (points.leftShoulder.x + points.rightShoulder.x) / 2,
            (points.leftShoulder.y + points.rightShoulder.y) / 2 - (points.leftHip.y + points.rightHip.y) / 2,
            Math.abs(points.leftWrist.x - points.rightWrist.x),
            Math.abs(points.leftAnkle.x - points.rightAnkle.x)
        );

        // Clean features
        return features.map(f => isNaN(f) || !isFinite(f) ? 0 : f);
    }

    getValidPoints(keypoints) {
        const points = {};
        
        for (const [name, index] of Object.entries(this.keypointIndices)) {
            if (keypoints[index] && keypoints[index].score > 0.3) {
                points[name] = {
                    x: keypoints[index].position.x / 640, // Normalize to 0-1
                    y: keypoints[index].position.y / 480
                };
            }
        }
        
        return points;
    }

    calculateAngle(a, b, c) {
        if (!a || !b || !c) return 0;
        
        const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
        let angle = Math.abs(radians * 180.0 / Math.PI);
        
        if (angle > 180.0) {
            angle = 360 - angle;
        }
        
        return angle;
    }

    calculateDistance(a, b) {
        if (!a || !b) return 0;
        return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
    }

    safeDiv(a, b) {
        return b !== 0 ? a / b : 0;
    }

    normalizeFeatures(features) {
        // Simple min-max normalization
        const min = Math.min(...features);
        const max = Math.max(...features);
        const range = max - min;
        
        if (range === 0) return features;
        
        return features.map(f => (f - min) / range);
    }
}