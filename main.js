// Main application logic
let poseDetector;
let featureExtractor;
let modelTrainer;
let canvas;
let currentPoseClass = 0;
let isCollecting = false;

function setup() {
    canvas = createCanvas(640, 480);
    canvas.parent('canvas-container');
    
    // Initialize components
    poseDetector = new WebPoseDetector();
    featureExtractor = new WebFeatureExtractor();
    modelTrainer = new WebModelTrainer();
    
    // Initialize pose detection
    poseDetector.initialize();
}

function draw() {
    if (poseDetector.video) {
        // Display video
        image(poseDetector.video, 0, 0, 640, 480);
        
        // Draw pose landmarks
        poseDetector.drawKeypoints();
        poseDetector.drawSkeleton();
        
        // Extract and display features
        const keypoints = poseDetector.getKeypoints();
        if (keypoints) {
            const features = featureExtractor.extractFeatures(keypoints);
            if (features) {
                document.getElementById('feature-count').textContent = features.length;
                
                // Auto-collect data if collecting mode is on
                if (isCollecting && frameCount % 30 === 0) { // Collect every 30 frames
                    modelTrainer.addTrainingData(features, currentPoseClass);
                }
                
                // Predict pose if model is trained
                predictPose(features);
            }
        }
    }
}

async function predictPose(features) {
    if (modelTrainer.model && !modelTrainer.isTraining) {
        const results = await modelTrainer.predict(features);
        if (results && results.length > 0) {
            const topResult = results[0];
            document.getElementById('pose-name').textContent = topResult.label;
            document.getElementById('confidence').textContent = 
                Math.round(topResult.confidence * 100) + '%';
        }
    }
}

function startCamera() {
    if (poseDetector.video) {
        poseDetector.video.play();
        document.getElementById('status').textContent = 'Camera started';
    }
}

function stopCamera() {
    if (poseDetector.video) {
        poseDetector.video.pause();
        document.getElementById('status').textContent = 'Camera stopped';
    }
}

function collectData() {
    const poseName = prompt('Enter pose class (0-9):', '0');
    const poseIndex = parseInt(poseName);
    
    if (poseIndex >= 0 && poseIndex < 10) {
        currentPoseClass = poseIndex;
        isCollecting = !isCollecting;
        
        if (isCollecting) {
            document.getElementById('status').textContent = 
                `Collecting data for ${featureExtractor.poseClasses[poseIndex]}...`;
        } else {
            document.getElementById('status').textContent = 'Data collection stopped';
            console.log('Training stats:', modelTrainer.getTrainingStats());
        }
    }
}

async function trainModel() {
    if (modelTrainer.trainingData.length > 0) {
        await modelTrainer.trainModel();
    } else {
        alert('No training data available. Collect some data first!');
    }
}

// Keyboard shortcuts
function keyPressed() {
    if (key === 's' || key === 'S') {
        modelTrainer.saveModel();
    } else if (key === 'l' || key === 'L') {
        modelTrainer.loadModel();
    } else if (key === 'c' || key === 'C') {
        collectData();
    } else if (key === 't' || key === 'T') {
        trainModel();
    }
}