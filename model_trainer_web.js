// Web-based model training using ML5.js
class WebModelTrainer {
    constructor() {
        this.trainingData = [];
        this.model = null;
        this.isTraining = false;
        this.poseClasses = [
            'Pranamasana', 'Hastauttanasana', 'Hastapadasana', 
            'Ashwa Sanchalanasana', 'Dandasana', 'Ashtanga Namaskara',
            'Bhujangasana', 'Adho Mukha Svanasana', 'Padmasana', 'Tadasana'
        ];
    }

    addTrainingData(features, label) {
        if (features && features.length > 0) {
            this.trainingData.push({
                inputs: features,
                outputs: this.oneHotEncode(label)
            });
            console.log(`Added training sample for ${this.poseClasses[label]}`);
        }
    }

    oneHotEncode(classIndex) {
        const encoded = new Array(this.poseClasses.length).fill(0);
        encoded[classIndex] = 1;
        return encoded;
    }

    async trainModel() {
        if (this.trainingData.length < 50) {
            alert('Need at least 50 training samples. Current: ' + this.trainingData.length);
            return;
        }

        this.isTraining = true;
        document.getElementById('status').textContent = 'Training model...';

        // Create neural network
        const options = {
            inputs: 24, // Number of features
            outputs: this.poseClasses.length,
            task: 'classification',
            debug: true
        };

        this.model = ml5.neuralNetwork(options);

        // Add training data
        for (const sample of this.trainingData) {
            this.model.data.addData(sample.inputs, sample.outputs);
        }

        // Normalize data
        this.model.normalizeData();

        // Train the model
        const trainingOptions = {
            epochs: 100,
            batchSize: 12,
            learningRate: 0.2,
            validationSplit: 0.2
        };

        await this.model.train(trainingOptions, () => {
            console.log('Training complete!');
            this.isTraining = false;
            document.getElementById('status').textContent = 'Model trained successfully!';
        });
    }

    predict(features) {
        if (!this.model || this.isTraining) return null;

        return new Promise((resolve) => {
            this.model.classify(features, (error, results) => {
                if (error) {
                    console.error(error);
                    resolve(null);
                } else {
                    resolve(results);
                }
            });
        });
    }

    saveModel() {
        if (this.model) {
            this.model.save('yoga_pose_model');
            console.log('Model saved!');
        }
    }

    async loadModel() {
        const modelInfo = {
            model: 'yoga_pose_model/model.json',
            metadata: 'yoga_pose_model/model_meta.json',
            weights: 'yoga_pose_model/model.weights.bin'
        };

        this.model = ml5.neuralNetwork(modelInfo, () => {
            console.log('Model loaded!');
            document.getElementById('status').textContent = 'Model loaded successfully!';
        });
    }

    getTrainingStats() {
        const stats = {};
        for (const sample of this.trainingData) {
            const classIndex = sample.outputs.indexOf(1);
            const className = this.poseClasses[classIndex];
            stats[className] = (stats[className] || 0) + 1;
        }
        return stats;
    }

    clearTrainingData() {
        this.trainingData = [];
        console.log('Training data cleared');
    }
}