// Yoga Session Timer with Pose-based Control
class YogaTimer {
    constructor() {
        this.startTime = 0;
        this.elapsedTime = 0;
        this.isRunning = false;
        this.isPaused = false;
        this.timerInterval = null;
        this.poseHoldTime = 0;
        this.targetHoldTime = 30; // 30 seconds default hold time
        this.lastCorrectPoseTime = 0;
        this.sessionStats = {
            totalTime: 0,
            correctPoseTime: 0,
            accuracy: 0
        };
    }

    start() {
        if (!this.isRunning) {
            this.startTime = Date.now() - this.elapsedTime;
            this.isRunning = true;
            this.isPaused = false;
            this.timerInterval = setInterval(() => this.updateTimer(), 100);
            console.log('Timer started');
        }
    }

    pause() {
        if (this.isRunning && !this.isPaused) {
            this.isPaused = true;
            clearInterval(this.timerInterval);
            console.log('Timer paused');
        } else if (this.isPaused) {
            this.isPaused = false;
            this.startTime = Date.now() - this.elapsedTime;
            this.timerInterval = setInterval(() => this.updateTimer(), 100);
            console.log('Timer resumed');
        }
    }

    reset() {
        this.isRunning = false;
        this.isPaused = false;
        this.elapsedTime = 0;
        this.poseHoldTime = 0;
        this.lastCorrectPoseTime = 0;
        this.sessionStats = {
            totalTime: 0,
            correctPoseTime: 0,
            accuracy: 0
        };
        
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        
        this.updateDisplay();
        console.log('Timer reset');
    }

    updateTimer() {
        if (this.isRunning && !this.isPaused) {
            this.elapsedTime = Date.now() - this.startTime;
            this.updateDisplay();
        }
    }

    updateWithPoseStatus(isCorrectPose, accuracy) {
        if (!this.isRunning || this.isPaused) return;

        const currentTime = Date.now();
        
        if (isCorrectPose) {
            if (this.lastCorrectPoseTime === 0) {
                this.lastCorrectPoseTime = currentTime;
            }
            
            this.poseHoldTime = (currentTime - this.lastCorrectPoseTime) / 1000;
            this.sessionStats.correctPoseTime += 0.1; // Update every 100ms
        } else {
            this.lastCorrectPoseTime = 0;
            this.poseHoldTime = 0;
        }

        // Update session stats
        this.sessionStats.totalTime = this.elapsedTime / 1000;
        this.sessionStats.accuracy = this.sessionStats.totalTime > 0 ? 
            (this.sessionStats.correctPoseTime / this.sessionStats.totalTime) * 100 : 0;

        this.updateDisplay();
    }

    updateDisplay() {
        const timerElement = document.getElementById('timerDisplay');
        if (timerElement) {
            const minutes = Math.floor(this.elapsedTime / 60000);
            const seconds = Math.floor((this.elapsedTime % 60000) / 1000);
            const formattedTime = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            timerElement.textContent = formattedTime;
            
            // Change color based on pose hold time
            if (this.poseHoldTime >= this.targetHoldTime) {
                timerElement.style.color = '#4CAF50'; // Green when target reached
            } else if (this.poseHoldTime > 0) {
                timerElement.style.color = '#FFC107'; // Yellow when holding pose
            } else {
                timerElement.style.color = '#f44336'; // Red when not holding pose
            }
        }

        // Update pose hold progress
        this.updatePoseHoldProgress();
    }

    updatePoseHoldProgress() {
        const progressBar = document.getElementById('poseHoldProgress');
        if (progressBar) {
            const progress = Math.min((this.poseHoldTime / this.targetHoldTime) * 100, 100);
            progressBar.style.width = progress + '%';
            
            if (progress >= 100) {
                progressBar.style.background = 'linear-gradient(90deg, #4CAF50, #8BC34A)';
            } else {
                progressBar.style.background = 'linear-gradient(90deg, #FFC107, #FF9800)';
            }
        }
    }

    getElapsedTime() {
        return this.elapsedTime;
    }

    getPoseHoldTime() {
        return this.poseHoldTime;
    }

    getSessionStats() {
        return {
            ...this.sessionStats,
            poseHoldTime: this.poseHoldTime,
            targetHoldTime: this.targetHoldTime
        };
    }

    setTargetHoldTime(seconds) {
        this.targetHoldTime = seconds;
    }

    isTimerRunning() {
        return this.isRunning && !this.isPaused;
    }

    formatTime(milliseconds) {
        const minutes = Math.floor(milliseconds / 60000);
        const seconds = Math.floor((milliseconds % 60000) / 1000);
        return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    // Audio feedback for pose completion
    playCompletionSound() {
        // Create audio context for sound feedback
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
        oscillator.frequency.setValueAtTime(1000, audioContext.currentTime + 0.1);
        
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.3);
    }

    // Check if pose hold target is reached
    checkPoseCompletion() {
        if (this.poseHoldTime >= this.targetHoldTime && this.lastCorrectPoseTime > 0) {
            this.playCompletionSound();
            return true;
        }
        return false;
    }
}