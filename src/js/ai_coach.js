// AI Yoga Coach - Intelligent Coaching and Progress Tracking
class AIYogaCoach {
    constructor() {
        this.userProfile = this.loadUserProfile();
        this.sessionHistory = this.loadSessionHistory();
        this.personalizedRecommendations = [];
        this.adaptiveDifficulty = 'beginner';
        this.progressMetrics = {
            accuracy: [],
            holdTime: [],
            consistency: [],
            improvement: 0
        };
        
        // AI Learning System
        this.learningModel = {
            userPatterns: {},
            commonMistakes: {},
            improvementAreas: [],
            strengths: []
        };
        
        // Motivational System
        this.motivationalPhrases = {
            encouragement: [
                "Great progress! Keep holding that pose!",
                "You're improving with each session!",
                "Excellent form! Feel the strength building!",
                "Perfect alignment! You're mastering this pose!",
                "Beautiful pose! Your dedication is showing!"
            ],
            correction: [
                "Small adjustment needed - you're almost there!",
                "Let's refine this pose together",
                "Focus on your breathing while adjusting",
                "Take your time to find the perfect alignment",
                "Remember, progress over perfection!"
            ],
            achievement: [
                "Congratulations! You've mastered this pose!",
                "Amazing! You've reached a new milestone!",
                "Your consistency is paying off beautifully!",
                "What an achievement! You should be proud!",
                "Incredible progress! You're becoming stronger!"
            ]
        };
        
        this.initializeAI();
    }
    
    initializeAI() {
        console.log('AI Yoga Coach initialized');
        this.analyzeUserHistory();
        this.generatePersonalizedPlan();
    }
    
    // Intelligent Progress Tracking
    trackProgress(poseIndex, accuracy, holdTime, corrections) {
        const session = {
            timestamp: Date.now(),
            pose: poseIndex,
            accuracy: accuracy,
            holdTime: holdTime,
            corrections: corrections,
            date: new Date().toISOString().split('T')[0]
        };
        
        this.sessionHistory.push(session);
        this.updateProgressMetrics(session);
        this.analyzePerformancePatterns(session);
        this.saveSessionHistory();
        
        return this.generateSessionFeedback(session);
    }
    
    updateProgressMetrics(session) {
        this.progressMetrics.accuracy.push(session.accuracy);
        this.progressMetrics.holdTime.push(session.holdTime);
        
        // Keep only last 20 sessions for analysis
        if (this.progressMetrics.accuracy.length > 20) {
            this.progressMetrics.accuracy.shift();
            this.progressMetrics.holdTime.shift();
        }
        
        // Calculate improvement trend
        if (this.progressMetrics.accuracy.length >= 5) {
            const recent = this.progressMetrics.accuracy.slice(-5);
            const older = this.progressMetrics.accuracy.slice(-10, -5);
            
            if (older.length > 0) {
                const recentAvg = recent.reduce((a, b) => a + b) / recent.length;
                const olderAvg = older.reduce((a, b) => a + b) / older.length;
                this.progressMetrics.improvement = ((recentAvg - olderAvg) / olderAvg) * 100;
            }
        }
    }
    
    // AI Pattern Recognition
    analyzePerformancePatterns(session) {
        const poseKey = `pose_${session.pose}`;
        
        if (!this.learningModel.userPatterns[poseKey]) {
            this.learningModel.userPatterns[poseKey] = {
                attempts: 0,
                averageAccuracy: 0,
                commonCorrections: {},
                improvementRate: 0,
                difficulty: 'unknown'
            };
        }
        
        const pattern = this.learningModel.userPatterns[poseKey];
        pattern.attempts++;
        pattern.averageAccuracy = (pattern.averageAccuracy * (pattern.attempts - 1) + session.accuracy) / pattern.attempts;
        
        // Track common corrections
        session.corrections.forEach(correction => {
            pattern.commonCorrections[correction] = (pattern.commonCorrections[correction] || 0) + 1;
        });
        
        // Determine difficulty level for user
        if (pattern.averageAccuracy > 85) {
            pattern.difficulty = 'easy';
        } else if (pattern.averageAccuracy > 70) {
            pattern.difficulty = 'moderate';
        } else {
            pattern.difficulty = 'challenging';
        }
        
        this.updateAdaptiveDifficulty();
    }
    
    updateAdaptiveDifficulty() {
        const patterns = Object.values(this.learningModel.userPatterns);
        if (patterns.length === 0) return;
        
        const avgAccuracy = patterns.reduce((sum, p) => sum + p.averageAccuracy, 0) / patterns.length;
        
        if (avgAccuracy > 85) {
            this.adaptiveDifficulty = 'advanced';
        } else if (avgAccuracy > 70) {
            this.adaptiveDifficulty = 'intermediate';
        } else {
            this.adaptiveDifficulty = 'beginner';
        }
        
        console.log(`AI adapted difficulty level to: ${this.adaptiveDifficulty}`);
    }
    
    // Intelligent Recommendations
    generatePersonalizedRecommendations() {
        const recommendations = [];
        
        // Analyze weak areas
        const weakPoses = this.identifyWeakAreas();
        const strongPoses = this.identifyStrengths();
        
        if (weakPoses.length > 0) {
            recommendations.push({
                type: 'improvement',
                title: 'Focus Areas',
                message: `Practice these poses more: ${weakPoses.map(p => p.name).join(', ')}`,
                poses: weakPoses.map(p => p.index),
                priority: 'high'
            });
        }
        
        if (strongPoses.length > 0) {
            recommendations.push({
                type: 'strength',
                title: 'Your Strengths',
                message: `You excel at: ${strongPoses.map(p => p.name).join(', ')}`,
                poses: strongPoses.map(p => p.index),
                priority: 'medium'
            });
        }
        
        // Time-based recommendations
        const timeRecommendation = this.generateTimeRecommendation();
        if (timeRecommendation) {
            recommendations.push(timeRecommendation);
        }
        
        // Consistency recommendations
        const consistencyRecommendation = this.generateConsistencyRecommendation();
        if (consistencyRecommendation) {
            recommendations.push(consistencyRecommendation);
        }
        
        this.personalizedRecommendations = recommendations;
        return recommendations;
    }
    
    identifyWeakAreas() {
        const poseNames = [
            "Pranamasana", "Hastauttanasana", "Hastapadasana", 
            "Ashwa Sanchalanasana", "Dandasana", "Ashtanga Namaskara",
            "Bhujangasana", "Adho Mukha Svanasana", "Padmasana", "Tadasana"
        ];
        
        const weakPoses = [];
        
        Object.entries(this.learningModel.userPatterns).forEach(([poseKey, pattern]) => {
            const poseIndex = parseInt(poseKey.split('_')[1]);
            if (pattern.averageAccuracy < 70 && pattern.attempts >= 3) {
                weakPoses.push({
                    index: poseIndex,
                    name: poseNames[poseIndex],
                    accuracy: pattern.averageAccuracy
                });
            }
        });
        
        return weakPoses.sort((a, b) => a.accuracy - b.accuracy).slice(0, 3);
    }
    
    identifyStrengths() {
        const poseNames = [
            "Pranamasana", "Hastauttanasana", "Hastapadasana", 
            "Ashwa Sanchalanasana", "Dandasana", "Ashtanga Namaskara",
            "Bhujangasana", "Adho Mukha Svanasana", "Padmasana", "Tadasana"
        ];
        
        const strongPoses = [];
        
        Object.entries(this.learningModel.userPatterns).forEach(([poseKey, pattern]) => {
            const poseIndex = parseInt(poseKey.split('_')[1]);
            if (pattern.averageAccuracy > 85 && pattern.attempts >= 3) {
                strongPoses.push({
                    index: poseIndex,
                    name: poseNames[poseIndex],
                    accuracy: pattern.averageAccuracy
                });
            }
        });
        
        return strongPoses.sort((a, b) => b.accuracy - a.accuracy).slice(0, 3);
    }
    
    generateTimeRecommendation() {
        const recentSessions = this.sessionHistory.slice(-10);
        if (recentSessions.length < 5) return null;
        
        const avgHoldTime = recentSessions.reduce((sum, s) => sum + s.holdTime, 0) / recentSessions.length;
        
        if (avgHoldTime < 15) {
            return {
                type: 'time',
                title: 'Hold Time Challenge',
                message: 'Try to hold poses for longer periods to build strength and stability',
                priority: 'medium'
            };
        } else if (avgHoldTime > 45) {
            return {
                type: 'time',
                title: 'Excellent Endurance',
                message: 'Your pose holding ability is excellent! Consider trying more challenging variations',
                priority: 'low'
            };
        }
        
        return null;
    }
    
    generateConsistencyRecommendation() {
        const last7Days = this.getSessionsInLastDays(7);
        const last30Days = this.getSessionsInLastDays(30);
        
        if (last7Days.length < 3) {
            return {
                type: 'consistency',
                title: 'Practice More Regularly',
                message: 'Try to practice at least 3 times per week for better results',
                priority: 'high'
            };
        } else if (last30Days.length > 20) {
            return {
                type: 'consistency',
                title: 'Amazing Dedication!',
                message: 'Your consistent practice is impressive! Keep up the great work!',
                priority: 'low'
            };
        }
        
        return null;
    }
    
    // Smart Motivation System
    getMotivationalMessage(context, accuracy, improvement) {
        let category = 'encouragement';
        
        if (accuracy > 90) {
            category = 'achievement';
        } else if (accuracy < 60) {
            category = 'correction';
        }
        
        const messages = this.motivationalPhrases[category];
        const randomMessage = messages[Math.floor(Math.random() * messages.length)];
        
        // Add personalized touch based on improvement
        if (improvement > 10) {
            return `${randomMessage} Your accuracy improved by ${improvement.toFixed(1)}%!`;
        } else if (improvement < -5) {
            return `${randomMessage} Remember, every expert was once a beginner.`;
        }
        
        return randomMessage;
    }
    
    // Session Analysis and Feedback
    generateSessionFeedback(session) {
        const feedback = {
            overall: this.getOverallFeedback(session),
            specific: this.getSpecificFeedback(session),
            recommendations: this.getSessionRecommendations(session),
            motivation: this.getMotivationalMessage('session', session.accuracy, this.progressMetrics.improvement)
        };
        
        return feedback;
    }
    
    getOverallFeedback(session) {
        if (session.accuracy > 90) {
            return "Excellent session! Your form was nearly perfect.";
        } else if (session.accuracy > 75) {
            return "Great session! You're showing good progress.";
        } else if (session.accuracy > 60) {
            return "Good effort! Keep practicing to improve your form.";
        } else {
            return "Keep practicing! Every session makes you stronger.";
        }
    }
    
    getSpecificFeedback(session) {
        const feedback = [];
        
        if (session.holdTime > 30) {
            feedback.push("Impressive hold time! Your endurance is building.");
        } else if (session.holdTime < 10) {
            feedback.push("Try to hold poses longer for better strength building.");
        }
        
        if (session.corrections.length === 0) {
            feedback.push("Perfect form with no corrections needed!");
        } else if (session.corrections.length > 5) {
            feedback.push("Focus on one correction at a time for better results.");
        }
        
        return feedback;
    }
    
    getSessionRecommendations(session) {
        const recommendations = [];
        
        // Based on accuracy
        if (session.accuracy < 70) {
            recommendations.push("Practice this pose more frequently to improve muscle memory");
        }
        
        // Based on corrections
        const mostCommonCorrection = this.getMostCommonCorrection(session.pose);
        if (mostCommonCorrection) {
            recommendations.push(`Focus on: ${mostCommonCorrection}`);
        }
        
        return recommendations;
    }
    
    getMostCommonCorrection(poseIndex) {
        const poseKey = `pose_${poseIndex}`;
        const pattern = this.learningModel.userPatterns[poseKey];
        
        if (!pattern || !pattern.commonCorrections) return null;
        
        let maxCount = 0;
        let mostCommon = null;
        
        Object.entries(pattern.commonCorrections).forEach(([correction, count]) => {
            if (count > maxCount) {
                maxCount = count;
                mostCommon = correction;
            }
        });
        
        return mostCommon;
    }
    
    // Progress Analytics
    getProgressAnalytics() {
        return {
            totalSessions: this.sessionHistory.length,
            averageAccuracy: this.calculateAverageAccuracy(),
            improvementTrend: this.progressMetrics.improvement,
            consistencyScore: this.calculateConsistencyScore(),
            strongestPoses: this.identifyStrengths(),
            improvementAreas: this.identifyWeakAreas(),
            currentLevel: this.adaptiveDifficulty,
            recommendations: this.personalizedRecommendations
        };
    }
    
    calculateAverageAccuracy() {
        if (this.sessionHistory.length === 0) return 0;
        
        const totalAccuracy = this.sessionHistory.reduce((sum, session) => sum + session.accuracy, 0);
        return totalAccuracy / this.sessionHistory.length;
    }
    
    calculateConsistencyScore() {
        const last30Days = this.getSessionsInLastDays(30);
        const uniqueDays = new Set(last30Days.map(s => s.date)).size;
        
        return Math.min(100, (uniqueDays / 30) * 100);
    }
    
    getSessionsInLastDays(days) {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - days);
        
        return this.sessionHistory.filter(session => {
            const sessionDate = new Date(session.timestamp);
            return sessionDate >= cutoffDate;
        });
    }
    
    // Data Persistence
    loadUserProfile() {
        const saved = localStorage.getItem('aiYogaCoach_userProfile');
        return saved ? JSON.parse(saved) : {
            name: 'Yogi',
            level: 'beginner',
            goals: [],
            preferences: {}
        };
    }
    
    loadSessionHistory() {
        const saved = localStorage.getItem('aiYogaCoach_sessionHistory');
        return saved ? JSON.parse(saved) : [];
    }
    
    saveSessionHistory() {
        localStorage.setItem('aiYogaCoach_sessionHistory', JSON.stringify(this.sessionHistory));
    }
    
    saveUserProfile() {
        localStorage.setItem('aiYogaCoach_userProfile', JSON.stringify(this.userProfile));
    }
    
    // Public API
    analyzeUserHistory() {
        this.sessionHistory.forEach(session => {
            this.analyzePerformancePatterns(session);
        });
    }
    
    generatePersonalizedPlan() {
        return this.generatePersonalizedRecommendations();
    }
    
    getAdaptiveDifficulty() {
        return this.adaptiveDifficulty;
    }
    
    reset() {
        this.sessionHistory = [];
        this.learningModel = {
            userPatterns: {},
            commonMistakes: {},
            improvementAreas: [],
            strengths: []
        };
        this.saveSessionHistory();
        console.log('AI Coach data reset');
    }
}