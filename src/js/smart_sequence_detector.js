// Smart Sequence Detector - AI-powered yoga sequence recognition and guidance
class SmartSequenceDetector {
    constructor() {
        this.sequences = {
            'sun_salutation': {
                name: 'Surya Namaskara (Sun Salutation)',
                poses: [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0], // Complete sun salutation
                transitions: this.defineSunSalutationTransitions(),
                difficulty: 'intermediate',
                duration: 300, // 5 minutes
                benefits: ['Full body workout', 'Improves flexibility', 'Builds strength', 'Enhances focus']
            },
            'beginner_flow': {
                name: 'Beginner Flow',
                poses: [9, 0, 1, 9, 8], // Mountain -> Prayer -> Raised Arms -> Mountain -> Lotus
                transitions: this.defineBeginnerTransitions(),
                difficulty: 'beginner',
                duration: 180, // 3 minutes
                benefits: ['Gentle introduction', 'Basic alignment', 'Breathing awareness']
            },
            'strength_builder': {
                name: 'Strength Building Sequence',
                poses: [4, 6, 7, 5, 4], // Staff -> Cobra -> Downward Dog -> Eight-limbed -> Staff
                transitions: this.defineStrengthTransitions(),
                difficulty: 'advanced',
                duration: 240, // 4 minutes
                benefits: ['Core strength', 'Upper body power', 'Stability']
            }
        };
        
        this.currentSequence = null;
        this.currentPoseIndex = 0;
        this.sequenceProgress = 0;
        this.isSequenceActive = false;
        this.transitionGuidance = null;
        this.adaptiveSequencing = true;
        
        // AI Sequence Learning
        this.userSequencePatterns = this.loadUserPatterns();
        this.customSequences = this.loadCustomSequences();
        
        // Transition detection
        this.transitionDetector = {
            isTransitioning: false,
            transitionStart: 0,
            expectedTransition: null,
            transitionAccuracy: 0
        };
        
        console.log('Smart Sequence Detector initialized');
    }
    
    defineSunSalutationTransitions() {
        return {
            '0->1': {
                name: 'Prayer to Raised Arms',
                instruction: 'Inhale, sweep arms overhead',
                duration: 3,
                keyMovements: ['arms_up', 'chest_open']
            },
            '1->2': {
                name: 'Raised Arms to Forward Fold',
                instruction: 'Exhale, hinge at hips, fold forward',
                duration: 4,
                keyMovements: ['forward_fold', 'hands_down']
            },
            '2->3': {
                name: 'Forward Fold to Low Lunge',
                instruction: 'Inhale, step right foot back',
                duration: 3,
                keyMovements: ['step_back', 'lunge_position']
            },
            '3->4': {
                name: 'Low Lunge to Staff',
                instruction: 'Step left foot back, sit down',
                duration: 4,
                keyMovements: ['both_feet_back', 'sit_down']
            },
            '4->5': {
                name: 'Staff to Eight-Limbed',
                instruction: 'Lower knees, chest, chin',
                duration: 3,
                keyMovements: ['knees_down', 'chest_down']
            },
            '5->6': {
                name: 'Eight-Limbed to Cobra',
                instruction: 'Slide forward, lift chest',
                duration: 3,
                keyMovements: ['chest_up', 'arms_straight']
            },
            '6->7': {
                name: 'Cobra to Downward Dog',
                instruction: 'Tuck toes, lift hips up',
                duration: 4,
                keyMovements: ['hips_up', 'inverted_v']
            }
        };
    }
    
    defineBeginnerTransitions() {
        return {
            '9->0': {
                name: 'Mountain to Prayer',
                instruction: 'Bring palms together at heart center',
                duration: 2,
                keyMovements: ['palms_together']
            },
            '0->1': {
                name: 'Prayer to Raised Arms',
                instruction: 'Inhale, sweep arms overhead',
                duration: 3,
                keyMovements: ['arms_up']
            },
            '1->9': {
                name: 'Raised Arms to Mountain',
                instruction: 'Exhale, lower arms to sides',
                duration: 2,
                keyMovements: ['arms_down']
            },
            '9->8': {
                name: 'Mountain to Lotus',
                instruction: 'Sit down cross-legged',
                duration: 4,
                keyMovements: ['sit_down', 'cross_legs']
            }
        };
    }
    
    defineStrengthTransitions() {
        return {
            '4->6': {
                name: 'Staff to Cobra',
                instruction: 'Lie down, lift chest with arms',
                duration: 4,
                keyMovements: ['lie_down', 'chest_up']
            },
            '6->7': {
                name: 'Cobra to Downward Dog',
                instruction: 'Tuck toes, lift hips up',
                duration: 4,
                keyMovements: ['hips_up', 'inverted_v']
            },
            '7->5': {
                name: 'Downward Dog to Eight-Limbed',
                instruction: 'Lower knees, chest, chin',
                duration: 3,
                keyMovements: ['knees_down', 'chest_down']
            },
            '5->4': {
                name: 'Eight-Limbed to Staff',
                instruction: 'Sit up, legs extended',
                duration: 4,
                keyMovements: ['sit_up', 'legs_straight']
            }
        };
    }
    
    // Start a sequence
    startSequence(sequenceKey, aiCoach = null) {
        if (!this.sequences[sequenceKey]) {
            console.error('Sequence not found:', sequenceKey);
            return false;
        }
        
        this.currentSequence = this.sequences[sequenceKey];
        this.currentPoseIndex = 0;
        this.sequenceProgress = 0;
        this.isSequenceActive = true;
        
        // Adapt sequence based on user level if AI coach is available\n        if (aiCoach && this.adaptiveSequencing) {\n            this.adaptSequenceToUser(aiCoach);\n        }\n        \n        console.log(`Started sequence: ${this.currentSequence.name}`);\n        this.announceSequenceStart();\n        \n        return true;\n    }\n    \n    adaptSequenceToUser(aiCoach) {\n        const userLevel = aiCoach.getAdaptiveDifficulty();\n        const analytics = aiCoach.getProgressAnalytics();\n        \n        // Modify sequence based on user's weak areas\n        if (analytics.improvementAreas.length > 0) {\n            const weakPoses = analytics.improvementAreas.map(area => area.index);\n            \n            // Add extra time for weak poses\n            this.currentSequence.adaptations = {\n                extraHoldTime: weakPoses,\n                skipAdvanced: userLevel === 'beginner'\n            };\n        }\n        \n        console.log(`Adapted sequence for ${userLevel} level user`);\n    }\n    \n    // Update sequence progress\n    updateSequence(currentPose, poseAccuracy, isCorrect) {\n        if (!this.isSequenceActive || !this.currentSequence) return null;\n        \n        const expectedPose = this.currentSequence.poses[this.currentPoseIndex];\n        const guidance = {\n            currentStep: this.currentPoseIndex + 1,\n            totalSteps: this.currentSequence.poses.length,\n            currentPose: expectedPose,\n            nextPose: this.getNextPose(),\n            progress: (this.currentPoseIndex / this.currentSequence.poses.length) * 100,\n            instruction: this.getCurrentInstruction(),\n            isOnTrack: currentPose === expectedPose\n        };\n        \n        // Check if user is performing correct pose\n        if (currentPose === expectedPose && isCorrect) {\n            guidance.status = 'correct';\n            guidance.message = 'Perfect! Hold this pose.';\n            \n            // Check for pose completion and transition\n            if (this.shouldTransitionToNext()) {\n                this.initiateTransition();\n            }\n        } else if (currentPose === expectedPose && !isCorrect) {\n            guidance.status = 'adjust';\n            guidance.message = 'Right pose, refine your form.';\n        } else {\n            guidance.status = 'wrong_pose';\n            guidance.message = `Move to ${this.getPoseName(expectedPose)}`;\n        }\n        \n        // Handle transitions\n        if (this.transitionDetector.isTransitioning) {\n            guidance.transition = this.updateTransition(currentPose, poseAccuracy);\n        }\n        \n        return guidance;\n    }\n    \n    shouldTransitionToNext() {\n        // Logic to determine when to move to next pose\n        // This could be time-based, accuracy-based, or user-triggered\n        return false; // Placeholder - implement based on requirements\n    }\n    \n    initiateTransition() {\n        if (this.currentPoseIndex >= this.currentSequence.poses.length - 1) {\n            this.completeSequence();\n            return;\n        }\n        \n        const currentPose = this.currentSequence.poses[this.currentPoseIndex];\n        const nextPose = this.currentSequence.poses[this.currentPoseIndex + 1];\n        const transitionKey = `${currentPose}->${nextPose}`;\n        \n        this.transitionDetector.isTransitioning = true;\n        this.transitionDetector.transitionStart = Date.now();\n        this.transitionDetector.expectedTransition = this.currentSequence.transitions[transitionKey];\n        \n        if (this.transitionDetector.expectedTransition) {\n            this.announceTransition(this.transitionDetector.expectedTransition);\n        }\n        \n        console.log(`Transitioning from pose ${currentPose} to ${nextPose}`);\n    }\n    \n    updateTransition(currentPose, poseAccuracy) {\n        const transition = this.transitionDetector.expectedTransition;\n        if (!transition) return null;\n        \n        const elapsed = (Date.now() - this.transitionDetector.transitionStart) / 1000;\n        const progress = Math.min(100, (elapsed / transition.duration) * 100);\n        \n        // Check if transition is complete\n        const nextPose = this.currentSequence.poses[this.currentPoseIndex + 1];\n        if (currentPose === nextPose && poseAccuracy > 70) {\n            this.completeTransition();\n            return {\n                status: 'completed',\n                message: 'Transition complete!'\n            };\n        }\n        \n        return {\n            status: 'in_progress',\n            progress: progress,\n            instruction: transition.instruction,\n            timeRemaining: Math.max(0, transition.duration - elapsed)\n        };\n    }\n    \n    completeTransition() {\n        this.transitionDetector.isTransitioning = false;\n        this.currentPoseIndex++;\n        this.sequenceProgress = (this.currentPoseIndex / this.currentSequence.poses.length) * 100;\n        \n        console.log(`Moved to pose ${this.currentPoseIndex + 1} of ${this.currentSequence.poses.length}`);\n    }\n    \n    completeSequence() {\n        console.log(`Sequence completed: ${this.currentSequence.name}`);\n        \n        const completionData = {\n            sequence: this.currentSequence.name,\n            completedAt: new Date(),\n            duration: this.calculateSequenceDuration(),\n            accuracy: this.calculateSequenceAccuracy()\n        };\n        \n        this.saveSequenceCompletion(completionData);\n        this.announceSequenceCompletion(completionData);\n        \n        this.isSequenceActive = false;\n        this.currentSequence = null;\n        this.currentPoseIndex = 0;\n        \n        return completionData;\n    }\n    \n    // AI-powered sequence generation\n    generatePersonalizedSequence(aiCoach, duration = 300, focus = 'balanced') {\n        const analytics = aiCoach.getProgressAnalytics();\n        const userLevel = aiCoach.getAdaptiveDifficulty();\n        \n        let poses = [];\n        \n        // Base poses for different focuses\n        const focusMap = {\n            'strength': [4, 5, 6, 7], // Staff, Eight-limbed, Cobra, Downward Dog\n            'flexibility': [1, 2, 3, 8], // Raised Arms, Forward Fold, Low Lunge, Lotus\n            'balance': [9, 0, 3, 4], // Mountain, Prayer, Low Lunge, Staff\n            'balanced': [9, 0, 1, 2, 7, 8] // Mix of all\n        };\n        \n        let basePoses = focusMap[focus] || focusMap['balanced'];\n        \n        // Add user's strong poses for confidence\n        if (analytics.strongestPoses.length > 0) {\n            const strongPoses = analytics.strongestPoses.slice(0, 2).map(p => p.index);\n            basePoses = [...basePoses, ...strongPoses];\n        }\n        \n        // Adjust for user level\n        if (userLevel === 'beginner') {\n            basePoses = basePoses.filter(pose => [0, 1, 8, 9].includes(pose)); // Easier poses\n        } else if (userLevel === 'advanced') {\n            basePoses = [...basePoses, 3, 5, 7]; // Add challenging poses\n        }\n        \n        // Create sequence with smooth transitions\n        poses = this.createSmoothSequence(basePoses, duration);\n        \n        const customSequence = {\n            name: `Personalized ${focus.charAt(0).toUpperCase() + focus.slice(1)} Flow`,\n            poses: poses,\n            transitions: this.generateTransitions(poses),\n            difficulty: userLevel,\n            duration: duration,\n            benefits: this.generateBenefits(focus),\n            isPersonalized: true\n        };\n        \n        // Save custom sequence\n        const sequenceId = `custom_${Date.now()}`;\n        this.customSequences[sequenceId] = customSequence;\n        this.saveCustomSequences();\n        \n        console.log(`Generated personalized sequence: ${customSequence.name}`);\n        return { id: sequenceId, sequence: customSequence };\n    }\n    \n    createSmoothSequence(basePoses, duration) {\n        // Algorithm to create smooth transitions between poses\n        const sequence = [];\n        const poseHoldTime = Math.max(15, duration / (basePoses.length * 2)); // Minimum 15 seconds per pose\n        \n        // Start with mountain pose for grounding\n        if (!basePoses.includes(9)) {\n            sequence.push(9);\n        }\n        \n        // Add base poses with transition logic\n        for (let i = 0; i < basePoses.length; i++) {\n            const currentPose = basePoses[i];\n            sequence.push(currentPose);\n            \n            // Add transition poses if needed\n            if (i < basePoses.length - 1) {\n                const nextPose = basePoses[i + 1];\n                const transitionPose = this.findTransitionPose(currentPose, nextPose);\n                if (transitionPose !== null && !sequence.includes(transitionPose)) {\n                    sequence.push(transitionPose);\n                }\n            }\n        }\n        \n        // End with relaxation pose\n        if (!sequence.includes(8)) {\n            sequence.push(8); // Lotus for meditation\n        }\n        \n        return sequence;\n    }\n    \n    findTransitionPose(fromPose, toPose) {\n        // Logic to find intermediate poses for smooth transitions\n        const transitionMap = {\n            '9->2': 1, // Mountain to Forward Fold via Raised Arms\n            '2->7': 3, // Forward Fold to Downward Dog via Low Lunge\n            '7->6': 5, // Downward Dog to Cobra via Eight-limbed\n            '6->4': null, // Direct transition\n            '4->8': null, // Direct transition\n        };\n        \n        const key = `${fromPose}->${toPose}`;\n        return transitionMap[key] || null;\n    }\n    \n    generateTransitions(poses) {\n        const transitions = {};\n        \n        for (let i = 0; i < poses.length - 1; i++) {\n            const from = poses[i];\n            const to = poses[i + 1];\n            const key = `${from}->${to}`;\n            \n            transitions[key] = {\n                name: `${this.getPoseName(from)} to ${this.getPoseName(to)}`,\n                instruction: this.generateTransitionInstruction(from, to),\n                duration: 3,\n                keyMovements: this.getKeyMovements(from, to)\n            };\n        }\n        \n        return transitions;\n    }\n    \n    generateTransitionInstruction(fromPose, toPose) {\n        // Generate natural language instructions for transitions\n        const instructions = {\n            '9->0': 'Bring your palms together at heart center',\n            '0->1': 'Inhale and sweep your arms overhead',\n            '1->2': 'Exhale and fold forward from your hips',\n            '2->3': 'Step your right foot back into a lunge',\n            '3->7': 'Step your left foot back and lift your hips',\n            '7->6': 'Lower your knees and chest, then lift your heart',\n            '6->8': 'Sit back and cross your legs comfortably',\n            '8->9': 'Come to standing with arms at your sides'\n        };\n        \n        const key = `${fromPose}->${toPose}`;\n        return instructions[key] || 'Transition smoothly to the next pose';\n    }\n    \n    getKeyMovements(fromPose, toPose) {\n        // Define key movement patterns for transition detection\n        return ['smooth_transition']; // Placeholder\n    }\n    \n    generateBenefits(focus) {\n        const benefitMap = {\n            'strength': ['Builds core strength', 'Improves muscle tone', 'Enhances stability'],\n            'flexibility': ['Increases range of motion', 'Improves joint mobility', 'Reduces stiffness'],\n            'balance': ['Enhances proprioception', 'Improves coordination', 'Builds mental focus'],\n            'balanced': ['Full body workout', 'Improves overall fitness', 'Enhances mind-body connection']\n        };\n        \n        return benefitMap[focus] || benefitMap['balanced'];\n    }\n    \n    // Utility methods\n    getCurrentInstruction() {\n        if (!this.currentSequence) return '';\n        \n        const currentPose = this.currentSequence.poses[this.currentPoseIndex];\n        return `Hold ${this.getPoseName(currentPose)}`;\n    }\n    \n    getNextPose() {\n        if (!this.currentSequence || this.currentPoseIndex >= this.currentSequence.poses.length - 1) {\n            return null;\n        }\n        \n        return this.currentSequence.poses[this.currentPoseIndex + 1];\n    }\n    \n    getPoseName(poseIndex) {\n        const poseNames = [\n            \"Pranamasana\", \"Hastauttanasana\", \"Hastapadasana\", \n            \"Ashwa Sanchalanasana\", \"Dandasana\", \"Ashtanga Namaskara\",\n            \"Bhujangasana\", \"Adho Mukha Svanasana\", \"Padmasana\", \"Tadasana\"\n        ];\n        \n        return poseNames[poseIndex] || 'Unknown Pose';\n    }\n    \n    calculateSequenceDuration() {\n        // Calculate actual time taken for sequence\n        return 0; // Placeholder\n    }\n    \n    calculateSequenceAccuracy() {\n        // Calculate average accuracy for the sequence\n        return 0; // Placeholder\n    }\n    \n    // Voice announcements\n    announceSequenceStart() {\n        if (window.speechSynthesis) {\n            const message = `Starting ${this.currentSequence.name}. This sequence has ${this.currentSequence.poses.length} poses.`;\n            const utterance = new SpeechSynthesisUtterance(message);\n            utterance.rate = 0.8;\n            window.speechSynthesis.speak(utterance);\n        }\n    }\n    \n    announceTransition(transition) {\n        if (window.speechSynthesis) {\n            const utterance = new SpeechSynthesisUtterance(transition.instruction);\n            utterance.rate = 0.8;\n            window.speechSynthesis.speak(utterance);\n        }\n    }\n    \n    announceSequenceCompletion(completionData) {\n        if (window.speechSynthesis) {\n            const message = `Congratulations! You completed ${completionData.sequence}. Great work!`;\n            const utterance = new SpeechSynthesisUtterance(message);\n            utterance.rate = 0.8;\n            window.speechSynthesis.speak(utterance);\n        }\n    }\n    \n    // Data persistence\n    loadUserPatterns() {\n        const saved = localStorage.getItem('smartSequence_userPatterns');\n        return saved ? JSON.parse(saved) : {};\n    }\n    \n    loadCustomSequences() {\n        const saved = localStorage.getItem('smartSequence_customSequences');\n        return saved ? JSON.parse(saved) : {};\n    }\n    \n    saveCustomSequences() {\n        localStorage.setItem('smartSequence_customSequences', JSON.stringify(this.customSequences));\n    }\n    \n    saveSequenceCompletion(completionData) {\n        const completions = JSON.parse(localStorage.getItem('smartSequence_completions') || '[]');\n        completions.push(completionData);\n        localStorage.setItem('smartSequence_completions', JSON.stringify(completions));\n    }\n    \n    // Public API\n    getAvailableSequences() {\n        return {\n            ...this.sequences,\n            ...this.customSequences\n        };\n    }\n    \n    isActive() {\n        return this.isSequenceActive;\n    }\n    \n    getCurrentSequence() {\n        return this.currentSequence;\n    }\n    \n    getProgress() {\n        return {\n            currentStep: this.currentPoseIndex + 1,\n            totalSteps: this.currentSequence ? this.currentSequence.poses.length : 0,\n            percentage: this.sequenceProgress\n        };\n    }\n    \n    stopSequence() {\n        this.isSequenceActive = false;\n        this.currentSequence = null;\n        this.currentPoseIndex = 0;\n        this.transitionDetector.isTransitioning = false;\n        console.log('Sequence stopped');\n    }\n}