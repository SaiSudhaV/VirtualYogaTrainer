// Device Compatibility and Performance Optimization
class DeviceCompatibility {
    constructor() {
        this.isMobile = this.detectMobile();
        this.isTablet = this.detectTablet();
        this.browserSupport = this.checkBrowserSupport();
        this.performanceLevel = this.detectPerformanceLevel();
        this.cameraConstraints = this.getOptimalCameraConstraints();
    }

    detectMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }

    detectTablet() {
        return /iPad|Android(?!.*Mobile)/i.test(navigator.userAgent) || 
               (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    }

    checkBrowserSupport() {
        const support = {
            webgl: this.checkWebGLSupport(),
            webrtc: this.checkWebRTCSupport(),
            mediaDevices: !!navigator.mediaDevices,
            getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
        };
        return support;
    }

    checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            return !!(window.WebGLRenderingContext && 
                     (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
        } catch (e) {
            return false;
        }
    }

    checkWebRTCSupport() {
        return !!(window.RTCPeerConnection || window.mozRTCPeerConnection || window.webkitRTCPeerConnection);
    }

    detectPerformanceLevel() {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        
        if (!gl) return 'low';
        
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
            const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
            if (renderer.includes('Intel') || renderer.includes('AMD')) return 'medium';
            if (renderer.includes('NVIDIA') || renderer.includes('GeForce')) return 'high';
        }
        
        // Fallback performance detection
        const cores = navigator.hardwareConcurrency || 4;
        const memory = navigator.deviceMemory || 4;
        
        if (cores >= 8 && memory >= 8) return 'high';
        if (cores >= 4 && memory >= 4) return 'medium';
        return 'low';
    }

    getOptimalCameraConstraints() {
        let constraints = {
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
            },
            audio: false
        };

        // Adjust for mobile devices
        if (this.isMobile) {
            constraints.video.width = { ideal: 480 };
            constraints.video.height = { ideal: 360 };
            constraints.video.frameRate = { ideal: 24 };
        }

        // Adjust for performance level
        if (this.performanceLevel === 'low') {
            constraints.video.width = { ideal: 320 };
            constraints.video.height = { ideal: 240 };
            constraints.video.frameRate = { ideal: 15 };
        }

        return constraints;
    }

    async requestCameraPermission() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia(this.cameraConstraints);
            stream.getTracks().forEach(track => track.stop()); // Stop immediately after permission
            return true;
        } catch (error) {
            console.error('Camera permission denied:', error);
            return false;
        }
    }

    optimizeForDevice() {
        // Add device-specific CSS classes
        document.body.classList.add(
            this.isMobile ? 'mobile-device' : 'desktop-device',
            this.isTablet ? 'tablet-device' : '',
            `performance-${this.performanceLevel}`
        );

        // Adjust UI for touch devices
        if (this.isMobile || this.isTablet) {
            this.enableTouchOptimizations();
        }

        // Show compatibility warnings
        this.showCompatibilityWarnings();
    }

    enableTouchOptimizations() {
        // Add touch-friendly styles
        const style = document.createElement('style');
        style.textContent = `
            .mobile-device .btn {
                min-height: 44px;
                min-width: 44px;
                touch-action: manipulation;
            }
            
            .mobile-device .pose-item {
                min-height: 44px;
                padding: 12px;
            }
            
            .mobile-device select {
                min-height: 44px;
                font-size: 16px;
            }
            
            .tablet-device .sidebar {
                width: 250px;
            }
            
            .performance-low .video-container {
                filter: contrast(1.1) brightness(1.1);
            }
        `;
        document.head.appendChild(style);

        // Prevent zoom on input focus (iOS)
        const viewport = document.querySelector('meta[name="viewport"]');
        if (viewport) {
            viewport.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no';
        }
    }

    showCompatibilityWarnings() {
        const warnings = [];

        if (!this.browserSupport.webgl) {
            warnings.push('WebGL not supported - performance may be limited');
        }

        if (!this.browserSupport.getUserMedia) {
            warnings.push('Camera access not supported in this browser');
        }

        if (this.performanceLevel === 'low') {
            warnings.push('Low performance device detected - reduced quality settings applied');
        }

        if (warnings.length > 0) {
            this.displayWarnings(warnings);
        }
    }

    displayWarnings(warnings) {
        const warningContainer = document.createElement('div');
        warningContainer.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(255, 193, 7, 0.9);
            color: #000;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            max-width: 300px;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        `;

        warningContainer.innerHTML = `
            <strong>⚠️ Compatibility Notice:</strong><br>
            ${warnings.map(w => `• ${w}`).join('<br>')}
            <button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; font-size: 16px; cursor: pointer;">×</button>
        `;

        document.body.appendChild(warningContainer);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (warningContainer.parentElement) {
                warningContainer.remove();
            }
        }, 10000);
    }

    getDeviceInfo() {
        return {
            isMobile: this.isMobile,
            isTablet: this.isTablet,
            performanceLevel: this.performanceLevel,
            browserSupport: this.browserSupport,
            userAgent: navigator.userAgent,
            screen: {
                width: screen.width,
                height: screen.height,
                pixelRatio: window.devicePixelRatio || 1
            }
        };
    }
}

// Auto-initialize device compatibility
const deviceCompatibility = new DeviceCompatibility();

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    deviceCompatibility.optimizeForDevice();
    console.log('Device Info:', deviceCompatibility.getDeviceInfo());
});