const CACHE_NAME = 'yoga-studio-v1';
const urlsToCache = [
  '/',
  '/index.html',

  '/src/js/device_compatibility.js',
  '/src/js/yoga_pose_detector.js',
  '/src/js/yoga_timer.js',
  '/src/js/yoga_studio_app.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js',
  'https://cdn.jsdelivr.net/npm/@tensorflow-models/posenet@latest/dist/posenet.min.js'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        if (response) {
          return response;
        }
        return fetch(event.request);
      }
    )
  );
});