// In-browser CropAI classifier. Mirrors the checks and response shape of /api/predict in app.py,
// so the scanner works on static hosting without a running Python server.
(() => {
    const TFJS_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
    const MODEL_URL = 'models/tfjs/model.json';
    const LABELS_URL = 'models/labels.json';
    const DISEASE_INFO_URL = 'disease_info.json';
    const IMAGE_SIZE = 224;
    const MIN_PLANT_SCORE = 0.01;
    const MIN_CONFIDENCE = 0.55;
    const PLANT_SCORE_MAX_SIDE = 512;
    const HEALTHY_TOKENS = ['healthy', 'normal', 'no_disease'];

    class PredictionError extends Error {}

    let tfPromise = null;
    let modelPromise = null;
    let metadataPromise = null;

    const loadTf = () => {
        if (window.tf) return Promise.resolve(window.tf);
        tfPromise ??= new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = TFJS_URL;
            script.async = true;
            script.onload = () => resolve(window.tf);
            script.onerror = () => {
                tfPromise = null;
                reject(new Error('The AI engine could not be downloaded. Check your internet connection and reload the page.'));
            };
            document.head.append(script);
        });
        return tfPromise;
    };

    const fetchJson = async (url) => {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Could not load ${url} (HTTP ${response.status}).`);
        return response.json();
    };

    const loadMetadata = () => {
        metadataPromise ??= Promise.all([
            fetchJson(LABELS_URL),
            fetchJson(DISEASE_INFO_URL).catch(() => ({}))
        ]).then(([labels, info]) => {
            if (!Array.isArray(labels) || !labels.length) throw new Error('labels.json must contain a non-empty JSON list.');
            const diseaseInfo = {};
            Object.entries(info && typeof info === 'object' ? info : {}).forEach(([key, value]) => {
                if (value && typeof value === 'object') diseaseInfo[key.toLowerCase()] = value;
            });
            return { labels: labels.map(String), diseaseInfo };
        }).catch((error) => {
            metadataPromise = null;
            throw error;
        });
        return metadataPromise;
    };

    const loadModel = () => {
        modelPromise ??= Promise.all([loadTf(), loadMetadata()]).then(async ([tf, metadata]) => {
            const model = await tf.loadLayersModel(MODEL_URL);
            const outputs = model.outputs[0].shape[1];
            if (outputs !== metadata.labels.length) throw new Error('The model output does not match labels.json.');
            tf.tidy(() => model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])));
            return { tf, model, ...metadata };
        }).catch((error) => {
            modelPromise = null;
            throw error;
        });
        return modelPromise;
    };

    const decodeImage = async (file) => {
        let bitmap;
        try {
            bitmap = await createImageBitmap(file);
        } catch {
            throw new PredictionError('The uploaded file is not a readable image.');
        }
        if (bitmap.width < 80 || bitmap.height < 80) {
            bitmap.close?.();
            throw new PredictionError('Please upload a larger crop image, at least 80 by 80 pixels.');
        }
        return bitmap;
    };

    const drawToCanvas = (bitmap, width, height) => {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d', { willReadFrequently: true });
        context.imageSmoothingEnabled = true;
        context.imageSmoothingQuality = 'high';
        context.drawImage(bitmap, 0, 0, width, height);
        return { canvas, context };
    };

    // OpenCV-compatible 8-bit HSV: H in [0, 180), S and V in [0, 255].
    const toHsv = (r, g, b) => {
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const delta = max - min;
        let hue = 0;
        if (delta > 0) {
            if (max === r) hue = 60 * ((g - b) / delta);
            else if (max === g) hue = 60 * ((b - r) / delta) + 120;
            else hue = 60 * ((r - g) / delta) + 240;
            if (hue < 0) hue += 360;
        }
        return [Math.round(hue / 2), max === 0 ? 0 : Math.round((255 * delta) / max), max];
    };

    const inRange = (h, s, v, low, high) =>
        h >= low[0] && h <= high[0] && s >= low[1] && s <= high[1] && v >= low[2] && v <= high[2];

    const morph = (mask, width, height, keep) => {
        const out = new Uint8Array(mask.length);
        for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
                let result = keep === 'erode' ? 1 : 0;
                for (let dy = -2; dy <= 2 && result === (keep === 'erode' ? 1 : 0); dy += 1) {
                    const yy = y + dy;
                    if (yy < 0 || yy >= height) continue;
                    for (let dx = -2; dx <= 2; dx += 1) {
                        const xx = x + dx;
                        if (xx < 0 || xx >= width) continue;
                        const value = mask[yy * width + xx];
                        if (keep === 'erode' && !value) { result = 0; break; }
                        if (keep === 'dilate' && value) { result = 1; break; }
                    }
                }
                out[y * width + x] = result;
            }
        }
        return out;
    };

    // Estimate plant material using green, yellow, and brown vegetation colors (same ranges as app.py).
    const plantScore = (bitmap) => {
        const scale = Math.min(1, PLANT_SCORE_MAX_SIDE / Math.max(bitmap.width, bitmap.height));
        const width = Math.max(1, Math.round(bitmap.width * scale));
        const height = Math.max(1, Math.round(bitmap.height * scale));
        const { data } = drawToCanvas(bitmap, width, height).context.getImageData(0, 0, width, height);
        const mask = new Uint8Array(width * height);
        for (let index = 0; index < mask.length; index += 1) {
            const [h, s, v] = toHsv(data[index * 4], data[index * 4 + 1], data[index * 4 + 2]);
            mask[index] = inRange(h, s, v, [25, 35, 25], [100, 255, 255]) ||
                inRange(h, s, v, [15, 45, 35], [40, 255, 245]) ||
                inRange(h, s, v, [5, 35, 20], [25, 255, 210]) ? 1 : 0;
        }
        const opened = morph(morph(mask, width, height, 'erode'), width, height, 'dilate');
        return opened.reduce((sum, value) => sum + value, 0) / opened.length;
    };

    const displayLabel = (label) => label.replace(/___/g, ' - ').replace(/__/g, ' - ').replace(/_/g, ' ').trim();

    const isHealthy = (label) => HEALTHY_TOKENS.some((token) => label.toLowerCase().includes(token));

    const predictionDescription = (label) => {
        const disease = displayLabel(label);
        if (isHealthy(label)) {
            return `The trained CropAI classifier identified ${disease}. No disease pattern was found in the supported classes.`;
        }
        return `The trained CropAI classifier identified ${disease}. Treat this as an AI-assisted ` +
            'screening result and confirm it with a local agronomist before applying treatment.';
    };

    const diseaseInformation = (diseaseInfo, label, healthy) => {
        const normalized = label.toLowerCase().replace(/___/g, ' ').replace(/__/g, ' ').replace(/_/g, ' ');
        const matchedKey = Object.keys(diseaseInfo).find((key) => key !== 'default' && normalized.includes(key));
        const matched = matchedKey ? diseaseInfo[matchedKey] : diseaseInfo[healthy ? 'healthy' : 'default'] || {};
        return {
            cause: matched.cause || 'Cause information is not available for this trained class.',
            symptoms: matched.symptoms || 'Review the image and confirm visible symptoms with an agronomist.',
            action: matched.action || 'Isolate affected plants and seek local agricultural guidance.',
            food_safety: matched.food_safety ||
                'Do not eat visibly diseased plant parts. Wash produce and follow local food-safety guidance.'
        };
    };

    const predict = async (file) => {
        const { tf, model, labels, diseaseInfo } = await loadModel();
        const bitmap = await decodeImage(file);
        let probabilities;
        try {
            if (plantScore(bitmap) < MIN_PLANT_SCORE) {
                throw new PredictionError(
                    'This does not look like a crop or plant image. Upload a clear leaf, fruit, stem, or crop image.'
                );
            }
            const { canvas } = drawToCanvas(bitmap, IMAGE_SIZE, IMAGE_SIZE);
            const output = tf.tidy(() => model.predict(tf.browser.fromPixels(canvas).toFloat().expandDims(0)));
            probabilities = await output.data();
            output.dispose();
        } finally {
            bitmap.close?.();
        }

        let index = 0;
        for (let i = 1; i < probabilities.length; i += 1) {
            if (probabilities[i] > probabilities[index]) index = i;
        }
        const confidence = probabilities[index];
        if (confidence < MIN_CONFIDENCE) {
            throw new PredictionError(
                'The image is a plant, but the model is not confident enough to identify a disease. ' +
                'Use a clearer leaf image or confirm it with an agronomist.'
            );
        }

        const label = labels[index];
        const plant = label.split('___')[0].split('__')[0].replace(/_/g, ' ').trim();
        const healthy = isHealthy(label);
        return {
            disease: displayLabel(label),
            condition: displayLabel(label),
            status: healthy ? 'Healthy' : 'Potential risk',
            plant: plant || 'Plant',
            confidence: Math.round(confidence * 10000) / 10000,
            description: predictionDescription(label),
            ...diseaseInformation(diseaseInfo, label, healthy)
        };
    };

    const labelCount = async () => (await loadMetadata()).labels.length;

    window.CropAI = { loadModel, predict, labelCount, PredictionError };
})();
