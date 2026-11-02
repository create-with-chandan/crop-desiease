document.addEventListener('DOMContentLoaded', () => {
    const body = document.body;
    const loader = document.querySelector('.loader');
    const menuButton = document.querySelector('.menu-btn, .menu-toggle, .menu');
    const navLinks = document.querySelector('.nav-links, .nav');
    const cursorGlow = document.querySelector('.cursor-glow');

    window.setTimeout(() => loader?.classList.add('hidden'), 900);

    menuButton?.addEventListener('click', () => {
        const isOpen = navLinks?.classList.toggle('open') ?? false;
        menuButton.setAttribute('aria-expanded', String(isOpen));
        body.classList.toggle('menu-open', isOpen);
    });

    navLinks?.querySelectorAll('a').forEach((link) => {
        link.addEventListener('click', () => {
            navLinks.classList.remove('open');
            body.classList.remove('menu-open');
            menuButton?.setAttribute('aria-expanded', 'false');
        });
    });

    if (cursorGlow && window.matchMedia('(pointer: fine)').matches) {
        window.addEventListener('pointermove', (event) => {
            cursorGlow.style.left = `${event.clientX}px`;
            cursorGlow.style.top = `${event.clientY}px`;
        });
    }

    const revealItems = document.querySelectorAll('.reveal');
    if ('IntersectionObserver' in window) {
        const revealObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.12 });
        revealItems.forEach((item) => revealObserver.observe(item));
    } else {
        revealItems.forEach((item) => item.classList.add('visible'));
    }

    document.querySelectorAll('.year').forEach((year) => {
        year.textContent = String(new Date().getFullYear());
    });

    const homeModelStatus = document.querySelector('#homeModelStatus');
    const homeModelMessage = document.querySelector('#homeModelMessage');
    const homeClassCount = document.querySelector('#homeClassCount');
    const homeClassMessage = document.querySelector('#homeClassMessage');
    if (homeModelStatus && homeModelMessage && homeClassCount && homeClassMessage) {
        const updateHomeModelStatus = async () => {
            try {
                const count = await window.CropAI.labelCount();
                homeModelStatus.textContent = 'READY';
                homeModelMessage.textContent = 'RUNS IN BROWSER';
                homeClassCount.textContent = String(count);
                homeClassMessage.textContent = 'LOADED CLASSES';
            } catch {
                homeModelStatus.textContent = 'OFFLINE';
                homeModelMessage.textContent = 'MODEL UNAVAILABLE';
                homeClassMessage.textContent = 'MODEL REQUIRED';
            }
        };

        updateHomeModelStatus();
    }

    const imageInput = document.querySelector('#imageInput');
    const imageChooser = document.querySelector('label[for="imageInput"]');
    const uploadZone = document.querySelector('#uploadZone');
    const previewContainer = document.querySelector('#previewContainer');
    const scanButton = document.querySelector('#scanBtn');
    const removeButton = document.querySelector('#removeBtn');
    const placeholder = document.querySelector('#resultPlaceholder');
    const resultCard = document.querySelector('#resultCard');
    const confidenceValue = document.querySelector('#confidenceValue');
    const confidenceFill = document.querySelector('#confidenceFill');
    const diseaseName = document.querySelector('#diseaseName');
    const resultDescription = document.querySelector('#resultDescription');
    const plantType = document.querySelector('#plantType');
    const resultCondition = document.querySelector('#resultCondition');
    const resultCause = document.querySelector('#resultCause');
    const resultSymptoms = document.querySelector('#resultSymptoms');
    const resultAction = document.querySelector('#resultAction');
    const resultFoodSafety = document.querySelector('#resultFoodSafety');
    const resultStatus = document.querySelector('#resultStatus');

    let selectedFile = null;
    let previewUrl = null;
    let activeController = null;
    let isAnalyzing = false;

    const showToast = (message) => {
        let toast = document.querySelector('.toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.className = 'toast';
            toast.setAttribute('role', 'status');
            body.append(toast);
        }
        toast.textContent = message;
        toast.classList.add('show');
        window.clearTimeout(showToast.timeoutId);
        showToast.timeoutId = window.setTimeout(() => toast.classList.remove('show'), 3200);
    };

    const clearPreview = () => {
        activeController?.abort();
        activeController = null;
        isAnalyzing = false;
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = null;
        selectedFile = null;
        if (previewContainer) previewContainer.innerHTML = '';
        if (scanButton) scanButton.disabled = true;
        if (scanButton) {
            scanButton.classList.remove('is-loading');
            scanButton.textContent = 'Analyze With AI →';
        }
        if (imageInput) imageInput.value = '';
        uploadZone?.classList.remove('has-preview');
        resultCard?.classList.remove('show');
        if (resultStatus) {
            resultStatus.hidden = true;
            resultStatus.textContent = '';
        }
        if (placeholder) placeholder.hidden = false;
    };

    const showResultError = (message) => {
        resultCard?.classList.remove('show');
        if (placeholder) placeholder.hidden = true;
        if (resultStatus) {
            resultStatus.hidden = false;
            resultStatus.textContent = message;
        }
    };

    const loadImage = async (file) => {
        if (!file) return;
        if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
            showToast('Choose a JPG, JPEG, or PNG image.');
            return;
        }
        if (file.size > 8 * 1024 * 1024) {
            showToast('Please choose an image smaller than 8 MB.');
            return;
        }

        const nextPreviewUrl = URL.createObjectURL(file);
        try {
            const preview = new Image();
            preview.src = nextPreviewUrl;
            await preview.decode();
        } catch {
            URL.revokeObjectURL(nextPreviewUrl);
            showToast('That file could not be opened as an image. Please choose another.');
            return;
        }

        activeController?.abort();
        activeController = null;
        isAnalyzing = false;
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        selectedFile = file;
        previewUrl = nextPreviewUrl;
        if (previewContainer) {
            const preview = document.createElement('img');
            preview.className = 'preview-image';
            preview.src = previewUrl;
            preview.alt = 'Selected crop image preview';
            previewContainer.replaceChildren(preview);
        }
        if (scanButton) scanButton.disabled = false;
        if (scanButton) {
            scanButton.classList.remove('is-loading');
            scanButton.textContent = 'Analyze With AI →';
        }
        uploadZone?.classList.add('has-preview');
        resultCard?.classList.remove('show');
        if (resultStatus) {
            resultStatus.hidden = true;
            resultStatus.textContent = '';
        }
        if (placeholder) placeholder.hidden = false;
        showToast('Image ready. Start the AI analysis when you are ready.');
    };

    imageInput?.addEventListener('change', (event) => {
        loadImage(event.target.files?.[0]);
    });
    imageChooser?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            imageInput?.click();
        }
    });
    imageInput?.addEventListener('click', () => {
        imageInput.value = '';
    });
    removeButton?.addEventListener('click', clearPreview);

    ['dragenter', 'dragover'].forEach((eventName) => {
        uploadZone?.addEventListener(eventName, (event) => {
            event.preventDefault();
            uploadZone.classList.add('dragover');
        });
    });
    ['dragleave', 'drop'].forEach((eventName) => {
        uploadZone?.addEventListener(eventName, (event) => {
            event.preventDefault();
            uploadZone.classList.remove('dragover');
        });
    });
    uploadZone?.addEventListener('drop', (event) => loadImage(event.dataTransfer.files?.[0]));

    const checkService = async () => {
        if (!resultStatus) return;
        if (window.location.protocol === 'file:') {
            showResultError('Open the scanner from the website instead of opening scanner.html directly from your computer.');
            return;
        }

        try {
            await window.CropAI.loadModel();
        } catch (error) {
            showResultError(`The CropAI model could not be loaded. ${error.message}`);
        }
    };

    checkService();

    scanButton?.addEventListener('click', async () => {
        if (!selectedFile || isAnalyzing) return;
        const controller = new AbortController();
        isAnalyzing = true;
        activeController = controller;
        scanButton.disabled = true;
        scanButton.classList.add('is-loading');
        scanButton.textContent = 'Analyzing image...';
        placeholder?.setAttribute('hidden', '');
        resultCard?.classList.remove('show');

        try {
            const payload = await window.CropAI.predict(selectedFile);
            if (controller.signal.aborted) return;
            if (!payload || typeof payload.disease !== 'string' ||
                typeof payload.plant !== 'string' ||
                typeof payload.confidence !== 'number' ||
                !Number.isFinite(payload.confidence)) {
                throw new Error('The AI service returned an invalid prediction. Please try again.');
            }

            const confidence = Math.round(Math.max(0, Math.min(1, payload.confidence)) * 100);
            if (diseaseName) diseaseName.textContent = payload.disease;
            confidenceValue.textContent = `${confidence}%`;
            confidenceFill.style.width = `${confidence}%`;
            plantType.textContent = payload.plant;
            if (resultCondition) resultCondition.textContent = payload.status || 'Potential risk';
            resultDescription.textContent = payload.description || 'No additional analysis was provided.';
            if (resultCause) resultCause.textContent = payload.cause || 'No cause information is available.';
            if (resultSymptoms) resultSymptoms.textContent = payload.symptoms || 'No symptom information is available.';
            if (resultAction) resultAction.textContent = payload.action || 'Consult a local agricultural specialist.';
            if (resultFoodSafety) resultFoodSafety.textContent = payload.food_safety || 'Follow local food-safety guidance.';
            if (resultStatus) resultStatus.hidden = true;
            resultCard?.classList.add('show');
            showToast('Analysis complete. Review the model result.');
        } catch (error) {
            if (!controller.signal.aborted && error.name !== 'AbortError') {
                const message = error instanceof window.CropAI.PredictionError
                    ? error.message
                    : `The AI model could not analyze this image: ${error.message}`;
                showResultError(message);
                showToast(message);
            }
        } finally {
            if (activeController === controller) {
                isAnalyzing = false;
                activeController = null;
                scanButton.disabled = !selectedFile;
                scanButton.classList.remove('is-loading');
                scanButton.textContent = 'Analyze With AI →';
            }
        }
    });
});
