// ============================================================
// IMPORTANT: Replace this with your actual Render backend URL
// after deploying. Example: 'https://deepfake-api.onrender.com'
// ============================================================
const API_URL = 'https://deepfake-detection-api.onrender.com';

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewArea = document.getElementById('preview-area');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingState = document.getElementById('loading-state');
    const resultsSection = document.getElementById('results-section');
    const errorMessage = document.getElementById('error-message');

    let selectedFile = null;

    const dropdownToggle = document.getElementById('dropdown-toggle');
    const dropdownContent = document.getElementById('dropdown-content');
    if (dropdownToggle && dropdownContent) {
        dropdownToggle.addEventListener('click', () => {
            if (dropdownContent.style.display === 'none') {
                dropdownContent.style.display = 'block';
                dropdownToggle.textContent = 'Hide Detailed Analysis ▲';
            } else {
                dropdownContent.style.display = 'none';
                dropdownToggle.textContent = 'View Detailed Analysis ▼';
            }
        });
    }

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('dragover'); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            showError('Please upload an image file (JPG, PNG, etc.)');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.style.display = 'none';
            previewArea.style.display = 'block';
            resultsSection.style.display = 'none';
            errorMessage.textContent = '';
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewArea.style.display = 'none';
        dropZone.style.display = 'block';
        resultsSection.style.display = 'none';
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        previewArea.style.display = 'none';
        loadingState.style.display = 'block';
        resultsSection.style.display = 'none';
        errorMessage.textContent = '';
        document.getElementById('result-card').style.display = 'block';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            loadingState.style.display = 'none';
            previewArea.style.display = 'block';

            if (data.error) { showError(data.error); return; }
            displayResults(data);

        } catch (error) {
            loadingState.style.display = 'none';
            previewArea.style.display = 'block';
            showError('Could not reach the analysis server. Please try again shortly.');
            console.error(error);
        }
    });

    function displayResults(data) {
        resultsSection.style.display = 'block';
        const verdictEl = document.getElementById('verdict');
        const confidenceEl = document.getElementById('confidence');

        verdictEl.textContent = data.result;
        let baseClass = data.result === 'Real' ? 'real' : 'fake';
        verdictEl.className = 'verdict ' + baseClass;
        confidenceEl.textContent = `Confidence: ${data.confidence}`;

        document.getElementById('real-val').textContent = data.real_percentage;
        document.getElementById('fake-val').textContent = data.fake_percentage;
        setTimeout(() => {
            document.getElementById('real-bar').style.width = data.real_percentage;
            document.getElementById('fake-bar').style.width = data.fake_percentage;
        }, 50);

        const dropdown = document.getElementById('reasoning-dropdown');
        const dropdownContent = document.getElementById('dropdown-content');

        if (data.analysis) {
            dropdown.style.display = 'block';
            const a = data.analysis;
            const fields = [
                ['asymmetry', a.asymmetry],
                ['colour',    a.colour],
                ['artifact',  a.artifacts],
                ['blur',      a.blur],
                ['noise',     a.noise],
            ];
            fields.forEach(([key, val]) => {
                document.getElementById(`${key}-val`).textContent = `${val}%`;
                setTimeout(() => {
                    document.getElementById(`${key}-bar`).style.width = `${val}%`;
                }, 50);
            });
        } else {
            dropdown.style.display = 'none';
            if (dropdownContent) dropdownContent.style.display = 'none';
        }
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        resultsSection.style.display = 'block';
        document.getElementById('result-card').style.display = 'none';
    }
});
