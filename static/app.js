document.addEventListener('DOMContentLoaded', async () => {
    // ── Element References ────────────────────────────────────────────────────
    const form               = document.getElementById('complaint-form');
    const submitBtn          = document.getElementById('submit-btn');
    const micBtn             = document.getElementById('mic-btn');
    const micIcon            = document.getElementById('mic-icon');
    const recorderStatus     = document.getElementById('recorder-status');
    const waveform           = document.getElementById('waveform');
    const hospitalSelect     = document.getElementById('hospital_name');

    // Image upload
    const uploadZone         = document.getElementById('upload-zone');
    const imageInput         = document.getElementById('image-input');
    const uploadTrigger      = document.getElementById('upload-trigger');
    const uploadZoneInner    = document.getElementById('upload-zone-inner');
    const imagePreviewWrapper= document.getElementById('image-preview-wrapper');
    const imagePreview       = document.getElementById('image-preview');
    const removeImageBtn     = document.getElementById('remove-image-btn');
    const captionStatus      = document.getElementById('caption-status');

    // AI badges
    const captionAiBadge     = document.getElementById('caption-ai-badge');
    const voiceAiBadge       = document.getElementById('voice-ai-badge');

    // ── State ─────────────────────────────────────────────────────────────────
    let isRecording   = false;
    let mediaRecorder = null;
    let audioChunks   = [];

    // ── Toast Notification ────────────────────────────────────────────────────
    function showToast(message, type = 'success') {
        const toast = document.getElementById('toast');
        toast.textContent = message;
        toast.classList.remove('show', 'error', 'warning');
        if (type === 'error')   toast.classList.add('error');
        if (type === 'warning') toast.classList.add('warning');
        void toast.offsetWidth; // force reflow
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 5000);
    }

    // ── Load Hospitals ────────────────────────────────────────────────────────
    async function loadHospitals() {
        try {
            const res = await fetch('/api/hospitals');
            const hospitals = await res.json();
            hospitalSelect.innerHTML = '<option value="" disabled selected>Select a hospital...</option>';
            hospitals.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                hospitalSelect.appendChild(opt);
            });
        } catch {
            hospitalSelect.innerHTML = '<option value="" disabled selected>Could not load hospitals</option>';
        }
    }
    await loadHospitals();

    // ═══════════════════════════════════════════════════════════════════════════
    //  WHISPER: Voice Recording via MediaRecorder API
    // ═══════════════════════════════════════════════════════════════════════════

    function setRecordingUI(recording) {
        if (recording) {
            micBtn.classList.add('recording');
            micIcon.className = 'fa-solid fa-stop';
            recorderStatus.textContent = 'Recording… Click to stop';
            recorderStatus.classList.add('recording');
            waveform.classList.add('active');
        } else {
            micBtn.classList.remove('recording');
            micIcon.className = 'fa-solid fa-microphone';
            recorderStatus.classList.remove('recording');
            waveform.classList.remove('active');
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];

            // Prefer webm/opus; fall back to whatever the browser supports
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : '';
            mediaRecorder = mimeType
                ? new MediaRecorder(stream, { mimeType })
                : new MediaRecorder(stream);

            mediaRecorder.ondataavailable = e => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                // Stop all microphone tracks
                stream.getTracks().forEach(t => t.stop());

                recorderStatus.textContent = '⏳ Transcribing with Whisper…';
                setRecordingUI(false);

                const mType = mediaRecorder.mimeType || 'audio/webm';
                const blob = new Blob(audioChunks, { type: mType });
                const ext  = mType.includes('ogg') ? '.ogg' : '.webm';

                try {
                    const formData = new FormData();
                    formData.append('file', blob, `recording${ext}`);

                    const res  = await fetch('/transcribe', { method: 'POST', body: formData });
                    const data = await res.json();

                    if (res.ok && data.text) {
                        document.getElementById('voice_text').value = data.text;
                        voiceAiBadge.style.display = 'inline-flex';
                        recorderStatus.textContent = '✅ Transcription complete';
                        showToast('🎙️ Voice transcribed by Whisper!', 'success');
                    } else {
                        throw new Error(data.detail || 'Transcription returned empty');
                    }
                } catch (err) {
                    recorderStatus.textContent = '❌ Transcription failed — please type manually';
                    showToast(`❌ Whisper error: ${err.message}`, 'error');
                } finally {
                    setTimeout(() => { recorderStatus.textContent = 'Click microphone to record'; }, 4000);
                }
            };

            mediaRecorder.start();
            isRecording = true;
            setRecordingUI(true);

        } catch (err) {
            showToast('❌ Microphone access denied or unavailable.', 'error');
            recorderStatus.textContent = 'Microphone not available';
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        isRecording = false;
    }

    micBtn.addEventListener('click', () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });

    // ═══════════════════════════════════════════════════════════════════════════
    //  BLIP: Image Upload + Caption Generation
    // ═══════════════════════════════════════════════════════════════════════════

    function showImagePreview(file) {
        const reader = new FileReader();
        reader.onload = e => {
            imagePreview.src = e.target.result;
            uploadZoneInner.style.display    = 'none';
            imagePreviewWrapper.style.display = 'flex';
        };
        reader.readAsDataURL(file);
    }

    function clearImage() {
        imageInput.value = '';
        imagePreview.src = '';
        uploadZoneInner.style.display     = 'flex';
        imagePreviewWrapper.style.display = 'none';
        captionStatus.style.display       = 'none';
        captionAiBadge.style.display      = 'none';
        document.getElementById('image_caption').value = '';
        uploadZone.classList.remove('dragover');
    }

    async function uploadAndCaption(file) {
        if (!file || !file.type.startsWith('image/')) {
            showToast('❌ Please upload a valid image file.', 'error');
            return;
        }

        showImagePreview(file);

        captionStatus.style.display = 'block';
        captionStatus.className     = 'caption-status loading';
        captionStatus.textContent   = '⏳ Generating caption with BLIP…';

        try {
            const formData = new FormData();
            formData.append('file', file, file.name);

            const res  = await fetch('/caption', { method: 'POST', body: formData });
            const data = await res.json();

            if (res.ok && data.caption) {
                document.getElementById('image_caption').value = data.caption;
                captionAiBadge.style.display = 'inline-flex';
                captionStatus.className      = 'caption-status success';
                captionStatus.textContent    = `✅ Caption: "${data.caption}"`;
                showToast('🖼️ Image captioned by BLIP!', 'success');
            } else {
                throw new Error(data.detail || 'No caption returned');
            }
        } catch (err) {
            captionStatus.className   = 'caption-status error';
            captionStatus.textContent = `❌ BLIP error: ${err.message} — type caption manually`;
            showToast(`❌ Caption error: ${err.message}`, 'error');
        }
    }

    // Trigger file dialog
    uploadTrigger.addEventListener('click', () => imageInput.click());
    uploadZoneInner.addEventListener('click', e => {
        if (e.target !== uploadTrigger) imageInput.click();
    });

    imageInput.addEventListener('change', () => {
        if (imageInput.files[0]) uploadAndCaption(imageInput.files[0]);
    });

    removeImageBtn.addEventListener('click', clearImage);

    // Drag and Drop
    uploadZone.addEventListener('dragover', e => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', e => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) uploadAndCaption(file);
    });

    // ═══════════════════════════════════════════════════════════════════════════
    //  Form Submission → POST /process
    // ═══════════════════════════════════════════════════════════════════════════

    form.addEventListener('submit', async e => {
        e.preventDefault();

        submitBtn.innerHTML  = '<i class="fa-solid fa-spinner fa-spin" style="margin-right:8px;"></i>Processing…';
        submitBtn.disabled   = true;

        const payload = {
            image_caption: document.getElementById('image_caption').value.trim(),
            voice_text:    document.getElementById('voice_text').value.trim(),
            hospital_name: hospitalSelect.value,
            city:          document.getElementById('city').value.trim(),
            name:          document.getElementById('patient_name').value.trim() || 'Anonymous',
            ward:          document.getElementById('ward').value.trim() || 'General Ward'
        };

        try {
            const res = await fetch('/process', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify(payload)
            });

            if (res.ok) {
                const data = await res.json();
                form.reset();
                clearImage();
                captionAiBadge.style.display = 'none';
                voiceAiBadge.style.display   = 'none';
                await loadHospitals();
                showToast(`✅ Complaint submitted! Severity: ${data.severity} · Dept: ${data.department}`, 'success');
            } else if (res.status === 409) {
                const err = await res.json();
                showToast(`⚠️ ${err.detail}`, 'warning');
            } else {
                const err = await res.json().catch(() => ({}));
                showToast(`❌ Submission failed: ${err.detail || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            console.error('Error processing complaint', error);
            showToast('❌ Network error. Please try again.', 'error');
        } finally {
            submitBtn.innerHTML = '<i class="fa-solid fa-paper-plane" style="margin-right:8px;"></i>Submit Complaint';
            submitBtn.disabled  = false;
        }
    });
});
