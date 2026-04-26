import whisper
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

# Load the model once at module level (cached in memory)
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading Whisper model '{MODEL_SIZE}'...")
        _model = whisper.load_model(MODEL_SIZE)
        logger.info("Whisper model loaded successfully.")
    return _model

def transcribe_audio(file_bytes: bytes, filename: str = "audio.webm") -> str:
    """
    Accepts raw audio bytes (WebM, WAV, MP4, etc.), saves to a temp file,
    runs Whisper inference, and returns the transcribed text.
    """
    ext = os.path.splitext(filename)[-1] if filename else ".webm"
    
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        model = get_model()
        result = model.transcribe(tmp_path, fp16=False)
        text = result.get("text", "").strip()
        logger.info(f"Whisper transcription: {text[:80]}...")
        return text if text else "Could not transcribe audio. Please try again."
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        raise RuntimeError(f"Transcription failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
