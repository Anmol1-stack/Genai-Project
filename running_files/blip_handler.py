from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import logging
import os

logger = logging.getLogger(__name__)

MODEL_NAME = "Salesforce/blip-image-captioning-base"
_processor = None
_model = None

def get_blip():
    global _processor, _model
    if _processor is None or _model is None:
        logger.info(f"Loading BLIP model '{MODEL_NAME}'...")
        _processor = BlipProcessor.from_pretrained(MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
        logger.info("BLIP model loaded successfully.")
    return _processor, _model

def generate_caption(file_bytes: bytes) -> str:
    """
    Accepts raw image bytes (JPEG/PNG/etc.), runs BLIP conditional captioning,
    and returns a descriptive caption string.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not read image: {str(e)}")

    try:
        processor, model = get_blip()

        # Unconditional captioning
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=60)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()

        logger.info(f"BLIP caption: {caption}")
        return caption if caption else "No caption generated."
    except Exception as e:
        logger.error(f"BLIP caption error: {e}")
        raise RuntimeError(f"Caption generation failed: {str(e)}")
