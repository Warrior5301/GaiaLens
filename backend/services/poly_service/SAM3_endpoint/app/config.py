import os


class Settings:
    MODEL_ID: str = os.getenv("SAM3_MODEL_ID", "facebook/sam3")
    MAX_IMAGE_BYTES: int = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
    MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", str(25_000_000)))
    MIN_POLYGON_AREA: float = float(os.getenv("MIN_POLYGON_AREA", "20"))
    DEFAULT_THRESHOLD: float = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))


settings = Settings()
