"""Project-wide named constants with documented meanings."""

# Randomness & reproducibility.
RANDOM_SEED: int = 42

# Default input size for CNN backbones (e.g., VGG16/ResNet).
DEFAULT_IMAGE_SIZE: int = 224

# Default batch size for training and evaluation.
DEFAULT_BATCH_SIZE: int = 32

# Default dataset paths (can be overridden via YAML).
TRAIN_CSV: str = "data/splits/train.csv"
TEST_CSV: str = "data/splits/test.csv"
CLASSES_TXT: str = "configs/classes.txt"

# ImageNet normalization statistics used by most pretrained CNNs.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
