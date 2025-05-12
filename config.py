class Config:
    # File paths
    BASE_PATH = '/content/drive/My Drive/PPN/'
    AMPLITUDE_PATH = BASE_PATH + 'amplitude.tiff'
    PHASE_PATH = BASE_PATH + 'phase.tiff'
    PROBE_PATH = BASE_PATH + 'probe.npy'

    # Data parameters
    IMAGE_SIZE = (512, 512)
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    OVERLAP_RATE = 75
    TRAIN_RATIO = 0.8
    POINT_SIZE = 9
    OVERLAP = 5 * POINT_SIZE

    # Model parameters
    PATCH_SIZE = 8
    EMBEDDING_DIM = 32
    NUM_HEADS = 2
    TRANSFORMER_LAYERS = 2
    BATCH_SIZE = 32
    EPOCHS = 30