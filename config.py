class Config:
    # File paths
    BASE_PATH = '/content/drive/My Drive/PtychoNN-master/TF2/apl_test/image/'
    AMPLITUDE_PATH = BASE_PATH + 'true_amplitude.tiff'
    PHASE_PATH = BASE_PATH + 'true_phase.tiff'
    PROBE_PATH = BASE_PATH + 'pxy_probe1000.npy'

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