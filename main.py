import tensorflow as tf
import numpy as np
from models.ppn import build_ppn_model
from train.train_ppn import train_ppn
from utils.data_loader import load_and_prepare_data
from utils.visualization import process_and_visualize
from config import Config

def main():
    # Set random seed
    tf.keras.backend.clear_session()
    np.random.seed(123)
    tf.random.set_seed(123)

    # Load and prepare data
    X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test = load_and_prepare_data(
        Config.AMPLITUDE_PATH,
        Config.PHASE_PATH,
        Config.PROBE_PATH,
        new_size=Config.IMAGE_SIZE,
        overlap_rate=Config.OVERLAP_RATE,
        ratio=Config.TRAIN_RATIO
    )

    # Load sub-images for visualization
    amplitude = load_and_preprocess_image(Config.AMPLITUDE_PATH, Config.IMAGE_SIZE)
    phase = load_and_preprocess_image(Config.PHASE_PATH, Config.IMAGE_SIZE)
    amplitude, phase = adjust_amplitude_phase(amplitude, phase)
    sub_image1 = amplitude[412:512, 412:512]
    sub_image2 = phase[412:512, 412:512]

    # Build model
    model = build_ppn_model(
        h=Config.IMAGE_HEIGHT,
        w=Config.IMAGE_WIDTH,
        patch_size=Config.PATCH_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        num_heads=Config.NUM_HEADS,
        transformer_layers=Config.TRANSFORMER_LAYERS
    )

    # Train model
    history, predictions = train_ppn(
        model,
        X_train,
        Y_I_train,
        Y_phi_train,
        X_test,
        Y_I_test,
        Y_phi_test,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS
    )

    # Visualize results
    process_and_visualize(
        model,
        X_test,
        Y_I_test,
        Y_phi_test,
        point_size=Config.POINT_SIZE,
        overlap=Config.OVERLAP,
        sub_image1=sub_image1,
        sub_image2=sub_image2
    )

if __name__ == "__main__":
    main()