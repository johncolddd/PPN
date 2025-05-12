import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import psutil
import time
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.epoch_start_time
        self.epoch_times.append(elapsed_time)
        memory_used = psutil.virtual_memory().used / (1024 ** 3)
        print(f'Memory used after epoch {epoch}: {memory_used:.2f} GB')
        print(f'Training time for epoch {epoch}: {elapsed_time:.3f} seconds')
        logs['memory_used'] = memory_used
        logs['training_time'] = elapsed_time

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        print(f'Average training time per epoch: {avg_epoch_time:.3f} seconds')
        print(f'Total training time: {total_time:.3f} seconds')

def calculate_psnr(image1, image2):
    mse_val = np.mean((image1 - image2) ** 2)
    if mse_val == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse_val))
    return psnr

def calculate_ssim(image1, image2):
    return ssim(image1, image2, data_range=image2.max() - image2.min())

def train_ppn(model, X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, batch_size=32, epochs=30):
    # Preprocessing
    X_train = X_train / np.max(X_train)
    Y_I_train = Y_I_train / np.max(Y_I_train)
    Y_phi_train = Y_phi_train / np.max(Y_phi_train)

    # Callbacks
    memory_callback = CustomCallback()
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=0.0001,
        verbose=1
    )

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train model
    history = model.fit(
        X_train, [Y_I_train, Y_phi_train],
        shuffle=True,
        batch_size=batch_size,
        verbose=1,
        epochs=epochs,
        validation_split=0.05,
        callbacks=[reduce_lr, memory_callback]
    )

    # Evaluate model
    predictions = model.predict(X_test)
    preds_intens_amp, preds_intens_ph = predictions[:2]

    amp_squeeze = np.squeeze(preds_intens_amp)
    ph_squeeze = np.squeeze(preds_intens_ph)
    amp_flat = amp_squeeze.reshape(len(amp_squeeze), -1)
    Y_I_test_flat = Y_I_test.reshape(len(Y_I_test), -1)
    ph_flat = ph_squeeze.reshape(len(ph_squeeze), -1)
    Y_phi_test_flat = Y_phi_test.reshape(len(Y_phi_test), -1)

    print("MSE in amplitude:", mse(amp_flat, Y_I_test_flat))
    print("MSE in phase:", mse(ph_flat, Y_phi_test_flat))
    print("PSNR in amplitude:", calculate_psnr(amp_flat, Y_I_test_flat))
    print("SSIM in amplitude:", calculate_ssim(amp_flat, Y_I_test_flat))
    print("PSNR in phase:", calculate_psnr(ph_flat, Y_phi_test_flat))
    print("SSIM in phase:", calculate_ssim(ph_flat, Y_phi_test_flat))

    # Plot training history
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], 'bo-', label='Training loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], 'r^-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return history, predictions