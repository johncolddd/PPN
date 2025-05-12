import tensorflow as tf
import psutil
import time

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