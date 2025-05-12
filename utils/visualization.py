import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.keras.models import Model

class AdaptiveImageStitcher:
    def __init__(self, point_size: int, overlap: int):
        self.point_size = point_size
        self.overlap = overlap

    def stitch(self, data: np.ndarray, grid_size: int) -> np.ndarray:
        tile_size = data.shape[1]
        composite_size = grid_size * self.point_size + self.overlap
        composite = np.zeros((composite_size, composite_size), dtype=float)
        weight_map = np.zeros_like(composite)

        reshaped_data = data.reshape(grid_size, grid_size, tile_size, tile_size)

        for i in range(grid_size):
            for j in range(grid_size):
                x_start, y_start = self.point_size * i, self.point_size * j
                x_end, y_end = x_start + tile_size, y_start + tile_size

                tile = reshaped_data[i, j]
                weight = self._generate_adaptive_weight(tile.shape)

                composite[x_start:x_end, y_start:y_end] += tile * weight
                weight_map[x_start:x_end, y_start:y_end] += weight

        normalized_image = np.divide(composite, weight_map, where=weight_map != 0)
        return self._crop_image(normalized_image)

    def _generate_adaptive_weight(self, shape: tuple) -> np.ndarray:
        y, x = np.ogrid[:shape[0], :shape[1]]
        center = np.array(shape) / 2
        dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        weight = 1 - dist_from_center / np.max(dist_from_center)
        return weight**2

    def _crop_image(self, image: np.ndarray) -> np.ndarray:
        crop = self.overlap // 2
        return image[crop:-crop, crop:-crop]

def process_and_visualize(model: Model, X_test: np.ndarray, Y_I_test: np.ndarray, Y_phi_test: np.ndarray, point_size: int, overlap: int, sub_image1: np.ndarray, sub_image2: np.ndarray):
    predictions = model.predict(X_test)
    preds_intens_amp, preds_intens_ph = predictions[:2]

    grid_size = int(np.sqrt(X_test.shape[0]))
    stitcher = AdaptiveImageStitcher(point_size, overlap)

    stitched_amp = stitcher.stitch(preds_intens_amp, grid_size)
    stitched_phase = stitcher.stitch(preds_intens_ph, grid_size)

    # Visualization
    plt.rcParams.update({'font.size': 26})
    fig, ax = plt.subplots(2, 2, figsize=(15, 15), dpi=400)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Amplitude
    im1 = ax[0, 0].imshow(sub_image1)
    plt.colorbar(im1, ax=ax[0, 0], fraction=0.046, pad=0.04)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Amplitude')

    im2 = ax[0, 1].imshow(stitched_amp)
    plt.colorbar(im2, ax=ax[0, 1], fraction=0.046, pad=0.04)
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Stitched Amplitude')

    # Phase
    def pi_formatter(x, pos):
        m = np.round(x / np.pi, decimals=2)
        if m == 0:
            return '0'
        elif m == 1:
            return r'$\pi$'
        elif m == -1:
            return r'$-\pi$'
        else:
            return r'${}\pi$'.format(m)

    im3 = ax[1, 0].imshow(sub_image2)
    cbar = plt.colorbar(im3, ax=ax[1, 0], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(pi_formatter))
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Phase')

    im4 = ax[1, 1].imshow(stitched_phase)
    cbar = plt.colorbar(im4, ax=ax[1, 1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(pi_formatter))
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Stitched Phase')

    print(f"Stitched phase shape: {stitched_phase.shape}")
    plt.show()