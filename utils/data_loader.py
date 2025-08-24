import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.measure import block_reduce
from numpy.fft import fft2, fftshift
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_and_preprocess_image(image_path, new_size):
    image = imread(image_path).astype(np.float32)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = rgb2gray(image)
    return resize(image, new_size)

def create_complex_object(amplitude, phase):
    return amplitude * np.exp(1j * phase)

def adjust_amplitude_phase(amplitude, phase):
    amplitude_adjusted = 0.5 * (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min()) + 0.5
    phase_adjusted = -np.pi/3 * (phase + np.pi) / (2 * np.pi)
    return amplitude_adjusted, phase_adjusted

def simulate_diffraction_with_amp_phase(probe, amplitude, phase, pos, add_noise=False, noise_type='gaussian', noise_level=0.01, apply_jitter=False, jitter=0):
    if apply_jitter:
        jitter_x, jitter_y = np.random.randint(-jitter, jitter+1, 2)
        pos_jittered = (
            max(min(pos[0] + jitter_y, amplitude.shape[0] - probe.shape[0]), 0),
            max(min(pos[1] + jitter_x, amplitude.shape[1] - probe.shape[1]), 0)
        )
    else:
        pos_jittered = pos

    y, x = pos_jittered
    amp_section = amplitude[y:y+probe.shape[0], x:x+probe.shape[1]]
    phase_section = phase[y:y+probe.shape[0], x:x+probe.shape[1]]
    obj_section = amp_section * np.exp(1j * phase_section)

    if obj_section.shape != probe.shape:
        raise ValueError(f"Object section shape {obj_section.shape} does not match probe shape {probe.shape}.")

    diffraction = np.abs(fftshift(fft2(obj_section * probe)))**2

    if add_noise:
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, diffraction.shape)
            diffraction += noise
        elif noise_type == 'poisson':
            diffraction = np.random.poisson(diffraction)

    return diffraction, amp_section, phase_section

def calculate_step_size(probe_size, overlap_rate):
    return int(probe_size * (1 - overlap_rate / 100))

def load_and_scale_probe(probe_path, block_size=(4, 4)):
    probe = np.load(probe_path).astype(np.complex64)
    scale_func = lambda part: block_reduce(part, block_size=block_size, func=np.mean)
    scaled_real = scale_func(probe.real)
    scaled_imag = scale_func(probe.imag)
    return probe, scaled_real + 1j * scaled_imag

def load_and_reshape(data_list, target_shape):
    data = np.array(data_list)
    assert int(np.sqrt(data.shape[0]))**2 == data.shape[0], "Data does not form a perfect square."
    return data.reshape(target_shape)

def normalize_data(data, target_range=(0, 1)):
    scale = (target_range[1] - target_range[0]) / (data.max() - data.min())
    return (data - data.min()) * scale + target_range[0]

def resize_images(data, new_size=(32, 32), mode='reflect', anti_aliasing=True, apply_resize=True):
    if not apply_resize:
        return data
    resized_data = np.zeros((data.shape[0], data.shape[1]) + new_size, dtype=data.dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            resized_data[i, j] = resize(data[i, j], new_size, mode=mode, anti_aliasing=anti_aliasing)
    return resized_data

def resize_center_region(data, new_size=(32, 32), preserve_range=True, anti_aliasing=True, apply_resize=True):
    if not apply_resize:
        return data
    original_cut_size = new_size[0] // 2
    center_region_resized = np.zeros((data.shape[0], data.shape[1]) + new_size, dtype=data.dtype)
    for i in tqdm(range(data.shape[0])):
        for j in range(data.shape[1]):
            center_slice = data[i, j,
                                data.shape[2]//2 - original_cut_size : data.shape[2]//2 + original_cut_size,
                                data.shape[3]//2 - original_cut_size : data.shape[3]//2 + original_cut_size]
            center_region_resized[i, j] = resize(center_slice, new_size, preserve_range=preserve_range, anti_aliasing=anti_aliasing)
    return center_region_resized

def load_and_prepare_data(amplitude_path, phase_path, probe_path, new_size=(512, 512), overlap_rate=75, ratio=0.8):
    # Load images and probe
    amplitude = load_and_preprocess_image(amplitude_path, new_size)
    phase = load_and_preprocess_image(phase_path, new_size)
    probe, scaled_probe = load_and_scale_probe(probe_path, block_size=(4, 4))
    probe = scaled_probe

    # Adjust amplitude and phase
    amplitude, phase = adjust_amplitude_phase(amplitude, phase)
    object_complex = create_complex_object(amplitude, phase)

    # Generate positions
    step_size = calculate_step_size(probe.shape[0], overlap_rate)
    positions = [(y, x) for y in range(0, object_complex.shape[0] - probe.shape[0] + 1, step_size)
                 for x in range(0, object_complex.shape[1] - probe.shape[1] + 1, step_size)]

    # Simulate diffraction patterns
    diffraction_patterns_amp_phase = [
        simulate_diffraction_with_amp_phase(probe, amplitude, phase, pos)
        for pos in positions
    ]
    diffraction_patterns, amplitude_sections, phase_sections = zip(*diffraction_patterns_amp_phase)

    # Reshape and normalize data
    dimension_side = int(np.sqrt(len(phase_sections)))
    new_shape = (dimension_side, dimension_side, np.array(phase_sections).shape[1], np.array(phase_sections).shape[2])
    amp = load_and_reshape(amplitude_sections, new_shape)
    ph = load_and_reshape(phase_sections, new_shape)
    diff = load_and_reshape(diffraction_patterns, new_shape)

    amp = normalize_data(amp)
    ph = normalize_data(ph)
    diff = normalize_data(diff)

    # Resize images
    amp = resize_images(amp, new_size=(32, 32), apply_resize=True)
    ph = resize_images(ph, new_size=(32, 32), apply_resize=True)
    diff = resize_center_region(diff, new_size=(32, 32), apply_resize=True)

        # Split data
    nlines = int(diff.shape[0] * ratio)       # 训练用的行数
    tst_strt = diff.shape[0] - (diff.shape[0] - nlines)  # 测试集起始行（其实就是 nlines）

    # 训练集 = 前 80% 行，所有列
    X_train = diff[:nlines, :].reshape(-1, 32, 32)[:, :, :, np.newaxis]
    Y_I_train = amp[:nlines, :].reshape(-1, 32, 32)[:, :, :, np.newaxis]
    Y_phi_train = ph[:nlines, :].reshape(-1, 32, 32)[:, :, :, np.newaxis]

    # 测试集 = 后 20% 行 × 前 20% 列（保证是正方形）
    nltest = diff.shape[0] - nlines   # 测试区域的边长
    X_test = diff[nlines:, :nltest].reshape(-1, 32, 32)[:, :, :, np.newaxis]
    Y_I_test = amp[nlines:, :nltest].reshape(-1, 32, 32)[:, :, :, np.newaxis]
    Y_phi_test = ph[nlines:, :nltest].reshape(-1, 32, 32)[:, :, :, np.newaxis]


    # Shuffle training data
    X_train, Y_I_train, Y_phi_train = shuffle(X_train, Y_I_train, Y_phi_train, random_state=0)

    return X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test
