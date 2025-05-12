import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2DTranspose, Dense, Flatten, Reshape, MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization, Embedding, BatchNormalization, Conv2D, AveragePooling2D, Concatenate
from tensorflow.keras.models import Model
import numpy as np

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, embedding_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.projection = Dense(embedding_dim)
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=embedding_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class EnhancedPolarAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(EnhancedPolarAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.projection = tf.keras.layers.Dense(3 * key_dim)
        self.final_proj = tf.keras.layers.Dense(key_dim)

    def build(self, input_shape):
        _, H, W, _ = input_shape
        self.H = H
        self.W = W
        self.center_x = self.add_weight(shape=(), initializer='zeros', name='center_x')
        self.center_y = self.add_weight(shape=(), initializer='zeros', name='center_y')
        y, x = tf.meshgrid(tf.range(H, dtype=tf.float32), tf.range(W, dtype=tf.float32))
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])
        dx = self.center_x * W
        dy = self.center_y * H
        self.r = tf.sqrt(tf.square(x - W/2 - dx) + tf.square(y - H/2 - dy)) + 1e-6
        self.theta = tf.atan2(y - H/2 - dy, x - W/2 - dx)
        self.log_r = tf.math.log(self.r) / tf.math.log(tf.reduce_max(self.r))
        self.theta = (self.theta + 2*np.pi) % (2*np.pi)
        self.r_weight = 1.0 / (self.log_r + 1.0)
        super(EnhancedPolarAttention, self).build(input_shape)

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], self.H, self.W, x.shape[3]
        N = H * W
        x_flat = tf.reshape(x, [B, N, C])
        qkv = self.projection(x_flat)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, self.key_dim // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = tf.matmul(q, k, transpose_b=True)
        attn = attn * tf.cast(1 / tf.math.sqrt(tf.cast(self.key_dim // self.num_heads, tf.float32)), attn.dtype)
        r_weights = tf.reshape(self.r_weight, [1, 1, 1, N])
        attn = attn * r_weights
        theta_diff = tf.expand_dims(self.theta, 0) - tf.expand_dims(self.theta, 1)
        theta_sim = tf.cos(theta_diff)
        theta_sim = tf.expand_dims(tf.expand_dims(theta_sim, 0), 0)
        attn = attn * theta_sim
        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B, N, self.key_dim])
        x = self.final_proj(x)
        return tf.reshape(x, [B, H, W, self.key_dim])

def build_ppn_model(h=32, w=32, patch_size=8, embedding_dim=32, num_heads=2, transformer_layers=2):
    num_patches = (h // patch_size) * (w // patch_size)
    
    inputs = Input(shape=(h, w, 1))
    x = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))(inputs)

    # Local Dependencies Branch
    x_vit = tf.keras.layers.Lambda(lambda x: tf.image.extract_patches(
        images=x,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    ))(x)
    x_vit = Reshape((num_patches, patch_size * patch_size * 3))(x_vit)
    encoded_patches = PatchEncoder(num_patches, embedding_dim)(x_vit)

    for _ in range(transformer_layers):
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x1, x1)
        x2 = LayerNormalization(epsilon=1e-6)(attention_output + x1)
        x3 = Dense(embedding_dim, activation='relu')(x2)
        encoded_patches = x3 + x2

    vit_features = LayerNormalization(epsilon=1e-6)(encoded_patches)
    vit_features = Flatten()(vit_features)
    vit_features = Dense(2048, activation='relu')(vit_features)
    vit_features = Reshape((4, 4, 128))(vit_features)

    # Non-local Dependencies Branch
    x_polar = x
    for _ in range(transformer_layers):
        x_polar = LayerNormalization(epsilon=1e-6)(x_polar)
        x_polar = EnhancedPolarAttention(num_heads=num_heads, key_dim=embedding_dim)(x_polar)
        x_polar = LayerNormalization(epsilon=1e-6)(x_polar)
        x_polar = Conv2D(embedding_dim, 1, activation='relu')(x_polar)

    polar_features = Conv2D(128, 1, activation='relu')(x_polar)
    polar_features = AveragePooling2D(pool_size=(8, 8))(polar_features)

    # Feature Fusion
    combined_features = Concatenate()([vit_features, polar_features])
    combined_features = Conv2D(128, 1, activation='relu')(combined_features)

    # Decoder
    def decoder_block(inputs):
        x = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        return Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

    decoded1 = decoder_block(combined_features)
    decoded2 = decoder_block(combined_features)

    return Model(inputs, [decoded1, decoded2])