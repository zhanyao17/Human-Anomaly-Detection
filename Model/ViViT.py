import tensorflow as tf
from keras import layers, ops, regularizers
from tensorflow.keras.utils import register_keras_serializable # type: ignore
import cv2
import numpy as np

@register_keras_serializable(package="Custom", name="TubeletEmbedding")
class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim, # embed_dim = 64
            kernel_size=patch_size, # patch_size = (8, 8, 8)
            strides=patch_size, 
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

@register_keras_serializable(package="Custom", name="PositionalEncoder")
class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape # input_shape =(numVid,3920,64)
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1) # Set positions

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions # Concat the position with it token
        return encoded_tokens


class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    
    # Load model from .keras
    def load_model(self):
        # Load the model with custom objects
        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={
                'TubeletEmbedding': TubeletEmbedding,
                'PositionalEncoder': PositionalEncoder
            }
        )
        return self.model

    def get_model(self):
        if self.model is None:
            raise ValueError("Model is not loaded. Call `load_model` first.")
        return self.model
    
    # Perform Resizing to (224,224)
    def load_all_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'frames': None, 'frames_dim': None, 'success': False}

        frames_dims = []
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h, w, c = frame.shape
            frames_dims.append([0, h, w, c])
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
            frames.append(frame)

        cap.release()
        return {'frames': np.asarray(frames), 'frames_dim': frames_dims, 'success': True}

    # Trim videos to 40 frames
    def trim_video_frames(self,video, max_frame):
        '''
        Args:
            video: video (collection of frames)
            max_frame: max number of frames
        '''
        f, _, _, _ = video.shape
        startf = f // 2 - max_frame // 2
        return video[startf:startf + max_frame, :, :, :]

    def preprocess_single_video(self,video):
        video = self.trim_video_frames(video, 40)
        video = tf.image.convert_image_dtype(video, tf.float32)
        return video
    
    # Make prediction
    def pred(self,model,video_path):
        label_dict = {0: 'Normal', 1: 'Abnormal'}
        load_data = self.load_all_frames(video_path)
        if not load_data['success']:
            print('Video is corrupt!!')
            return 0
        else:
            video = load_data['frames']
            preprocessed_video = self.preprocess_single_video(video)
            output = model.predict(tf.expand_dims(preprocessed_video, axis=0))[0]
            # print(output)
            pred = np.argmax(output, axis=0)
            # print(label_dict[pred])
            return output, pred, label_dict[pred] 

"""
initialize in main.py

from Model.ViViT import ModelLoader

model_path = './Saved_model/ViViT_3July_2.keras'
loader = ModelLoader(model_path)
model = loader.load_model()
"""