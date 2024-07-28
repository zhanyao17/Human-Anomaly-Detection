import tensorflow as tf
import keras
import os
import cv2
import numpy as np

class CNN_GRU:
    def __init__(self,model_path) -> None:
        self.model_path = model_path
        self.model = None
    
    # laod model
    def load_GRU(self):
        self.model = tf.keras.models.load_model(self.model_path)
        return self.model
    
    # Download Inception V3
    def load_inception(self):
        IMG_SIZE = 224 # change
        feature_extractor = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        preprocess_input = keras.applications.inception_v3.preprocess_input

        inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")
    
    # load video frames
    def load_all_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {'frames':None,'frames_dim':None,'success':False}
        frames_dims = []
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h,w,c = frame.shape
            frames_dims.append(list([0,h,w,c]))
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
            frames.append(frame)
        cap.release()
        return {'frames':np.asarray(frames),'frames_dim':frames_dims,'success':True}

    # trim video
    def trim_video_frames(self,video,max_frame):
        f,_,_,_ = video.shape
        startf = f//2 - max_frame//2
        return video[startf:startf+max_frame, :, :, :]
    
    
    def prepare_data(self,vid_path,feature_extractor):
        frames = self.load_all_frames(vid_path)
        trimmed_vid = self.trim_video_frames(frames['frames'],40) 
        
        # Initialize temporary arrays for current video
        temp_frame_mask = np.zeros(shape=(1, 40), dtype=bool)
        temp_frame_features = np.zeros(shape=(1, 40, 2048), dtype=np.float32)
        
        # Extract features from frames
        for i, f in enumerate(trimmed_vid):
            expand_f = tf.expand_dims(f, axis=0)
            temp_frame_features[0, i, :] = feature_extractor.predict(expand_f, verbose=0)
            temp_frame_mask[0, i] = 1  # Not masked
        return (temp_frame_features, temp_frame_mask)
    
    def pred(self,data,gru_model):
        label_dict = {0:'Normal', 1:'Abnormal'} 
        output = gru_model.predict([data[0],data[1]])[0]
        pred = np.argmax(output.tolist(),axis=0)
        return label_dict[pred]
        
    

# model_path = './Saved_model/CNN-RNN_26Jul_1.keras'
# vid_path = './Dataset/Test_dataset/abnormal/video_254_flip.avi'
# model = CNN_GRU(model_path)
# gru_model = model.load_GRU()
# inception_model = model.load_incpetion()
# frame_features = model.prepare_data(vid_path,inception_model)
# predictions = model.pred(frame_features,gru_model)
# print(predictions)
