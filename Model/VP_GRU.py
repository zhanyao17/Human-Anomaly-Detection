import tensorflow as tf
import keras
import os
import cv2
import numpy as np
from easy_ViTPose import VitInference
from huggingface_hub import hf_hub_download

class VP_GRU:
    def __init__(self,model_path) -> None:
        self.model_path = model_path
        self.model= None

    # load model
    def load_GRU(self):
        self.model = tf.keras.models.load_model(self.model_path)
        return self.model

    # Download and setting up ViT-Pose (VP)
    def load_ViT(self):
        ''' Define ViTpose & YOLO model params '''
        MODEL_SIZE = 'b' 
        DATASET = 'coco' 

        
        ''' 
        # NOTE: First time setting up pls remove this command block
        ext = '.pth'
        ext_yolo = '.pt'
        MODEL_TYPE = "torch"
        YOLO_SIZE = 's' 

        REPO_ID = 'JunkyByte/easy_ViTPose'
        FILENAME = os.path.join(MODEL_TYPE, f'{DATASET}/vitpose-' + MODEL_SIZE + f'-{DATASET}') + ext
        FILENAME_YOLO = 'yolov8/yolov8' + YOLO_SIZE + ext_yolo
        
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME) # ViT-Pose - Human Pose
        yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO) # YOLO - Bouding box
        '''
        model_path = '/Users/linzhanyao/Project/Human-Anomaly-Detection/easy_ViTPose/models--JunkyByte--easy_ViTPose/snapshots/2757e82adcccda02f9f7fef66e5a115b7be439fe/torch/coco/vitpose-b-coco.pth'
        
        yolo_path = '/Users/linzhanyao/Project/Human-Anomaly-Detection/easy_ViTPose/models--JunkyByte--easy_ViTPose/snapshots/2757e82adcccda02f9f7fef66e5a115b7be439fe/yolov8/yolov8s.pt'

        VP_model = VitInference(model_path, yolo_path, MODEL_SIZE,
                     dataset=DATASET, yolo_size=320, is_video=False)
        return VP_model
    
    # load all frames
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

    # preprocess data
    def prepare_data(self,file_path,VP_models):
        load_data = self.load_all_frames(file_path)
        trimmed_vid = self.trim_video_frames(load_data['frames'],40) # trim vid
        key_frames = []
        for frame in trimmed_vid:
            frame_keypoints = VP_models.inference(frame)
            if 0 in frame_keypoints: # shoudl write this way to prevent 
                key_frames.append(frame_keypoints[0][:,:2])
        return key_frames
    
    # make prediction
    def pred(self,gru,key_frames):
        label_dict = {0:'Normal', 1:'Abnormal'} 
        output = gru.predict(tf.expand_dims(key_frames, axis=0))[0]
        pred = np.argmax(output.tolist(),axis=0)
        return label_dict[pred]


import timeit 
model_path = './Saved_model/VP-GRU_25Jul-97.keras'
loader = VP_GRU(model_path)
gru_model = loader.load_GRU()
vp_model = loader.load_ViT()
st_time_pose = timeit.default_timer()
key_frames = loader.prepare_data('./Dataset/Test_dataset/abnormal/video_254_flip.avi', vp_model) 
duration_pose = round(timeit.default_timer() - st_time_pose, 3)
st_time_pred = timeit.default_timer()
pred = loader.pred(gru_model, key_frames)
duration_pred = round(timeit.default_timer() - st_time_pred, 3)
print(f'Duration for extracting human pose: {duration_pose}')
print(f'Duration for prediction: {duration_pred}')