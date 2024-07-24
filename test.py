import timeit
from Model.ViViT import ModelLoader

model_path = './Saved_model/ViViT_3July_2.keras'
vid_path = './Dataset/Test_dataset/abnormal/Fall53_Cam3_cutup.avi'
st = timeit.default_timer()
loader = ModelLoader(model_path)
model = loader.load_model()
output,pred,label = loader.pred(model,vid_path)
et = timeit.default_timer()
print(f'Time take: {et-st} seconds')