from Model.ViViT import ModelLoader

model_path = './Saved_model/ViViT_3July_2.keras'
loader = ModelLoader(model_path)
model = loader.load_model()