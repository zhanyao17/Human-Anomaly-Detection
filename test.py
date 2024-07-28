# import timeit
# from Model.ViViT import ModelLoader

# model_path = './Saved_model/ViViT_3July_2.keras'
# vid_path = './Dataset/Test_dataset/abnormal/Fall53_Cam3_cutup.avi'
# st = timeit.default_timer()
# loader = ModelLoader(model_path)
# model = loader.load_model()
# output,pred,label = loader.pred(model,vid_path)
# et = timeit.default_timer()
# print(f'Time take: {et-st} seconds')


import streamlit as st
import pandas as pd
import time
# Initialize ss_count if it does not exist
if 'ss_count' not in st.session_state:
    st.session_state.ss_count = pd.DataFrame(columns=['Model','Prediction_Label','Time_Taken'])

# Increment ss_count


st.button("Reset", type="primary")
if st.button("Say hello"):
    with st.spinner('Processing....'):
        new_records = {'Model':'model_choice','Prediction_Label':'prediction','Time_Taken':'str(duration)'}
        st.session_state.ss_count = st.session_state.ss_count._append(new_records, ignore_index=True)
        time.sleep(3)        
        st.write('DONE.')
else:
    st.write("Goodbye")

print(st.session_state.ss_count)