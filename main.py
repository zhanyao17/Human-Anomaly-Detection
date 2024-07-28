import streamlit as st
import os
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import timeit
from datetime import datetime
import numpy as np

# import SessionState # type: ignore

# Model packages
from Model.ViViT import ViViT
from Model.VP_GRU import VP_GRU
from Model.CNN_GRU import CNN_GRU


st.set_page_config(layout='wide')

# Define the file path to your image
# image_path = './Icon/Human_body_logo.png'

# NOTE: change this links to github.com
st.markdown("""
<h2 style='text-align: left; font-size: 15px;'>
For more information or looking for collaboration please visit this 
<a href="https://huggingface.co/spaces/hysts/ViTPose_video" target="_blank">link</a>.
</h2>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 40px;color: black;'> 🏃‍♂️Human Anomaly Detection - Manufacturing Safety Solution</h1>", 
            unsafe_allow_html=True)

input_video, result_col = st.columns(2, gap='large')

with input_video:
    # Define your text content
    st.markdown("<h2 style='text-align: center; font-size: 25px; color: black;'>Drop Video Here 🎬</h2>", 
                unsafe_allow_html=True)
    upload_vid = st.file_uploader('', type=['mp4', 'mov', 'avi']) 
    # Expander here
    with st.expander('View Video'):
        if upload_vid:
            video_file = upload_vid.read()
            st.video(video_file)

            # Save video
            save_path = 'Uploads'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            video_filename = os.path.join(save_path, upload_vid.name)
            with open(video_filename, 'wb') as f:
                f.write(video_file)
            st.success(f'Saved video: {video_filename}')

    if upload_vid:
        # Choose model
        model_choices = ['Select a model', 'ViViT', 'VP-GRU', 'CNN-GRU (Pre-trained)']
        model_choice = st.selectbox('Choose One Model:', model_choices)


# Result columns
with result_col:
    st.markdown("<h2 style='text-align: center; font-size: 25px; color: black;'>Result 📈</h2>", 
                unsafe_allow_html=True)
    
    if upload_vid:
        # Initialize result DataFrame in session state if not already
        if 'df_result' not in st.session_state:
            st.session_state.df_result = pd.DataFrame(columns=['Model', 'Prediction_Label', 'Time_Taken','File_Name','Exec_Date'])
        # Button to refresh data
        if st.button('Refresh Data'):
            st.session_state.df_result = pd.DataFrame(columns=['Model', 'Prediction_Label', 'Time_Taken', 'File_Name','Exec_Date'])
            st.query_params(refresh='true')
        
        # Call model
        if model_choice == 'Select a model':
            st.write('No model selected.')
        else:

            st.write('Now presenting result for :', model_choice)
            if model_choice == 'ViViT':
                with st.spinner('Processing....'):
                    ViViT_model_path = './Saved_model/ViViT_3July_2.keras'
                with st.spinner('Loading model....'):
                    st_time = timeit.default_timer()
                    loader = ViViT(ViViT_model_path)
                    model = loader.load_model()
                with st.spinner('Predicting....'):
                    output, pred, label = loader.pred(model, video_filename)
                    duration = round(timeit.default_timer() - st_time, 3)
                    new_record = {'Model': model_choice, 'Prediction_Label': label, 'Time_Taken': str(duration),'File_Name':video_filename, 'Exec_Date':str(datetime.now())}
                    st.session_state.df_result = st.session_state.df_result._append(new_record, ignore_index=True)

            elif model_choice == 'VP-GRU':
                with st.spinner('Processing....'):
                    model_path = './Saved_model/VP-GRU_25Jul-97.keras'
                    loader = VP_GRU(model_path)
                with st.spinner('Loading nodel....'):
                    gru_model = loader.load_GRU()
                    vp_model = loader.load_ViT()
                    st_time = timeit.default_timer()
                    # extract human pose
                with st.spinner('Extracting human pose....'):
                    key_frames = loader.prepare_data(video_filename, vp_model) 
                    # precheck on the pose extracted
                    print((np.array(key_frames).shape)[0])
                with st.spinner('Predicting....'):
                    if (np.array(key_frames).shape)[0] == 40:
                        pred = loader.pred(gru_model, key_frames)
                    else:
                        pred = 'NaN'
                    duration = round(timeit.default_timer() - st_time, 3)
                    new_record = {'Model': model_choice, 'Prediction_Label': pred, 'Time_Taken': str(duration),'File_Name':video_filename,'Exec_Date':str(datetime.now())}
                    st.session_state.df_result = st.session_state.df_result._append(new_record, ignore_index=True)

            elif model_choice == 'CNN-GRU (Pre-trained)':
                with st.spinner('Processing....'):
                    model_path = './Saved_model/CNN-RNN_26Jul_1.keras'
                    model = CNN_GRU(model_path)
                with st.spinner('Loading nodel....'):
                    gru_model = model.load_GRU()
                    inception_model = model.load_inception()
                with st.spinner('Extracting spatial features....'):
                    st_time = timeit.default_timer()
                    # extract spatial features
                    frame_features = model.prepare_data(video_filename, inception_model)
                with st.spinner('Predicting....'):
                    prediction = model.pred(frame_features, gru_model)
                    duration = round(timeit.default_timer() - st_time, 3)
                    new_record = {'Model': model_choice, 'Prediction_Label': prediction, 'Time_Taken': str(duration),'File_Name':video_filename,'Exec_Date':str(datetime.now())}
                    st.session_state.df_result = st.session_state.df_result._append(new_record, ignore_index=True)

            
            # Display output result here in table
            sorted_df = st.session_state.df_result.sort_values(by='Exec_Date',ascending=False)
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(sorted_df.columns),
                            fill_color='rgb(240,242,246)',
                            align='center',
                            font=dict(color='black', size=19, weight='bold')
                            ),
                cells=dict(values=[sorted_df[col] for col in sorted_df.columns],
                        #    fill_color='rgb(239,246,252)',
                            align='center',
                            font=dict(color='black', size=16)
                           ))
            ])

            fig.update_layout(width=800, height=400)
            st.plotly_chart(fig)

            # Display average speed time 
            mean_df = st.session_state.df_result
            mean_df['Time_Taken'] = pd.to_numeric(mean_df['Time_Taken'], errors='coerce')
            mean_time_taken = mean_df.groupby('Model')['Time_Taken'].mean().reset_index()

            fig1 = px.bar(mean_time_taken, x='Model', y='Time_Taken', color='Model', title='Average Time Taken by Each Model',
                          labels={'Time_Taken':'Mean Time Taken'})
            st.plotly_chart(fig1)

            # Display detection speed result here in bar charts
            fig = px.bar(st.session_state.df_result, x='Model', y='Time_Taken', color='Model', title='Cumulated Time Taken by Each Model')
            st.plotly_chart(fig)

