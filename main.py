import streamlit as st
import os
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import timeit

# Model packages
from Model.ViViT import ModelLoader

st.set_page_config(layout='wide')
# Define the file path to your image
# image_path = './Icon/Human_body_logo.png'

#NOTE: change this links to github.com
st.markdown("""
<h2 style='text-align: left; font-size: 15px;'>
For more information or looking for collaboration please visit this 
<a href="https://huggingface.co/spaces/hysts/ViTPose_video" target="_blank">link</a>.
</h2>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; font-size: 40px;color: black;'> 🏃‍♂️Human Anomaly Detection - Manufacturing Safety Solution</h1>", 
            unsafe_allow_html=True)

input_video, result_col = st.columns(2,gap='large')

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
            video_filename = os.path.join(save_path,upload_vid.name)
            with open(video_filename,'wb') as f:
                f.write(video_file)
            st.success(f'Svaed video: {video_filename}')

    if upload_vid:
        # Choose model
        model_choices = ['Select a model', 'ViViT', 'VP-GRU', 'CNN-GRU (Pre-trained)']
        model_choice = st.selectbox('Choose One Model:', model_choices)



# Reuslt columns
with result_col:
    st.markdown("<h2 style='text-align: center; font-size: 25px; color: black;'>Result</h2>", 
                unsafe_allow_html=True)
    if upload_vid:
        #TODO: Call model..
        if model_choice == 'Select a model':
            st.write('No model selected.')
        else:
            st.write('You have selected:', model_choice)
            if model_choice == 'ViViT':
                ViViT_model_path = './Saved_model/ViViT_3July_2.keras'
                loader = ModelLoader(ViViT_model_path)
                model = loader.load_model()
                output,pred,label = loader.pred(model,video_filename)
                # store valie in to a pandas
                st.write(f'Output: {output}')
                st.write(f'Prediction: {pred}')
                st.write(f'Label: {label}')
                table_data = {
                    'Metric': ['Output', 'Prediction', 'Label'],
                    'Value': [output, pred, label]
                }

                # Result table
                df_table = pd.DataFrame(table_data)
                
                fig = go.Figure(data=[go.Table(
                    header=dict(values=['Metric', 'Value'],
                                fill_color='paleturquoise',
                                align='left'),
                    cells=dict(values=[df_table['Metric'], df_table['Value']],
                            fill_color='lavender',
                            align='left'))
                ])
                
                fig.update_layout(
                    title='Model Prediction Results',
                    width=500,
                    height=300
                )
                
                st.plotly_chart(fig)

        # Crate a plotly here
        df = pd.DataFrame({
            'Category': ['A', 'B', 'C', 'D'],
            'Values': [10, 23, 17, 5]
        })
        
        # Create a Plotly bar chart
        fig = px.bar(df, x='Category', y='Values', title='Dummy Data Bar Chart')
        
        # Display the Plotly chart
        st.plotly_chart(fig)