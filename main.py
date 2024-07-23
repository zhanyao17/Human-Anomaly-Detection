import streamlit as st
import os

st.set_page_config(layout='wide')
# Define the file path to your image
# image_path = './icon/logoM.png'

st.markdown("<h1 style='text-align: center; font-size: 40px;color: black;'>Human Anomaly Detection - Manufacturing Safety Solution</h1>", 
            unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; color: black;'>\n</h1>", 
#             unsafe_allow_html=True)

# Forming columns
input_video, col2 = st.columns(2,gap='large')

with input_video:
    # Define your text content
    st.markdown("<h2 style='text-align: center; font-size: 25px; color: black;'>Drop Video Here</h2>", 
                unsafe_allow_html=True)
    upload_vid = st.file_uploader('') # Get video
    # Crate a expander here
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


with col2:
    st.markdown("<h2 style='text-align: center; font-size: 25px; color: black;'>Result</h2>", 
                unsafe_allow_html=True)


# Preview video
