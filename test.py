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


# import streamlit as st
# import pandas as pd
# import time
# # Initialize ss_count if it does not exist
# if 'ss_count' not in st.session_state:
#     st.session_state.ss_count = pd.DataFrame(columns=['Model','Prediction_Label','Time_Taken'])

# # Increment ss_count
# model_choices = ['Select a model', 'ViViT', 'VP-GRU', 'CNN-GRU (Pre-trained)']
# model_choice = st.selectbox('Choose One Model:', model_choices,key='model_selections')

# def reset():
#     st.session_state.model_selection = 'Select a model'


    
# st.button("Reset", type="primary",on_click=reset)
# if st.button("Say hello"):
#     with st.spinner('Processing....'):
#         new_records = {'Model':'model_choice','Prediction_Label':'prediction','Time_Taken':'str(duration)'}
#         st.session_state.ss_count = st.session_state.ss_count._append(new_records, ignore_index=True)
#         # here change to almost done
#         st.write('DONE.')
        
# else:
#     st.write("Goodbye")


# import streamlit as st

# # Initialize session state if it doesn't already exist
# if 'selection' not in st.session_state:
#     st.session_state.selection = 'Please Select'

# # Define the selectbox with its key
# choices = st.selectbox('Select:', ['Please Select', 1, 2, 3], key='selection')

# # Define the reset function
# def reset():
#     # Set session state to the default value, which resets the selectbox
#     st.session_state.selection = 'Please Select'

# # Create the reset button
# st.button('Reset', on_click=reset)

# # Optional: Handle logic based on the selected choice
# if st.session_state.selection == 1:
#     st.write('Selected choice is 1')
