# Human-Anomaly-Detection
## Overview
- TODO: Add in Abstract

## Some Key terms
- ViViT (Video Vission Transformer)
- VP-GRU (ViTPose GRU)
- CNN-GRU (Inception V3 Pretrained)
- Data set: [kaggle link](https://www.kaggle.com/datasets/ngoduy/dataset-video-for-human-action-recognition)
- Train environment: T4 GPU x2 (32 GB ram in total)
- GUI interface framework: streamlit


# Folders Breakdown
- `Dataset`: Include the test and training dataset been used. (NOTES: Test inside this folder is for demo)
- `Model`: Essential code for post preprocessing on model loading
- `Saved_model` Ready for loading (.keras format)
- `Uploads`: For user to upload `test` files from the interface

# Setup
- `git clone` this repo
- `pip install -r requirements.txt`
- Make sure you have cloned `easy_ViTPose` if no here is the [links to access](https://github.com/JunkyByte/easy_ViTPose)
    - Read thru the `./Model/VP_GRU.py` some of the modification should be done while first time running it
    
# Resutl & Discussion
- TODO: Add in result disucssion

# Deployment
- TODO: front end design

# TODO
- Documentation
- Rewrite setup 