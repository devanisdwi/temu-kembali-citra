import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# for preprocessing & models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

st.set_page_config(page_title='TKC A: Devanis Dwi Sutrisno', page_icon='🧕', layout='wide')

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        img = img.resize((224, 224))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)

fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("D://KULIAH//SEMESTER 7//temu-kembali-citra//feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("D://KULIAH//SEMESTER 7//temu-kembali-citra//busana_muslim") / (feature_path.stem + ".jpg"))
features = np.array(features)

st.markdown("""
    # Women Moslem Fashion 🧕
    Analysis of Women's Muslim Fashion that has RED and NAVY colors based on the following categories:
    
    1. Blouse
    2. Dresses
""")

uploaded_file = st.file_uploader("Upload a Woman Moslem Fashion...", type="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1,1])
    with col1:
        # Extract its features
        query = fe.extract(img)

        # Calculate the similarity (distance) between images
        dists = np.linalg.norm(features - query, axis=1)

        # Extract 12 images that have highest distance
        ids = np.argsort(dists)[12:]
        scores = [(dists[id], img_paths[id]) for id in ids]

        # Visualize the query
        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.title('Query')
        plt.imshow(img)
        fig.tight_layout()
        st.pyplot(plt)

    with col2:
        # Visualize the result
        axes = []
        fig = plt.figure(figsize=(8,8))

        for a in range(4*3):
            score = scores[a]
            axes.append(fig.add_subplot(4, 3, a+1))
            subplot_title=str(score[0])
            axes[-1].set_title(subplot_title)  
            plt.axis('off')
            plt.imshow(Image.open(score[1]))
            
        fig.tight_layout()
        st.pyplot(plt)
    

