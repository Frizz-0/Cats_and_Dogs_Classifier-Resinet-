import torch.nn as nn
import torch
from torchvision import transforms , models
import streamlit as st
import numpy as np
from PIL import Image
import time

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def load_trained_model(model_path,num_classes):

    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.load_state_dict(torch.load(model_path, map_location= device))
    model.to(device)
    model.eval()

    return model

transform = transforms.Compose([
    transforms.Resize((244,244)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(img_path, model):

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():

        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)  # Get probabilities
        _,pred = torch.max(output,1)

         # Print probabilities for debugging
        print(f"Probabilities: Cat={probabilities[0][0]:.4f}, Dog={probabilities[0][1]:.4f}")
        label = ("Cat" if pred.item() == 0 else "Dog")
        confidence = (f'{probabilities[0][0]:.4f}' if label == "Cat" else f"{probabilities[0][1]:.4f}" )

        
        return label, confidence

model = load_trained_model(r'../models/CatsnDogs.pt',num_classes=2)
# result = predict_image(r'x:\Python\ML\Projects\Cats_and_Dogs\images\00.jpg',model)
# print(f'Prediction : {result}')


#Streamlit UI

st.set_page_config(
    page_title="Cats vs Dogs AI",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

html, body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.title-text {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f7971e, #ffd200);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    animation: fadeIn 1s ease-in-out;
}

.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #dcdcdc;
    margin-bottom: 30px;
    animation: fadeIn 1.1s ease-in-out;
}

.stProgress > div > div > div > div {
    background-color: #fffff;
}
            
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
    animation: fadeIn 0.8s ease-in-out;
}

.result-cat {
    color: #f7971e;
    font-size: 2rem;
    font-weight: bold;
    animation: fadeIn 1s ease-in-out;
}
    
.result-cat-conf {
    color: #f7971e;
    font-size: 1.5rem;
    font-weight: bold;
    margin-top: 8px;
    animation: fadeIn 1s ease-in-out;
}

.result-dog {
    color: #ffcc00;
    font-size: 2rem;
    font-weight: bold;
    animation: fadeIn 1s ease-in-out;
}

.result-dog-conf {
    color: #ffcc00;
    font-size: 1.5rem;
    font-weight: bold;
    animation: fadeIn 1s ease-in-out;
}
                       
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">Cats vs Dogs AI 🐾</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return "your_model_here"

with st.container():
 

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = uploaded_file
        st.image(image, caption="Uploaded Image", width='stretch')


        # if st.button("Predict", use_container_width=True): #Incase if needed a PREDICT Button to predict

        with st.spinner("Thinking..."):
            label, confidence = predict_image(image,model)
 
            st.divider()

            if label == "Cat":
                # st.markdown(f"<div class='result-cat'>It's a CAT! </div>", unsafe_allow_html=True)  
                st.markdown(f"<div style='display: flex; justify-content: space-between;'><span class='result-cat'>It's a CAT!</span><span class='result-cat-conf'>Confidence = {float(confidence):.3f}%</span></div>", unsafe_allow_html=True)

            else:
                # st.markdown(f"<div class='result-dog'>It's a DOG!</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='display: flex; justify-content: space-between;'><span class='result-dog'>It's a DOG!</span><span class='result-dog-conf'>Confidence = {float(confidence):.3f}%</span></div>", unsafe_allow_html=True)


            st.progress(float(confidence))
            # st.markdown(f"<div class='result-cat'>Confidence = {float(confidence):.3f}%</div>", unsafe_allow_html=True,text_alignment="right",)
            # st.caption(f"Confidence: **{float(confidence):.3f}%**")

    else:
        st.info("Drag & drop or upload a cat or dog image 🐕🐈")
