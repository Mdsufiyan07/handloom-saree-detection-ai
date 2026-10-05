import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Handloom Detection")
st.title("Handloom Saree Detection - AI Project")
st.write("Upload a saree image, AI will tell Handloom or Normal")

DEVICE = torch.device("cpu")
classes = ['handloom_sarees', 'normal_sarees']

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

file = st.file_uploader("Choose saree image", type=["jpg","png","jpeg"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()
        conf = torch.softmax(out,1)[0,pred].item()*100
    if classes[pred]=="handloom_sarees":
        st.success(f"Prediction: **HANDLOOM SAREE ✅** ({conf:.1f}% confident)")
    else:
        st.error(f"Prediction: **NORMAL / POWERLOOM SAREE** ({conf:.1f}% confident)")