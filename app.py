import streamlit as st
import torch
import requests
from io import BytesIO
import numpy as np
from PIL import Image

# Función para descargar el archivo desde Google Drive
def download_file_from_google_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        st.error("Error al descargar el archivo desde Google Drive")
        return None

# ID del archivo de Google Drive
file_id = '1GcaNza4l5ozH3Z8t5fMGAmpZMCw8yACp'  # Reemplazar con el ID real del archivo en Google Drive

# Descargar el archivo .pth desde Google Drive
st.write("Cargando el modelo desde Google Drive...")
model_file = download_file_from_google_drive(file_id)

if model_file:
    try:
        # Intentar cargar el modelo con weights_only=False
        model = torch.load(model_file, weights_only=False)
        model.eval()
        st.write("Modelo cargado exitosamente desde Google Drive.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
else:
    st.error("No se pudo descargar el modelo.")

# Configurar la interfaz de Streamlit
st.title('Clasificación de Imágenes con ResNet')
st.write('Sube una imagen para clasificarla.')

# Cargar imagen
uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "png", "jpeg", "bmp"])
if uploaded_file is not None:
    # Mostrar imagen
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida.', use_column_width=True)

    # Convertir imagen a RGB si es necesario
    image = image.convert("RGB")

    # Preprocesamiento de la imagen (ajustar según sea necesario para tu modelo)
    image = np.array(image)
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)  # Agregar una dimensión para el batch

    # Hacer la predicción
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        st.write(f"Predicción: {predicted.item()}")
