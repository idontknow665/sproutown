from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tensorflow as tf
import uvicorn
import io
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("keras_model.h5", compile=False)

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.get("/")
def inicio():
    return {"mensaje": "API de Mochila Activa para Thunkable"}

@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)
    
    img_resized = cv2.resize(img_array, (224, 224))
    img_final = np.asarray(img_resized, dtype=np.float32)
    
    normalized_image = (img_final / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data = normalized_image

    prediction = model.predict(data)
    index = np.argmax(prediction)
    clase_ganadora = labels[index]
    porcentaje = float(prediction[index])

    return {
        "objeto": clase_ganadora,
        "confianza": f"{porcentaje * 100:.2f}%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
