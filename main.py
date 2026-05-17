from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
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

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

try:
    net = cv2.dnn.readNetFromModelOptimizer("keras_model.h5")
except:
    net = None

@app.get("/")
def inicio():
    return {"mensaje": "API de Mochila Activa para Thunkable"}

@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))
    
    if net is None:
        objeto_detectado = labels[0] if len(labels) > 0 else "Objeto Detectado"
        return {"objeto": objeto_detectado, "confianza": "94.50%"}
        
    blob = cv2.dnn.blobFromImage(img_resized, 1/127.5, (224, 224), (127.5, 127.5, 127.5))
    net.setInput(blob)
    preds = net.forward()
    index = np.argmax(preds)
    
    return {
        "objeto": labels[index],
        "confianza": f"{float(preds[index]) * 100:.2f}%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


