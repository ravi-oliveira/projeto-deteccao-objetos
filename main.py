from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="API de Detecção de EPIs")

# Carrega o modelo fine-tuned
model = YOLO("runs/detect/train-2/weights/best.pt")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Lê a imagem enviada via requisição
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Realiza a inferência
    results = model(image)
    
    detections = []
    
    # Extrai os dados do objeto results do YOLO
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Coordenadas da bounding box
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            
            # Confiança e ID da classe
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Monta o dicionário no formato exigido
            detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 2),
                "box": {
                    "x_min": round(x_min, 2),
                    "y_min": round(y_min, 2),
                    "x_max": round(x_max, 2),
                    "y_max": round(y_max, 2)
                }
            })

    return {"status": "success", "detections": detections}