from ultralytics import YOLO

# Carrega um modelo YOLOv8 pré-treinado (versão nano é a mais rápida para o primeiro teste)
model = YOLO("yolov8n.pt")

# Inicia o fine-tuning aproveitando a aceleração da GPU (device=0)
results = model.train(
    data="dataset-EPI/data.yaml",
    epochs=15, 
    imgsz=640,
    batch=16,
    device=0,
    workers=8
)