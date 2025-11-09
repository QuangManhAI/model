from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import os, tempfile, requests

app = FastAPI()

MODEL_PATH = "/home/quangmanh/Documents/pineline/model/src/internal_assets/weights/yolo12s.pt"
model = YOLO(MODEL_PATH)
NAMES = model.names

class InferenceRequest(BaseModel):
    image_path: str

@app.post("/inference")
async def inference(request: InferenceRequest):
    image_path = request.image_path
    tmp_path = None
    try:
        # hỗ trợ URL
        if image_path.startswith("http"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            r = requests.get(image_path, timeout=10)
            r.raise_for_status()
            tmp.write(r.content)
            tmp.close()
            tmp_path = tmp.name
            image_path = tmp_path
        elif not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}

        # chạy YOLO
        results = model(image_path, imgsz=640)
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy()

        anns = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(float, boxes[i])
            conf = float(confs[i])
            cls = int(clss[i])
            anns.append({
                "label": NAMES[cls],
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })

        # chỉ trả JSON để BE lưu
        return {"annotations": anns}

    except Exception as e:
        return {"error": str(e)}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9100)
