from fastapi import FastAPI
from pydantic import BaseModel
import torch, cv2, os, yaml, sys, types, tempfile, requests
import numpy as np
from utils.util import *
import nn
from dotenv import load_dotenv

load_dotenv()

nets = types.ModuleType("nets")
nets.nn = nn
sys.modules["nets"] = nets
sys.modules["nets.nn"] = nn

WEIGHT_DIR = os.getenv("WEIGHT_PATH", "./weights")
WEIGHT = os.path.join(WEIGHT_DIR, "v11_l.pt")
DATA_YAML = os.getenv("DATA_YAML_PATH", "./utils/args.yaml")
CONF_THR = float(os.getenv("CONF_THR", 0.25))
IOU_THR = float(os.getenv("IOU_THR", 0.05))
INPUT_SIZE = (640, 640)
PORT = int(os.getenv("PORT", 9100))
HOST = os.getenv("HOST", "0.0.0.0")

app = FastAPI()

ckpt = torch.load(WEIGHT, map_location="cpu", weights_only=False)
if "model" not in ckpt:
    raise RuntimeError("Checkpoint missing model key")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ckpt["model"].eval().float().to(device)

with open(DATA_YAML, "r") as f:
    names = yaml.safe_load(f)["names"]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def preprocess_image(image_path, new_shape=(640, 640)):
    im0 = cv2.imread(image_path)
    assert im0 is not None, f"Image not found: {image_path}"
    h0, w0 = im0.shape[:2]
    img, gain, pad = letterbox(im0, new_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0) / 255.0
    return img_tensor, (h0, w0), (gain, pad)

def scale_boxes(img_shape, boxes, im0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img_shape[0] / im0_shape[0], img_shape[1] / im0_shape[1])
        pad = (img_shape[1] - im0_shape[1] * gain) / 2, (img_shape[0] - im0_shape[0] * gain) / 2
    else:
        gain, pad = ratio_pad
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, im0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, im0_shape[0])
    return boxes

class InferenceRequest(BaseModel):
    image_path: str

@app.post("/inference")
async def inference(request: InferenceRequest):
    image_path = request.image_path
    tmp_path = None
    try:
        if image_path.startswith("http"):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            response = requests.get(image_path, timeout=10)
            if response.status_code != 200:
                return {"error": f"Download failed {response.status_code}"}
            tmp_file.write(response.content)
            tmp_file.close()
            tmp_path = tmp_file.name
            image_path = tmp_path
        elif not os.path.exists(image_path):
            return {"error": f"File not found {image_path}"}

        img_tensor, im0_shape, ratio_pad = preprocess_image(image_path)
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            preds = model(img_tensor)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]

        per_class_dict, _, _ = load_thresholds_from_args_yaml(DATA_YAML, default_threshold=CONF_THR)
        detections = non_max_suppression_editable(
            preds,
            confidence_threshold=per_class_dict,
            iou_threshold=IOU_THR,
            default_threshold=CONF_THR
        )[0]

        results = []
        if detections is not None:
            detections[:, :4] = scale_boxes(img_tensor.shape[2:], detections[:, :4], im0_shape, ratio_pad)
            for *xyxy, conf, cls in detections:
                x1, y1, x2, y2 = map(float, xyxy)
                results.append({
                    "label": names[int(cls.item())],
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf.item())
                })

        return {"annotations": results}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("jang:app", host=HOST, port=PORT, reload=False)
