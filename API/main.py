import time
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import cv2
from yolov4 import yolov4_model,detech_frame
app = FastAPI(redoc_url=None)

def on_success(data=None, message='Thành công', status=1):
    if data is not None:
        return {
            'message': message,
            'data': data,
            'status': status,
        }

    return {
        'message': message,
        'status': status
    }


def on_fail(message='Thất bại', status=0):
    return {
        'message': message,
        'status': status
    }

@app.post("/detect")
async def _detect(files: UploadFile = File(...)):
    contents = await files.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = detech_frame(img,yolov4_model)
    img_path = f"./detected_image/{str(time.time())}.jpg"
    cv2.imwrite(img_path,img)
    return FileResponse(img_path)