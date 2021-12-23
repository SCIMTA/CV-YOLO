import time
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import cv2
from yolov5 import yolo_model,detech_frame
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
    try:
        contents = await files.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = detech_frame(img,yolo_model)
        img_path = f"./detected_image/{str(time.time())}.jpg"
        cv2.imwrite(img_path,img)
        return FileResponse(img_path)
    except Exception as e:
        print(f'err: {e}')
        return on_fail(e.__str__())