import time
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import cv2
from yolov5 import yolov5_model,detech_frame_v5
from yolov4 import yolov4_model,detech_frame_v4
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(redoc_url=None)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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

@app.get("/image/{image}")
async def _image(image:str):
    try:
        img_path = f"./detected_image/{image}.jpg"
        return FileResponse(img_path)
    except Exception as e:
        return on_fail(e.__str__())


@app.post("/detect")
async def _detect(files: UploadFile = File(...),ver:int=5):
    try:
        contents = await files.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if ver == 4:
            img = detech_frame_v4(img, yolov4_model)
        else:
            img = detech_frame_v5(img,yolov5_model)
        image_name = str(time.time())
        img_path = f"./detected_image/{image_name}.jpg"
        cv2.imwrite(img_path,img)
        return {
            "image":f'http://localhost:8000/image/{image_name}'
        }
    except Exception as e:
        print(f'err: {e}')
        return on_fail(e.__str__())