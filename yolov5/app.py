import cv2
import torch, time, base64
import numpy as np
import boto3
import os
import sys
import json
import datetime
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

CONF_THRESH = 0.25
IOU_THRESH = 0.45
WORK_JPG = '/tmp/in.png'
SAVE_PATH = '/tmp/out.png'
JSON_PATH = '/tmp/out.json'


def handler(event, context):
    s3 = boto3.resource('s3')
    device = select_device('cpu')

    bucket = event['Records'][0]['s3']['bucket']['name']
    inputpath = event['Records'][0]['s3']['object']['key']
    filename = os.path.basename(inputpath)
    outputfolder = os.environ['OUTPUT_PATH']
    outputpath = os.path.join(outputfolder, filename)
    inputfile = s3.Object(bucket, inputpath)
    inputimg = inputfile.get()
    
    data = {}
    data['img']= base64.b64encode(inputimg["Body"].read()).decode('utf-8')
    data['filename'] = filename
    data['flg'] = True 
    output_b64 = execyolo(data,'context')
    # imgファイルのS3へのアップロード
    newobject = s3.Object(bucket, outputpath)
    newobject.upload_file(SAVE_PATH)
   
    # jsonファイルのS3へのアップロード
    outputjsonfolder = os.environ['OUTPUTJSON_PATH']
    outputfileorg = os.path.splitext(os.path.basename(inputpath))[0]
    outputjsonpath = os.path.join(outputjsonfolder, outputfileorg + '.json')    
    newjsonobject = s3.Object(bucket, outputjsonpath)
    newjsonobject.upload_file(JSON_PATH)
    
    response = {'newobject':output_b64['img']}
    return response

def execyolo(event, context):
    device = select_device('cpu')
    filename = event['filename']
    
    # load_model
    model =attempt_load('yolov5s.pt', map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)
    
    img_bin = base64.b64decode(event['img'].encode('utf-8'))
    img_array = np.frombuffer(img_bin,dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    cv2.imwrite(WORK_JPG,img)
    dataset = LoadImages(WORK_JPG, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    
    t0 = time.time()
    
    work = dict()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH, classes=None, agnostic=False)
        t2 = time_synchronized()
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    work[names[int(c)]] = n.item() # JSONで返すための辞書に入れる
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                
        print(f'{s}Done. ({t2 - t1:.3f}s)') 
        cv2.imwrite(SAVE_PATH, im0)
    with open(SAVE_PATH,'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
        
    # Lambdaコンテナ起動時　実行時間を日本時間に調整
    nowdatetime = datetime.datetime.now()
    if event['flg'] :
        td = datetime.timedelta(hours=9)
        nowdatetime = nowdatetime + td
    
    # JSONファイルの書き出し
    outputdict = {'filename' : filename, 'time' : nowdatetime.strftime('%Y/%m/%d %H:%M:%S'), 'content' : work}
    with open(JSON_PATH, mode='wt', encoding='utf-8') as file:
        json.dump(outputdict, file, ensure_ascii=False)
    print(outputdict)
    response = {'img':img_b64, 'dict': outputdict}
    return response

if __name__ == '__main__':
    args = sys.argv
    if 2 > len(args):
        input_file = './data/images/bus.jpg'
    else :
        input_file = args[1]
    output_file = os.path.splitext(os.path.basename(input_file))
    filename = os.path.basename(input_file)
    data = {}
    with open(input_file,'rb') as f:
        data['img']= base64.b64encode(f.read()).decode('utf-8')
    data['filename'] = filename
    data['flg'] = False 
    output_b64 = execyolo(data,'context')
    img_bin = base64.b64decode(output_b64['img'].encode('utf-8'))
    img_array = np.frombuffer(img_bin,dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    outputimg = os.path.join('.', output_file[0] + '.' + output_file[1])
    cv2.imwrite(outputimg,img)

    outputjson = os.path.join('./json', output_file[0] + '.json')
    with open(outputjson, mode='wt', encoding='utf-8') as file:
        json.dump(output_b64['dict'], file, ensure_ascii=False)
    exit()
