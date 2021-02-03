import base64
import requests
import cv2
import numpy as np
import boto3
import json 
client = boto3.client('lambda')

input_file = './yolov5/data/images/bus.jpg'
data = {}
with open(input_file,'rb') as f:
    data['img']= base64.b64encode(f.read()).decode('utf-8')
    
# response = requests.post(url, json=data)
# img_b64 = response.json()['img']
# img_bin = base64.b64decode(img_b64)
# img_array = np.frombuffer(img_bin,dtype=np.uint8)
# img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
# cv2.imwrite('out.png',img)


res = client.invoke(
    FunctionName='lambda-container-yolov5-function',
    Payload=json.dumps(data)
    )
res2 = res['Payload'].read()# .decode('utf-8')
res3 = json.loads(res2)
img_b64 = res3['img']
img_bin = base64.b64decode(img_b64)
img_array = np.frombuffer(img_bin,dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
cv2.imwrite('out.png',img)