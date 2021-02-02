import torch, time, base64, cv2, numpy as np
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

CONF_THRESH = 0.25
IOU_THRESH = 0.45
WORK_JPG = '/tmp/in.png'
SAVE_PATH = '/tmp/out.png'

def handler(event, context): 
    device = select_device('cpu')
    
    # load_model
    model =attempt_load('yolov5s.pt', map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)
    img_bin = base64.b64decode(event['img'])
    img_array = np.frombuffer(img_bin,dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    cv2.imwrite(WORK_JPG,img)
    dataset = LoadImages(WORK_JPG, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    
    t0 = time.time()
    
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
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        print(f'{s}Done. ({t2 - t1:.3f}s)') 
        cv2.imwrite(SAVE_PATH, im0)
    with open(SAVE_PATH,'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    print("done")
    return img_b64

if __name__ == '__main__':
    input_file = './data/images/bus.jpg'
    data = {}
    with open(input_file,'rb') as f:
        data['img']= base64.b64encode(f.read())
    output_b64 = handler(data,'context')
    img_bin = base64.b64decode(output_b64.encode('utf-8'))
    img_array = np.frombuffer(img_bin,dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    cv2.imwrite('./out.png',img)