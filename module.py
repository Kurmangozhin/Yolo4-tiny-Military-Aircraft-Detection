import numpy as  np
from glob import glob
import cv2, os, random, colorsys, itertools, argparse, time, functools


def display_process_time(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        s1 = time.time()
        res = func(*args, **kwargs)
        s2 = time.time()
        print('%s process time %f s' % (func.__name__, (s2-s1)/60))
        return res
    return decorated


parser = argparse.ArgumentParser("YOLO4-TINY Inference")
parser.add_argument('--i', type = str, required = False, default = False, help='input--img')
parser.add_argument('--v', type = str, required = False, default = False, help='input--video')
parser.add_argument('--o', type = str, required = True, default = False)


class Recognition(object):
    def __init__(self, path_cls:str, weghts:str, cfg:str):
        self.class_labels = self.read_classes(path_cls)
        self.colors()        
        yolo_model = cv2.dnn.readNet(weghts, cfg)
        self.model = cv2.dnn_DetectionModel(yolo_model)
        self.model.setInputParams(size = (416, 416), scale=1/255., swapRB = True, crop=False)
        self.CONFIDENCE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.4
        # cuda opencv dnn
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
   
    def read_classes(self, path:str):
        with open(f'{path}', 'r') as f:
             class_labels = f.readlines()
        class_labels = [cls.strip() for cls in class_labels]
        return class_labels
    
    
    def colors(self):
        hsv_tuples = [(x / len(self.class_labels), 1., 1.) for x in range(len(self.class_labels))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        class_colors = list(map(lambda x:(int(x[0]*255), int(x[1]*255), int(x[2]*255)), colors))
        np.random.seed(43)
        np.random.shuffle(colors)
        np.random.seed(None)
        self.class_colors = np.tile(class_colors,(16,1))


    def draw_line(self, image, x, y, x1, y1, color, l = 25, t = 2):
        cv2.line(image, (x, y), (x + l, y), color, t)
        cv2.line(image, (x, y), (x, y + l), color, t)    
        cv2.line(image, (x1, y), (x1 - l, y), color, t)
        cv2.line(image, (x1, y), (x1, y + l), color, t)    
        cv2.line(image, (x, y1), (x + l, y1), color, t)
        cv2.line(image, (x, y1), (x, y1 - l), color, t)   
        cv2.line(image, (x1, y1), (x1 - l, y1), color, t)
        cv2.line(image, (x1, y1), (x1, y1 - l), color, t)    
        return image


    def draw_visual(self, image, boxes_out, scores_out, classes_out, lines = True):
        _box_color = [255,0,0]
        for i, c in reversed(list(enumerate(classes_out))):
            predicted_class = self.class_labels[c]
            box = boxes_out[i]
            score = scores_out[i]
            predicted_class_label = '{}: {:.2f}%'.format(predicted_class, score*100)
            box_color = self.class_colors[c]
            box_color = list(map(int, box_color))
            box = list(map(int, box))
            x_min, y_min, h, w = box
            x_max, y_max  = x_min + h, y_min + w
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 2)      
            if lines: self.draw_line(image, x_min, y_min, x_max, y_max, _box_color)
            cv2.putText(image, predicted_class_label, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _box_color, 1)
        return image
  
    
    def detection(self, image_path:str):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR);
        classes, scores, boxes = self.model.detect(image, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        image_pred = self.draw_visual(image, boxes, scores, classes)
        return image
    
    @display_process_time
    def detection_video(self, path:str, output_path:str,fps = 25):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        frame_height, frame_width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width,frame_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            classes, scores, boxes = self.model.detect(frame, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
            output = self.draw_visual(frame, boxes, scores, classes)
            output  = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            out.write(output)
        out.release()


if __name__=='__main__':
    args = parser.parse_args()
    opt = {'path_cls':'weights/classes.txt','weghts':'weights/air_best.weights','cfg':'weights/air.cfg'}
    cls = Recognition(**opt)
    if args.v:
       cls.detection_video(args.v, args.o) 
    else:
        image = cls.detection(args.i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(args.o, image)



        
    
    


