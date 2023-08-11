# video_analysis.py
import cv2
import numpy as np

class ColorObject:
    tolerance = 20

    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.x = []
        self.y = []
        self.area = []

    def process_frame(self, frame):
        mask = np.zeros_like(frame[:, :, 0])
        lower_bound = np.array([self.color[0] - ColorObject.tolerance, 
                                self.color[1] - ColorObject.tolerance, 
                                self.color[2] - ColorObject.tolerance])
        upper_bound = np.array([self.color[0] + ColorObject.tolerance, 
                                self.color[1] + ColorObject.tolerance, 
                                self.color[2] + ColorObject.tolerance])
        within_range = cv2.inRange(frame, lower_bound, upper_bound)
        mask[within_range > 0] = 255

        output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        num_labels, labels, stats, centroids = output

        if num_labels > 1:
            areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
            max_label = np.argmax(areas)
            self.x.append(centroids[max_label + 1][0])
            self.y.append(centroids[max_label + 1][1])
            self.area.append(areas[max_label])
        else:
            self.x.append(None)
            self.y.append(None)
            self.area.append(None)

class VideoAnalyzer:
    def __init__(self, object_defs):
        self.objects = {name: ColorObject(name, color) for name, color in object_defs.items()}

    def process_video(self, input_file):
        cap = cv2.VideoCapture(input_file)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for obj in self.objects.values():
                obj.process_frame(frame)

        cap.release()
        return self.objects