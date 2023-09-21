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
        self.labels = []

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
            self.labels.append(labels == (max_label+1))
        else:
            self.x.append(np.nan)
            self.y.append(np.nan)
            self.area.append(np.nan)
            self.labels.append(np.zeros_like(mask, dtype=np.bool_))


class VideoAnalyzer:
    def __init__(self, objs):
        
        self.objects = {f"obj_{i+1}": ColorObject(f"obj_{i+1}", color)
                        for i, color in enumerate(objs)}

    def process_video(self, input_file, progress):
        cap = cv2.VideoCapture(input_file)

        num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for obj in self.objects.values():
                obj.process_frame(frame)

            if cnt % 10 == 0:
                progress(cnt / num, desc="Analysis Video")
            cnt += 1

        cap.release()
        return self.objects
