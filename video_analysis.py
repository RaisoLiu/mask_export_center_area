# 定義四種顏色的範圍
objects = {
    'R_leg': (242, 110, 176),
    'L_leg': (250, 230, 250),
    'R_hand': (143, 209, 66),
    'L_hand': (133, 237, 148),
}
tolerance = 20

import cv2
import numpy as np
import plotly.graph_objects as go
import gradio as gr
def process_video(input_file):
    cap = cv2.VideoCapture(input_file)

    results = {color: {'x': [], 'y': [], 'area': []} for color in objects}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for name, target_color in objects.items():
            mask = np.zeros_like(frame[:,:,0])
            lower_bound = np.array([target_color[0] - tolerance, target_color[1] - tolerance, target_color[2] - tolerance])
            upper_bound = np.array([target_color[0] + tolerance, target_color[1] + tolerance, target_color[2] + tolerance])
            within_range = cv2.inRange(frame, lower_bound, upper_bound)
            mask[within_range > 0] = 255


            output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            num_labels, labels, stats, centroids = output

            
            # 跳過 index 0，因為它是背景


            if num_labels > 1:
                areas = []

                # print(num_labels, centroids)
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]

                    areas.append(area)

                max_label = np.argmax(areas)
                results[name]['x'].append(centroids[max_label+1][0])
                results[name]['y'].append(centroids[max_label+1][1])
                results[name]['area'].append(areas[max_label])
            else:
                results[name]['x'].append(None)
                results[name]['y'].append(None)
                results[name]['area'].append(None)

    cap.release()
    return results
