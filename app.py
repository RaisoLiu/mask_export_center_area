import cv2
import numpy as np
import plotly.graph_objects as go
import gradio as gr

def extract_first_frame(input_file):
    cap = cv2.VideoCapture(input_file)
    ret, frame = cap.read()
    cap.release()
    return frame

def get_color_range_from_pixel(frame, x, y, threshold=40):
    color_pixel = frame[y, x]
    lower_bound = [max(val - threshold, 0) for val in color_pixel]
    upper_bound = [min(val + threshold, 255) for val in color_pixel]
    return lower_bound, upper_bound

def process_video(input_file, color_positions):
    cap = cv2.VideoCapture(input_file.name)

    first_frame = extract_first_frame(input_file.name)
    colors = {}
    for color_name, (x, y) in color_positions.items():
        colors[color_name] = get_color_range_from_pixel(first_frame, x, y)

    results = {color: {'x': [], 'y': [], 'area': []} for color in colors}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for color, (lower, upper) in colors.items():
            mask = cv2.inRange(frame, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                results[color]['x'].append(cx)
                results[color]['y'].append(cy)
                results[color]['area'].append(cv2.contourArea(max_contour))

    cap.release()
    return results

def plot_results(results):
    fig = go.Figure()

    for color, data in results.items():
        fig.add_trace(go.Scatter(x=list(range(len(data['x']))), y=data['area'], mode='lines', name=f'{color}_area'))
        fig.add_trace(go.Scatter(x=list(range(len(data['x']))), y=data['x'], mode='lines', name=f'{color}_x'))
        fig.add_trace(go.Scatter(x=list(range(len(data['x']))), y=data['y'], mode='lines', name=f'{color}_y'))

    fig.show()

def interface(input_file, red, blue, green, yellow):
    color_positions = {
        'red': red,
        'blue': blue,
        'green': green,
        'yellow': yellow
    }
    results = process_video(input_file, color_positions)
    plot_results(results)
    return "Processed and plotted successfully!"

inputs = {
    "input_file": gr.inputs.File(label="Upload a video"),
    "red": gr.inputs.ImageClickbox(label="Click on RED color"),
    "blue": gr.inputs.ImageClickbox(label="Click on BLUE color"),
    "green": gr.inputs.ImageClickbox(label="Click on GREEN color"),
    "yellow": gr.inputs.ImageClickbox(label="Click on YELLOW color")
}

outputs = gr.outputs.Textbox()
gr.Interface(fn=interface, inputs=inputs, outputs=outputs, live=True).launch()
