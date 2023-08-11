import cv2
import numpy as np
import plotly.graph_objects as go
import gradio as gr
from video_analysis import process_video
import plotly.subplots as sp


def plot_area(results):
    fig = go.Figure()

    for color, data in results.items():
        fig.add_trace(go.Scatter(x=list(range(len(data['area']))), y=data['area'], mode='lines', name=f'{color} area'))
    
    fig.show()

def plot_coordinates(results):
    fig = sp.make_subplots(rows=4, cols=1, subplot_titles=list(results.keys()))
   

    row_col = [(1, 1), (2, 1), (3, 1), (4, 1)]
    for idx, (color, data) in enumerate(results.items()):
        r, c = row_col[idx]
        fig.add_trace(go.Scatter(x=list(range(len(data['x']))), y=data['x'], mode='lines', name=f'{color} x'), r, c)
        fig.add_trace(go.Scatter(x=list(range(len(data['y']))), y=data['y'], mode='lines', name=f'{color} y'), r, c)

    fig.update_layout(title_text="XY-coordinates of the colors over time")

    fig.show()



def interface(input_file):
    results = process_video(input_file.name)
    plot_area(results)
    plot_coordinates(results)

inputs = gr.inputs.File(label="Upload a video")
outputs = gr.outputs.Textbox()
gr.Interface(fn=interface, inputs=inputs, outputs=outputs).launch()
