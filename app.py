

# app.py
import plotly.graph_objects as go
import plotly.subplots as sp
import gradio as gr

from video_analysis import VideoAnalyzer

class Plotter:
    @staticmethod
    def plot_area(results):
        fig = go.Figure()

        for color_obj in results.values():
            fig.add_trace(go.Scatter(x=list(range(len(color_obj.area))), y=color_obj.area, mode='lines', name=f'{color_obj.name} area'))

        fig.show()

    @staticmethod
    def plot_coordinates(results):
        fig = sp.make_subplots(rows=4, cols=1, subplot_titles=list(results.keys()))

        row_col = [(1, 1), (2, 1), (3, 1), (4, 1)]
        for idx, color_obj in enumerate(results.values()):
            r, c = row_col[idx]
            fig.add_trace(go.Scatter(x=list(range(len(color_obj.x))), y=color_obj.x, mode='lines', name=f'{color_obj.name} x'), r, c)
            fig.add_trace(go.Scatter(x=list(range(len(color_obj.y))), y=color_obj.y, mode='lines', name=f'{color_obj.name} y'), r, c)

        fig.update_layout(title_text="XY-coordinates of the colors over time")
        fig.show()

def interface(input_file):
    object_defs = {
        'R_leg': (242, 110, 176),
        'L_leg': (250, 230, 250),
        'R_hand': (143, 209, 66),
        'L_hand': (133, 237, 148),
    }

    analyzer = VideoAnalyzer(object_defs)
    results = analyzer.process_video(input_file.name)
    Plotter.plot_area(results)
    Plotter.plot_coordinates(results)

inputs = gr.inputs.File(label="Upload a video")
outputs = gr.outputs.Textbox()
gr.Interface(fn=interface, inputs=inputs, outputs=outputs).launch()