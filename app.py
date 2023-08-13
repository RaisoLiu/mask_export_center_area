# # app.py


class Plotter:
    @staticmethod
    def plot_area(results):
        fig = go.Figure()

        for color_obj in results.values():
            fig.add_trace(go.Scatter(x=list(range(len(color_obj.area))),
                          y=color_obj.area, mode='lines', name=f'{color_obj.name} area'))

        return fig

    @staticmethod
    def plot_coordinates(results):
        num = len(results)
        fig = sp.make_subplots(
            rows=num, cols=1, subplot_titles=list(results.keys()))

        row_col = [(i+1, 1) for i in range(num)]
        for idx, color_obj in enumerate(results.values()):
            r, c = row_col[idx]
            fig.add_trace(go.Scatter(x=list(range(len(color_obj.x))),
                          y=color_obj.x, mode='lines', name=f'{color_obj.name} x'), r, c)
            fig.add_trace(go.Scatter(x=list(range(len(color_obj.y))),
                          y=color_obj.y, mode='lines', name=f'{color_obj.name} y'), r, c)

        fig.update_layout(title_text="XY-coordinates of the colors over time")
        return fig




import plotly.graph_objects as go
import plotly.subplots as sp
import gradio as gr

import cv2
import os
import numpy as np
import pandas as pd
from video_analysis import VideoAnalyzer
from media_processor import get_meta_from_video


def object_selector(frame, evt:gr.SelectData):
    point = (evt.index[0], evt.index[1])

    return frame[point[1], point[0]], "click: " + str(point) + ", color: " + str(frame[point[1], point[0]])

def add_obj(obj, objs):
    objs.append(obj)

    return objs, f"obj list count: {len(objs)}"

def clean_objs():
    return []

def plot_traces(input_video, objects, progress=gr.Progress()):
    analyzer = VideoAnalyzer(objects)
    results = analyzer.process_video(input_video.name, progress)


    return Plotter.plot_coordinates(results), Plotter.plot_area(results), result_2_csv(input_video, results), "plot_traces"

def result_2_csv(input_video, results):
    num = len(results)
    n_time = max([len(it.area) for it in results.values()])

    arr = np.zeros((n_time, num*3+1))

    arr[:, 0] = np.arange(n_time)
    cols = ["index"]
    for i, color_obj in enumerate(results.values()):
        arr[:, i*3+1] = color_obj.x
        arr[:, i*3+2] = color_obj.y
        arr[:, i*3+3] = color_obj.area
        cols.append(color_obj.name + ".x")
        cols.append(color_obj.name + ".y")
        cols.append(color_obj.name + ".area")

    
    
    df = pd.DataFrame(arr, columns = cols)
    os.makedirs("./export/", exist_ok=True)

    export_name = "./export/" + input_video.name.split('/')[-1].split('.')[0] + '.csv'
    print(export_name)
    df.to_csv(export_name, index=False)
    return export_name

    


app = gr.Blocks()
with app:
    origin_frist_frame = gr.State(None)
    objects = gr.State([])
    selected_obj = gr.State(None)
    gr.Markdown(
        '''
        <div style="text-align:center;">
            <span style="font-size:3em; font-weight:bold;">Mask Video Analysis Tool</span>
        </div>
        '''
    )
    with gr.Row():
        with gr.Column(scale=0.5):

            input_video = gr.File(label='Input video', height=100)

            result_frame = gr.Image(
                label='Crop result of first frame', height=550, interactive=True)

            with gr.Row():

                new_object_button = gr.Button(
                    value="Add new object",
                    interactive=True
                )
                reset_button = gr.Button(
                    value="Reset",
                    interactive=True,
                )
                plot_trace = gr.Button(
                    value="Plot",
                    interactive=True,
                )
                log = gr.Textbox(
                    label="log",
                )
            pass

        with gr.Column(scale=0.5):
            trace_plot = gr.Plot(label='Trace result')
            area_plot = gr.Plot(label='Area result')
            output_file = gr.File(label="Download CSV file")
            pass

        # Back-end
        input_video.change(
            fn=get_meta_from_video,
            inputs=[
                input_video,
            ],
            outputs=[
                origin_frist_frame, result_frame
            ]
        )
        result_frame.select(
            fn=object_selector,
            inputs=[
                origin_frist_frame,
            ],
            outputs=[
                selected_obj, log,
            ]
        )
        new_object_button.click(
            fn=add_obj,
            inputs=[
                selected_obj, objects
            ],
            outputs=[
                objects, log
            ]
        )
        reset_button.click(
            fn=clean_objs,
            outputs=[
                objects,
            ]
        )
        plot_trace.click(
            fn=plot_traces,
            inputs=[
                input_video, objects
            ],
            outputs=[
                trace_plot, area_plot, output_file, log
            ]
        )


if __name__ == "__main__":
    app.queue(concurrency_count=5)
    app.launch(debug=True, share=False,
               server_name="0.0.0.0", server_port=10002).queue()
