# # app.py

import cv2
from tqdm import tqdm

import plotly.graph_objects as go
import plotly.subplots as sp
import gradio as gr


import os
import numpy as np
import pandas as pd
import platform
os_sys = platform.uname().system

from libs.video_analysis import VideoAnalyzer
from libs.media_processor import get_meta_from_video
from libs.plotter import Plotter
from libs.dinov2_latent_generator import DinoV2latentGen
model_cfg = {
    'name': 'dinov2_vitb14_reg',
    'struct': 'dinov2/',
    'path': 'dinov2_vitb14_reg4_pretrain.pth',
}
device = "mps" if os_sys == 'Darwin' else "cuda"
latentGenerator = DinoV2latentGen(model_cfg, device)
latent_dim = latentGenerator.model.embed_dim

import torchvision.transforms as tt
resolution = 518
img2tensor = tt.Compose([
    tt.ToTensor(), # range [0, 255] -> [0.0,1.0]
    tt.Resize((resolution, resolution), antialias=True),
    tt.Normalize(mean=0.5, std=0.2), # range [0.0,1.0] -> [-2.5, 2.5]
])

img2mask = tt.Compose([
    tt.ToTensor(), # range [0, 255] -> [0.0,1.0]
    tt.Resize((resolution, resolution), antialias=True),
])
def object_selector(frame, evt:gr.SelectData):
    point = (evt.index[0], evt.index[1])

    return frame[point[1], point[0]], "click: " + str(point) + ", color: " + str(frame[point[1], point[0]])

def add_obj(obj, objs):
    objs = []
    objs.append(obj)

    return objs, f"obj list count: {len(objs)}"

def clean_objs():
    return []

def get_representation(vin, results, progress):
    cap = cv2.VideoCapture(vin.name)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    latents = []
    for i in progress.tqdm(range(n)):
        ok, image = cap.read()
        if not ok:
            print("[E] Video Reading Error")
            break
        for it in results:
            mask = np.array(results[it].labels[i], dtype=float)
            break
        tmask = img2mask(mask)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = img2tensor(image)
        latent = latentGenerator.single_run(tensor)
        latent = latent.reshape((37, 37, latent_dim))
        small_mask = np.zeros((37, 37))
        small_mask = tmask[0].reshape(resolution//14, 14, resolution//14, 14).sum(axis=(1, 3))
        sum_mask = small_mask.sum()
        result = small_mask[:, :, np.newaxis] * latent
        latent_mask_ave = result.sum(axis=0).sum(axis=0) / sum_mask
        latents.append(latent_mask_ave)

        # if i == 5:
        #     break

    cap.release()
    latents = np.array(latents)
    return latents

def plot_traces(input_video, raw_video, objects, progress=gr.Progress()):
    analyzer = VideoAnalyzer(objects)
    results = analyzer.process_video(input_video.name, progress)
    latents = get_representation(raw_video, results, progress)

    return Plotter.plot_coordinates(results), Plotter.plot_area(results), result_2_csv(input_video, results), result_2_npy(input_video, latents), "plot_traces"

def result_2_npy(input_video, latents):
    os.makedirs("./export/", exist_ok=True)

    export_name = "./export/" + input_video.name.split('/')[-1].split('.')[0] + '.npy'
    print(export_name)
    np.save(export_name, latents)
    return export_name

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

            input_video = gr.File(label='Mask video', height=100)
            raw_video = gr.File(label='Raw video', height=100)
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
            npy_file = gr.File(label="Download NPY file")
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
                input_video, raw_video, objects
            ],
            outputs=[
                trace_plot, area_plot, output_file, npy_file, log
            ]
        )


if __name__ == "__main__":
    app.queue(concurrency_count=5)
    app.launch(debug=True, share=False,
               server_name="0.0.0.0", server_port=10002).queue()
