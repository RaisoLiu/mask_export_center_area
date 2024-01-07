import json
import gradio as gr
import os
import numpy as np
import pandas as pd
from libs.roi_observer import DinoV2latentGen
from libs.source_video import SourceVideo
from libs.plotter import Plotter
import cv2
from torch.utils.data import Dataset, DataLoader
from libs.media_processor import get_meta_from_video
import torchvision.transforms as tt

resolution = 518
patch_len = resolution // 14
img2tensor = tt.Compose([
    tt.ToTensor(), # range [0, 255] -> [0.0,1.0]
    tt.Resize((resolution, resolution), antialias=True),
    tt.Normalize(mean=0.5, std=0.2), # range [0.0,1.0] -> [-2.5, 2.5]

])

img2mask = tt.Compose([
    tt.ToTensor(), # range [0, 255] -> [0.0,1.0]
    tt.Resize((resolution, resolution), antialias=True),
])


model_config = json.load(open('config/model_config.json', 'r'))


def extrack_roi_mask(frame, roi):
    mask = np.zeros_like(frame[:, :, 0])
    tolerance = 10
    lower_bound = np.array([roi[0] - tolerance,
                            roi[1] - tolerance,
                            roi[2] - tolerance])
    upper_bound = np.array([roi[0] + tolerance,
                            roi[1] + tolerance,
                            roi[2] + tolerance])
    within_range = cv2.inRange(frame, lower_bound, upper_bound)
    mask[within_range > 0] = 255
    output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    num_labels, labels, stats, centroids = output
    if num_labels == 1:
        return False, None, None, None, None
    

    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    max_label = np.argmax(areas)
    x = centroids[max_label + 1][0]
    y = centroids[max_label + 1][1]
    area = areas[max_label]
    roi_labels = labels == (max_label+1)
    return True, x, y, area, roi_labels


class CustomImageDataset(Dataset):
    def __init__(self, source_video, mask_video, select_roi):
        self.source_video = source_video
        self.mask_video = mask_video
        self.select_roi = select_roi
        self.x_list = []
        self.y_list = []
        self.area_list = []


    def __len__(self):
        return min(self.source_video.total_frames, self.mask_video.total_frames)

    def __getitem__(self, index):
        frame = self.source_video.read_by_index(index).to_rgb().to_ndarray()
        mask = self.mask_video.read_by_index(index).to_rgb().to_ndarray()
        ret, x, y, area, mask = extrack_roi_mask(mask, self.select_roi)
        assert ret, f'extrack_roi_mask error at {index}'
        self.x_list.append(x)
        self.y_list.append(y)
        self.area_list.append(area)

        return img2tensor(frame), img2mask(mask)
    


def extract_roi_latent_from_video(observer, source_video, mask_video, batch_size, select_roi, progress):
    latent_list = []
    batch_size = int(batch_size)
    print('batch_size', (batch_size))
    dataset = CustomImageDataset(source_video, mask_video, select_roi)
    print('dataset', len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    print('dataloader', len(dataloader))
    for i, (frames, masks) in enumerate(progress.tqdm(dataloader)):
        patch_feature = observer.batch_run(frames)
        for img, mask in zip(patch_feature, masks):
            latent = img.reshape((patch_len, patch_len, observer.n_feature))
            small_mask = mask.reshape(resolution//14, 14, resolution//14, 14).sum(axis=(1, 3))
            sum_mask = small_mask.sum()
            result = small_mask[:, :, np.newaxis] * latent
            latent_mask_ave = result.sum(axis=0).sum(axis=0) / sum_mask
            latent_list.append(latent_mask_ave)

    return dataset.x_list, dataset.y_list, dataset.area_list, latent_list






def start_fn(mask_video, raw_video, selected_obj, batch_size, progress=gr.Progress()):
    batch_size = int(batch_size)
    os.makedirs('export', exist_ok=True)
    observer = DinoV2latentGen(model_config['dinov2_args'])
    # observer_dim = observer.n_feature

    raw_video = SourceVideo(raw_video.name)
    mask_video = SourceVideo(mask_video.name)
    x_list, y_list, area_list, latent_list = extract_roi_latent_from_video(observer, raw_video, mask_video, batch_size, selected_obj, progress)

    basename = raw_video.video_name.split('.')[0]
    latent_path = os.path.join('export', f'{basename}_latent.npz')
    np.savez_compressed(latent_path, latent=np.array(latent_list))
    

    arr = np.zeros((len(x_list), 3))
    arr[:, 0], arr[:, 1], arr[:, 2]= x_list, y_list, area_list
    df = pd.DataFrame(arr, columns = ['x', 'y', 'area'])
    kin_path = os.path.join('export', f'{basename}_kinematic.csv')
    df.to_csv(kin_path, index=True)



    return Plotter.plot_coordinates(x_list, y_list), Plotter.plot_area(area_list), kin_path, latent_path

def select_roi_fn(frame, evt:gr.SelectData):
    point = (evt.index[0], evt.index[1])
    return frame[point[1], point[0]], "click: " + str(point) + ", color: " + str(frame[point[1], point[0]])




def create_ui(OS_SYS):
    with gr.Blocks() as app:
        selected_obj = gr.State(None)
        gr.Markdown(
        '''
        <div style="text-align:center;">
            <span style="font-size:3em; font-weight:bold;">Extract Specific ROI Latent Tool</span>
        </div>
        '''
        )
        with gr.Row():
            with gr.Column(scale=5):
                mask_video = gr.File(label='Mask video', height=100)
                raw_video = gr.File(label='Raw video', height=100)
                first_frame = gr.Image(label='Select ROI', height=550, interactive=True)
                batch_size = gr.Textbox(label='Batch Size', value='16', interactive=True)
                log = gr.Textbox()
                start_btn = gr.Button(value="Generate Latent", interactive=True,)


            with gr.Column(scale=5):
                trace_plot = gr.Plot(label='Trace result')
                area_plot = gr.Plot(label='Area result')
                kin_file = gr.File(label='Kinematic file')
                npy_file = gr.File(label='Latent')
            pass

        first_frame.select(
            fn=select_roi_fn,
            inputs=[
                first_frame,
            ],
            outputs=[
                selected_obj, log,
            ]
        )

        start_btn.click(
            fn=start_fn,
            inputs=[
                mask_video, raw_video, selected_obj, batch_size
            ],
            outputs=[
                trace_plot, area_plot, kin_file, npy_file
            ]
        )
        mask_video.change(
            fn=get_meta_from_video,
            inputs=[
                mask_video,
            ],
            outputs=[
                first_frame
            ]
        )
    return app



# def get_representation(vin, results, progress):
#     cap = cv2.VideoCapture(vin.name)
#     n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     latents = []
    
#     for i in progress.tqdm(range(n)):
#         ok, image = cap.read()
#         if not ok:
#             print("[E] Video Reading Error")
#             break
#         for it in results:
#             if i < len(results[it].labels):
#                 mask = np.array(results[it].labels[i], dtype=float)
#             else:
#                 ok = False
#             break
#         if not ok:
#             print("[E] mask Reading Error")
#             break
#         tmask = img2mask(mask)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         tensor = img2tensor(image)
#         latent = latentGenerator.single_run(tensor)
#         latent = latent.reshape((37, 37, latent_dim))
#         small_mask = np.zeros((37, 37))
#         small_mask = tmask[0].reshape(resolution//14, 14, resolution//14, 14).sum(axis=(1, 3))
#         sum_mask = small_mask.sum()
#         result = small_mask[:, :, np.newaxis] * latent
#         latent_mask_ave = result.sum(axis=0).sum(axis=0) / sum_mask
#         latents.append(latent_mask_ave)

#         # if i == 5:
#         #     break

#     cap.release()
#     latents = np.array(latents)
#     return latents

    


# app = gr.Blocks()
# with app:
#     origin_frist_frame = gr.State(None)
#     objects = gr.State([])
#     selected_obj = gr.State(None)
#     gr.Markdown(
#         '''
#         <div style="text-align:center;">
#             <span style="font-size:3em; font-weight:bold;">Mask Video Analysis Tool</span>
#         </div>
#         '''
#     )
#     with gr.Row():
#         with gr.Column(scale=0.5):

#             input_video = gr.File(label='Mask video', height=100)
#             raw_video = gr.File(label='Raw video', height=100)
#             result_frame = gr.Image(
#                 label='Crop result of first frame', height=550, interactive=True)

#             with gr.Row():

#                 new_object_button = gr.Button(
#                     value="Select clicked ROI",
#                     interactive=True
#                 )
#                 plot_trace = gr.Button(
#                     value="Start",
#                     interactive=True,
#                 )
#             pass

#         with gr.Column(scale=0.5):
#             trace_plot = gr.Plot(label='Trace result')
#             area_plot = gr.Plot(label='Area result')
#             output_file = gr.File(label="Download kinematic file")
#             npy_file = gr.File(label="Download latent file")
#             pass

#         # Back-end
#         input_video.change(
#             fn=get_meta_from_video,
#             inputs=[
#                 input_video,
#             ],
#             outputs=[
#                 origin_frist_frame, result_frame
#             ]
#         )
#         result_frame.select(
#             fn=object_selector,
#             inputs=[
#                 origin_frist_frame,
#             ],
#             outputs=[
#                 selected_obj, log,
#             ]
#         )
#         new_object_button.click(
#             fn=add_obj,
#             inputs=[
#                 selected_obj, objects
#             ],
#             outputs=[
#                 objects, log
#             ]
#         )
     
#         plot_trace.click(
#             fn=plot_traces,
#             inputs=[
#                 input_video, raw_video, objects
#             ],
#             outputs=[
#                 trace_plot, area_plot, output_file, npy_file, log
#             ]
#         )


# if __name__ == "__main__":
#     app.queue(concurrency_count=5)
#     app.launch(debug=True, share=False,
#                server_name="0.0.0.0", server_port=10002).queue()
