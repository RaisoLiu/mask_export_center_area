import av
import os
import numpy as np

class SourceVideo:
    def __init__(self, video_path):
        self.video_name = os.path.basename(video_path)
        self.video_path = video_path
        self.container = av.open(video_path)
        self.video_stream = self.container.streams.video[0]
        self.fps = self.video_stream.average_rate
        self.pts2index = self.video_stream.time_base * self.video_stream.average_rate
        self.total_frames = self.count_frames()
        
        self.index = 0
    
    def count_frames(self):
        container = av.open(self.video_path)
        video_stream = container.streams.video[0]
        total_frame_meta = video_stream.frames
        timestamp = np.max((0, (total_frame_meta-300))) / self.pts2index
        
        count = 0
        container.seek(int(timestamp), stream=video_stream, backward=True)
        for frame in container.decode(video_stream):
            index = int(frame.pts * self.pts2index)
            count = index + 1

        container.close()
        print('count', count)
        return count

    
    def read_by_index(self, frame_index):
        assert frame_index >= 0 and frame_index < self.total_frames
        
        if frame_index == self.index + 1:
            return next(self.container.decode(self.video_stream))
        

        timestamp = frame_index / self.pts2index
        self.container.seek(int(timestamp), stream=self.video_stream, backward=True)
        for frame in self.container.decode(self.video_stream):
            index = int(frame.pts * self.pts2index)
            if index == frame_index:
                self.index = frame_index
                break
        return frame

    def __del__(self):
        self.container.close()

