import cv2


def get_meta_from_video(input_video):
    if input_video is None:
        return None, None, None

    cap = cv2.VideoCapture(input_video.name)

    _, first_frame = cap.read()
    cap.release()

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    return first_frame