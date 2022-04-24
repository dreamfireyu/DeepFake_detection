import cv2
import numpy as np
from PIL import Image

from towhee import register
from towhee.operator.base import Operator,OperatorFlag
import logging

log = logging.getLogger()

@register(output_schema=["video","framelist","rescale"],
          flag=OperatorFlag.STATELESS | OperatorFlag.REUSEABLE)
class LDvideo(Operator):
    """
    Loads a video.
    Called by:
    1) The finding faces algorithm where it pulls a frame every FACE_FRAMES frames
    up to MAX_FRAMES_TO_LOAD at a scale of FACEDETECTION_DOWNSAMPLE, and then half that if there's a CUDA memory error.
    2) The inference loop where it pulls EVERY frame up to a certain amount which it the last needed frame for each face for that video
    """
    def __init__(self):
        pass

    def __call__(self,
        filename: str,
        every_n_frames: int=None,
        specific_frames: int=None,
        to_rgb: bool=True,
        rescale: int=None,
        max_frames: int=None):
        assert every_n_frames or specific_frames
        assert bool(every_n_frames) != bool(specific_frames)
        cap = cv2.VideoCapture(filename)
        n_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if rescale:
            rescale = rescale * 1920./np.max((width_in, height_in))
        width_out = int(width_in*rescale) if rescale else width_in
        height_out = int(height_in*rescale) if rescale else height_in
        if max_frames:
            n_frames_in = min(n_frames_in, max_frames)
        if every_n_frames:
            specific_frames = list(range(0,n_frames_in,every_n_frames))
        n_frames_out = len(specific_frames)
        out_pil = []
        out_video = np.empty((n_frames_out, height_out, width_out, 3), np.dtype("uint8"))
        i_frame_in = 0
        i_frame_out = 0
        ret = True
        while (i_frame_in < n_frames_in and ret):
            try:
                try:
                    if every_n_frames == 1:
                        ret, frame_in = cap.read()  # Faster if reading all frames
                    else:
                        ret = cap.grab()
                        if i_frame_in not in specific_frames:
                            i_frame_in += 1
                            continue
                        ret, frame_in = cap.retrieve()
                    if rescale:
                        frame_in = cv2.resize(frame_in, (width_out, height_out))
                    if to_rgb:
                        frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
                except Exception as e: # pylint: disable=broad-except
                    log.error("Error for frame %d for video %s: %s; using 0s",i_frame_in,filename,e)
                    frame_in = np.zeros((height_out, width_out, 3))
                out_video[i_frame_out] = frame_in
                i_frame_out += 1
                try:
                    pil_img = Image.fromarray(frame_in)
                except Exception as e: # pylint: disable=broad-except
                    log.error("Using a blank frame for video %s frame %d as error %s",filename,i_frame_in,e)
                    pil_img = Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))  # Use a blank frame
                out_pil.append(pil_img)
                i_frame_in += 1
            except Exception as e: # pylint: disable=broad-except
                log.error("Error for file %s: %s",filename,e)
        cap.release()
        return out_video, out_pil, rescale
if __name__ == "__main__":
    file_name = "/home/xuyu/Deepfake/deepfake_detec/aagfhgtpmv.mp4"
    op = LDvideo()
    video,pil_frames,res = op(filename=file_name,every_n_frames=10,to_rgb=True,rescale=0.25,max_frames=100)
    print(video,pil_frames,res)
