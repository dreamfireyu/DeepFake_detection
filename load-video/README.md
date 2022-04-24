# Operator: video-decoder

*Author: Zhuoran Yu*

## Description

A video preprocess operator implementation with OpenCV

## Code Example

 
## Interface

```python
__init__(self)
```

```python
__call__(self,
        file_name: str,
        every_n_frames: int,
        specific_frames: int,
        to_rgb: bool,
        rescale: int,
        max_frames: int)
```
Args:
- file_name: video path
- every_n_frames: select 1 frame every n frames
- specific_frames: select specific frame from video
- to_rgb: whether to convert color to RGB
- rescale: rescale size
- max_frame: maximum frames to extract

Returns:

- Video,pil_frames,rescale

## Requirements
torchvision

pillow

opencv-python

pathlib

## How it works



## Reference
