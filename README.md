# RTFM

A famous weakly supervised learning model for video anomaly detection (VAD). 

[original repository](https://github.com/tianyu0207/RTFM)

## Implementation
- ShanghaiTech i3d features
- Revised few inconvenient codes
- i3d features
    - Extracted for every 16 frames
    - 10 crops augmentation means torchvision TenCrop

## Re-Tooling
- 3d features extraction: Use x3d rather than i3d, c3d which are 3D conv models
    - Use torchvideo: [pytorchvideo_x3d](http://192.168.1.37/ais/pytorchvideo_x3d)
    - UCF_Crime x3d feature extraction (requires more than 24 hrs)
    - Train RTFM with UCF_Crime + x3d dataset
        - Good performance!
        - Tested with new motion data (never seen):  Got correct: white_check_mark:, Got correct: x:
            - normal:{drinking :white_check_mark:, googling :x:, normal :white_check_mark:, toilet :white_check_mark:}
            - abnormal:{capture :x:, drawing :white_check_mark:, writing :white_check_mark:}
            - Things to notice: 1. Result changes even the data does not have big difference 2. Model works fairly good for webcam domain data even it is not trained with
            - If webcam domain data is given and weakly supervised, there is hope :)
    
## TODO
- Webcam abnormal face data
- Webcam abnormal motion data

## 관련 오류
- RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
    - See cuda_troubleshooting.txt
- No visdom error
    - visdom is a web-based visualization tool
    - pip install visdom
    - (in a new terminal) python -m visdom.server
    - can see the result by accessing via web
