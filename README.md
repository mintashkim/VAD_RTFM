# RTFM

Video Anomaly Detection의 한 종류인 Weakly Supervised learning 기법을 이용한 RTFM 재현 및 연구

[original repository](https://github.com/tianyu0207/RTFM)

## RTFM 재현
- 원본 저자가 제공한 ShanghaiTech i3d features를 이용해서 학습해봄
- 몇몇 불편한 코드(데이터 불러오는 등)를 수정
- i3d features를 어떻게 추출했는지에 대한 정보가 없어 삽질을 좀 했음
    - 16frame마다 추출했다고 함
    - 10 crops augmentation을 사용했다는데 찾아보니 torchvision의 TenCrop을 말하는 듯 함

## RTFM 개선
- 이왕 3d features를 추출하는 김에 3D conv를 이용한 i3d, c3d보다 개선된 최신모델인 x3d을 사용하기로 함
    - torchvideo를 활용. 자세한 내용은 [pytorchvideo_x3d](http://192.168.1.37/ais/pytorchvideo_x3d)를 참고
    - UCF_Crime 데이터에 대해 x3d feature 추출 (24시간 이상소요, 진행중)
    - UCF + x3d 로 RTFM 학습
        - UCF 데이터에 대해선 좋은 성능을 보임. 따라서 모델자체의 성능은 나쁘지 않음.
        - 완전히 학습되지 않은 7가지 act를 녹화해서 돌려봄. 정답 :white_check_mark:, 오답 :x:
            - normal:{drinking :white_check_mark:, googling :x:, normal :white_check_mark:, toilet :white_check_mark:}
            - abnormal:{capture :x:, drawing :white_check_mark:, writing :white_check_mark:}
            - 여기서 확인할 것은 1. 차이가 크지 않은 영상임에도 결과값이 어느정도 변한다는 것. 2. webcam domain은 전혀 학습되지 않았음에도 우연인지 결과가 꽤 좋다는 것
            - 데이터를 확보해서 webcam domain을 weakly supervised learning하면 가능성이 있다고 생각함.
    
## TODO
- 속임수 얼굴 데이터셋 확보 및 학습
- 아도스 데이터 확보 및 학습

## 관련 오류
- RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
    - cuda이슈해결.txt 참고
- visdom이 없다 관련 오류
    - visdom은 웹상의 visualization 툴임
    - pip install visdom
    - (새 터미널에서) python -m visdom.server
    - 웹으로 접속하여 실험 내용을 볼 수 있음