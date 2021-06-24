# Petals to the Metal - Flower Classification
**Kaggle competition link** : https://www.kaggle.com/c/tpu-getting-started/


**ResNet paper** : [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
## Survey
- [Link](https://docs.google.com/spreadsheets/d/12Im83uIkQc6xnxnphKwH8cOnKqIcwoUeNexsDw1awY0/edit?usp=sharing) for google sheet
	- [x] AlexNet
	- [x] VGGNet
	- [x] ResNet
## Implementation
 - **Dataset preparation**
	 - [X] TFRecord converter
	 - [X] PyTorch Dataset
	 - [X] PyTorch Transforms(Image augmentation)
 - **Plain nets vs residual nets (34 layers)**
 	 - [X] PlainBlock for 34-layer plain nets
 - **BasicBlock vs BottleNeck**
 - **Transfer learning from ImageNet pre-trained weight**
## Usage
- **Tested Environment**
    - Linux Ubuntu 18.04
    - Titan RTX GPU
    - Nvidia driver version 418.67
- **Build docker container**
    - (host) $ sh build_docker_img.sh
    - (host) $ sh build_docker_container.sh
- **Setup package**
    - (container) $ python setup.py develop
- **Train**
    - (container) $ sh run.sh
## TODO

1. **Hyperparameter tuning**
	- 처음에는 train epochs 를 50으로 설정했지만, 실제 ResNet paper 에서는 600000 iters 까지 학습을 진행함
	- We use SGD with a mini-batch size of 256. The learning rate starts from 0.1 and is divided by 10 when the error plateaus, and the models are trained for up to 60 × 10^4 iterations
	- 600,000 batch * 256 imgs/batch * (1/128000) epoch/imgs = 120 epoch
	- Train epochs 를 50에서 원래 논문처럼 120 Epochs 까지 학습하면 더 뚜렷한 convergence rate 를 확인할 수 있을 것으로 예상됨 -> ImageNet보다 데이터셋 사이즈가 작아서 실제로는 70Epoch 의 학습도 충분했음
	- LR scheduler 의 patience 값 감소(validation loss 가 진전이 없는 상태에서 더 빠른 convergence를 유도)
	- 실제로 LR scheduler의 patience 값을 10에서 7로 바꾸고, 학습을 70 Epoch 까지 해서 모델 수렴 결과를 볼 수 있었음

2. **Preprocess**
	-  color transform 사용 여부에 따른 ablation study
	- preprocess tensor normalization 에서 train, validation dataset 의 pixel mean, std를 한번에 구하고, cross validation 적용

3. **Test**
	-  Test dataset 에 대한 testing process 구현
	-  10-crop testing (horizontal flips x FiveCrop) 적용
	- Fully-convolutional form 을 통한 multiple scales {224, 256, 384, 480, 640} size 에서 테스트

4. **Ensemble models**
	-  plain34, resnet34, resnet50, pretrained-resnet50 각 모델의 결과의 vote 기반 ensemble 을 통한 성능 향상
5. **Code Refactoring**
	-  Tensorboard에서 epoch에 따른 loss를 바로 확인할 수 있도록 SummaryWriter hook에 추가
	-  Validation loss 가 min 값일 때만 Checkpoint 저장하도록 변경
	-  Hyper parameter, Image preprocessing pipeline 에 대한 configuration file 작성 및 연동
	-  Kaggle submission format에 따른 test result csv builder script 작성
