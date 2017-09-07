# ResNet38 on CIFAR10

## Implementation

Implement original Model A in paper [Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/abs/1611.10080).
Downsampling 3 times with maxpool. No dilated convolution. No dropout.

Model A structure: Input(32x32) -> B0 -> max-pool -> B2(x3) -> max-pool -> B3(x3) -> max-pool -> B4(x6) -> B5(x3) -> B6(x1) -> B7(x1) ->
Global-avg-pool -> Fully connected -> Softmax

## Results

- Under development
- Best accuracy so far: 93.94%
- Data set: 50000 training image. 10000 test images(not used for training).
- Data augmentation: Per image standardization. Per image pad to 36x36, then randomly crop to 32x32. Randomly shuffle all images per epoch. Per image randomly flip during training.
- Training: Train 130 epochs. Batch 128. Momentum optimizer(initial learning rate 0.1(150 epoch) -> 0.01(150 epoch) -> 0.001(170 epoch, todo) ). L2 weight decay 0.0005.
- Device: GTX TITAN (Pascal) 12GB

## Acknowledge

Thanks for the GPU provided by [Computer Vision and Pattern Recongnition Group at Technical University Munich](https://vision.in.tum.de/)

## TODO

- Try reducing number of parameters

## Other tries 1

- use L2 weight decay of 0.0005, train Adam(lr=0.001) 140 epoch -> 85.6%
- Decrease Adam(lr=0.0001) 100 epoch -> 91%

