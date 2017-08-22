# ResNet38 on CIFAR10

## Implementation

Implement Model A in paper [Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/abs/1611.10080).
Use convolution with stride=2 as downsampling operation instead of max pooling. Only downsample 3 times, then use dilated convolutions.
Use dropout on wide layers 2048 and 4096, however this is not specified in original paper.

Model A structure: Input(32x32) -> B0 -> B2(x3, downsample once) -> B3 (x3, downsampled once) -> B4(x6, downsampled once) -> B5(x3, dilated) -> B6(x1, dilated) -> B7(x1, dilated) -> 
Global-avg-pool -> Fully connected -> Softmax

## Results

- Best accuracy so far: 89.38%
- Data set: 50000 training image. 10000 test images(not used for training).
- Data augmentation: Per image standardization. Per image pad to 36x36, then randomly crop to 32x32. Randomly shuffle all images per epoch. Flip all images per epoch.
- Training: Train 150 epochs. Batch 128. Adam optimizer(with default hyperparameters). L2 weight decay 0.0002
- Device: GTX TITAN (Pascal) 12GB

## Acknowledge

Thanks for the GPU provided by [Computer Vision and Pattern Recongnition Group at Technical University Munich](https://vision.in.tum.de/)

## Update

 - Data augmentation: Randomly flip per image during training instead of flip all images per epoch.
 - Accuracy: ???
