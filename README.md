# ResNet38 on CIFAR10

## Implementation

Implement Model A in paper [Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/abs/1611.10080).
Use max pooling to perform downsampling. Downsample 3 times. No dilated convolutions

Model A structure: Input(32x32) -> B0 -> max-pool -> B2(x3) -> max-pool -> B3(x3) -> max-pool -> B4(x6) -> B5(x3) -> B6(x1) -> B7(x1) ->
Global-avg-pool -> Fully connected -> Softmax

## Results

- Under development
- Best accuracy so far: ???
- Data set: 50000 training image. 10000 test images(not used for training).
- Data augmentation: Per image standardization. Per image pad to 36x36, then randomly crop to 32x32. Randomly shuffle all images per epoch. Flip all images per epoch.
- Training: Train ??? epochs. Batch 128. Adam optimizer(with default hyperparameters). L2 weight decay.
- Device: GTX TITAN (Pascal) 12GB

## Acknowledge

Thanks for the GPU provided by [Computer Vision and Pattern Recongnition Group at Technical University Munich](https://vision.in.tum.de/)
