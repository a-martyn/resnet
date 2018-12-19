# Deep Residual Learning for Image Recognition: CIFAR-10 

This repository provides an implementation of the paper *Deep Residual Learning for Image Recogniton* by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Han Sun [1] which won the first place on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation. The experiment from section 4.2 of the paper, based on the CIFAR-10 dataset, is reproduced here using the PyTorch framework.

The key insight provided by the paper is that "shortcuts" between network layers allow layers to model the residual function and improve the performance of deeper networks with more than 20 layers. This result is reproduced here. 

This implementation produces a slightly higher test error than the paper with rougly a 1% increase for the plain network baselines and a 2% increase for the residual nets.


## Training results from the original paper

![Figure 6.](./assets/fig6.png)

Figure 6. (from original paper) Training on CIFAR-10. Dashed lines denote training error, and bold lines denote testing error. **Left**: plain networks. The error of plain-110 is higher than 60% and not displayed. **Right**: ResNets.
# Deep Residual Learning for Image Recognition: CIFAR-10 

This repository provides an implementation of the paper *Deep Residual Learning for Image Recogniton* by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Han Sun [1] which won the first place on the tasks of ImageNet detection, ImageNet localisation, COCO detection, and COCO segmentation. The experiment from section 4.2 of the paper, based on the CIFAR-10 dataset, is reproduced here using the PyTorch framework.

The key insight provided by the paper is that "shortcuts" between network layers allow layers to model the residual function and improve the performance of deeper networks with more than 20 layers. This result is reproduced here. 

This implementation produces a slightly higher test error than the paper with roughly a 1% increase for the plain network baselines and a 2% increase for the residual nets.


## Training results from the original paper

![Figure 6.](./assets/fig6.png)

Figure 6. (from original paper) Training on CIFAR-10. Dashed lines denote training error, and bold lines denote testing error. **Left**: plain networks. The error of plain-110 is higher than 60% and not displayed. **Right**: ResNets.

## Training results from this implementation

![Figure 6. Recreation](./assets/fig6_recreation.png)

A recreation of Figure 6. showing the results from this implementation for comparison. All annotations are matched. Epochs are used for x-axis where 1 epoch is equivalent to 391 iterations in Figure 6. 110 layer networks are not tested.

## Best test error

| Architecture | #layers | % Test Error (original paper) | % Test Error (this implementation)  |
| --- | --- | --- | --- |
| Plain Net | 20 | 9.5\* | 9.5 |
| Plain Net | 32 | 10\* | 9.92 |
| Plain Net | 44 | 12\* | 11.35 |
| Plain Net | 56 | 13.5\* | 12.76 |
| ResNet | 20 | 8.75 | 8.0 |
| ResNet | 32 | 7.51 | 7.51 |
| ResNet | 44 | 7.17 | 7.38 |
| ResNet | 56 | 6.97 | 7.33 |

\* These figures are approximate readings from Figure 6. as they aren't provided by the original paper.

Classification error on the CIFAR-10 test set. All methods are with data augmentation. The lowest test error achieved across all training epochs is reported.  

The best test error reproduced for plain networks is approximately equivalent to the original paper. The best test error reproduced for residual networks is 0.36% higher than cited in the original paper.

For the 20 layer residual network in this implementation we observe a test error 0.75% below that reported in the original paper.

This implementations seems to yield notably higher variance in test error prior to epoch 81 when the learning rate is 0.1. After this point the learning rate is reduced to 0.01 and the variance decreases. 

## Analysis

The key observation of the original paper is that residual layers enable deep networks, with more than 20 layers, to outperform shallower networks. That observation is reproduced here with the deepest 56 layer residual network outperforming all other networks tested, whilst the equivalent 56 layer plain network performed the worst.

The original paper also reported that residual layers improved the performance of smaller networks, for example in Figure 6. the 20 layer ResNet outperforms its 'plain' counterpart. That result is also reproduced here with the residual 20 layer network outperforming the plain network by 1.5%.

## Implementation notes and uncertainties

Here I note some ambiguities found in the original paper that for which I made assumptions, as well as some interesting observations that I stumbled across during implementation:

1. **Data Augmentation**: I don't think it is explicitly clear what cropping algorithm the authors employed. The authors describe a 'random crop' whilst referencing a paper [2] that desribes 'corner cropping'. It seems this could be interpreted as either of the following, of which I chose RandomCrop:
  - `torchvision.transforms.RandomCrop`: Crop the given PIL Image at a random location.
  - `torchvision.transforms.FiveCrop`: Crop the given PIL Image into four corners and the central crop

1. **Zero Padding**: The paper specifies kernel size of 3x3 with stride of 1 for most convolutions in the architectures. Where this occurs it is stated that the feature map size is maintained, yet this would not be possible unless a zero-padding of 1 pixel is accounted for during convolution. This implementation therefore assumes a zero padding of 1.

1. **Subsampling**: The architecture described includes subsampling to decrease the feature map size from 32x32 to 16x16 to 8x8, with filter counts increasing 16 to 32 to 64 respectively. This is achieved by using a kernel size of 3x3 pixels with a stride of 2. It can be shown that this convolution does not exactly fit into any even sized feature map. The leading edge of the final convolution in each row will have no input. It is assumed that this discrepancy is of no effect to the performance of the network and so can be ignored. 

1. **Downsampling Shortcuts**: When a residual block downsamples its input the feature map size is halved in both dimensions whilst the number of filters is doubled. This means that the residual shortcut cannot simply add the block's input matrix to the its output because the dimensions do not match. Some form of 'downsampling' is required. For the CIFAR-10 experiment the authors describe linear mapping to perform this downsampling (see 'option A' section 3.3). The important characteristic here is that the downsampling procedure should add no additional learnable parameters. This implementation achieves this with a 2d average pooling of kernel size 1 and stride 2, effectively dropping every other pixel, which results in feature maps halved in each dimension. We then need to double the number of filters which is achieved crudely by concatenating a duplicate of the downsampled feature maps multiplied by zero. This approach seems crude because half of the input information is simply ignored by the residual shortcut. It seems that this approach might have a desirable a regularisation effect similar to that achieved by dropout [3]. Or would performance be improved by reshaping the input to match the output dimensions? Some relevant investigation is provided here [4].


## References

- [1] K. He, X. Zhang, S. Ren, and J. Sun.  Deep residual learning for image recognition. In CVPR, 2016.
- [2] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply-supervised nets. arXiv:1409.5185, 2014.
- [3] Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 2014
- [4] K. He, X. Zhang, S. Ren, and J. Sun.  Identity mappings in deep residual networks. In ECCV, 2016

## Training results from this implementation

![Figure 6. Recreation](./assets/fig6_recreation.png)

A recreation of Figure 6. showing the results from this implementation for comparison. All annotations are matched. Epochs are used for x-axis where 1 epoch is equivalent to 391 iterations in Figure 6. 110 layer networks are not tested.

## Best test error

| Architecture | #layers | % Test Error (original paper) | % Test Error (this implementation)  |
| --- | --- | --- | --- |
| Plain Net | 20 | 9.5\* | 9.5 |
| Plain Net | 32 | 10\* | 9.92 |
| Plain Net | 44 | 12\* | 11.35 |
| Plain Net | 56 | 13.5\* | 12.76 |
| ResNet | 20 | 8.75 | 8.0 |
| ResNet | 32 | 7.51 | 7.51 |
| ResNet | 44 | 7.17 | 7.38 |
| ResNet | 56 | 6.97 | 7.33 |

\* These figures are approximate readings from Figure 6. as they aren't provided by the original paper.

Classification error on the CIFAR-10 test set. All methods are with data augmentation. The lowest test error achieved across all training epochs is reported.  

The best test error reproduced for plain networks is approximately equivalent to the original paper. The best test error reproduced for residual networks is 0.36% higher than cited in the original paper.

For the 20 layer residual network in this implementation we observe a test error 0.75% below that reported in the original paper.

This implementations seems to yield notably higher variance in test error prior to epoch 81 when the learning rate is 0.1. After this point the learning rate is reduced to 0.01 and the variance decreases. 

## Analysis

The key observation of the original paper is that residual layers enable deep networks, with more than 20 layers, to outperform shallower networks. That observation is reproduced here with the deepest 56 layer residual network outperforming all other networks tested, whilst the equivalent 56 layer plain network performed the worst.

The original paper also reported that residual layers improved the performance of smaller networks, for example in Figure 6. the 20 layer ResNet outperforms its 'plain' counterpart. That result is also reproduced here with the residual 20 layer network outperforming the plain network by 1.5%.

## Implementation notes and uncertainties

Here I note some ambiguities found in the original paper that for which I made assumptions, as well as some interesting observations that I stumbled across during implementation:

1. **Data Augmentation**: I don't think it is explicitly clear what cropping algorithm the authors employed. The authors describe a 'random crop' whilst referencing a paper [2] that desribes 'corner cropping'. It seems this could be interpreted as either of the following, of which I chose RandomCrop:
    - `torchvision.transforms.RandomCrop`: Crop the given PIL Image at a random location.
    - `torchvision.transforms.FiveCrop`: Crop the given PIL Image into four corners and the central crop

2. **Zero Padding**: The paper specifies kernel size of 3x3 with stride of 1 for most convolutions in the architectures. Where this occurs it is stated that the feature map size is maintained, yet this would not be possible unless a zero-padding of 1 pixel is accounted for during convolution. This implementation therefore assumes a zero padding of 1.

3. **Subsampling**: The architecture described includes subsampling to decrease the feature map size from 32x32 to 16x16 to 8x8, with filter counts increasing 16 to 32 to 64 respectively. This is achieved by using a kernel size of 3x3 pixels with a stride of 2. It can be shown that this convolution does not exactly fit into any even sized feature map. The leading edge of the final convolution in each row will have no input. It is assumed that this discrepency is of no effect to the performance of the network and so can be ignored. 

4. **Downsampling Shortcuts**: When a residual block downsamples its input the feature map size is halved in both dimensions whilst the number of filters is doubled. This means that the residual shortcut cannot simply add the block's input matrix to the its output because the dimensions do not match. Some form of 'downsampling' is required. For the CIFAR-10 experiment the authors describe linear mapping to perform this downsampling (see 'option A' section 3.3). The important characteristic here is that the downsampling procedure should add no additional learnable parmaeters. This implementation achieves this with a 2d average pooling of kernel size 1 and stride 2, effectively dropping every other pixel, which results in feature maps halved in each dimension. We then need to double the number of filters which is achieved crudely by concatenating a duplicate of the downsampled feature maps multiplied by zero. This approach seems crude because half of the input information is simply ignored by the residual shortcut. It seems that this approach might have a desirable a regularisation effect similar to that achieved by dropout [3]. Or would performance be improved by reshaping the input to match the output dimensions? Some relevant investigation is provided here [4].

5. **Batch Normalisation**: During implementation I observed that batch normalisation is critical to reproducing the original results. Specifically it is critical that the batch normalisation implementation uses running estimates of mean and standard deviation across batches (`track_running_stats=True` in PyTorch). This yields a 2-3% reduction in test error, and interestingly without this feature residual shortcuts provide no improvement for the smallest 20-layer network. In addition I discovered a [comment](https://github.com/KaimingHe/deep-residual-networks/issues/10) from the authors clarifying  that they chose not to implement bias in the convolutional layers, instead adding learnable bias in the batch normalisation layers. I implemented this with the `affine=True` parameter of the PyTorch function `torch.nn.BatchNorm2d`.

## References

- [1] K. He, X. Zhang, S. Ren, and J. Sun.  Deep residual learning for image recognition. In CVPR, 2016.
- [2] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply-supervised nets. arXiv:1409.5185, 2014.
- [3] Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 2014
- [4] K. He, X. Zhang, S. Ren, and J. Sun.  Identity mappings in deep residual networks. In ECCV, 2016
