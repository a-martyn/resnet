# Deep Residual Learning for Image Recognition: CIFAR-10 

This repository provides an implementation of the paper *Deep Residual Learning for Image Recogniton* by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Han Sun which achived state of the art in 2015 by winning the ILSVRC and COCO challenges. The experiment from section 4.2 of the paper, based on the CIFAR10 dataset, is reproduced here using the PyTorch framework.

The key insight provided by the paper is that "shortcuts" between network layers allow layers to model the residual function and improve the performance of deeper networks with more than 20 layers. This result is reproduced here. 

This implementation produces a slightly higher test error than the paper with rougly a 1% increase for the plain network baselines and a 2% increase for the residual nets.


## Training results from the original paper

![Figure 6.](./assets/fig6.png)

Figure 6. (from original paper) Training on CIFAR-10. Dashed lines denote training error, and bold lines denote testing error. **Left**: plain networks. The error of plain-110 is higher than 60% and not displayed. **Right**: ResNets.

## Training results from this implementation

![Figure 6. Recreation](./assets/fig6_recreation.png)

A recreation of Figure 6. showing the results from this implementation for comparison. All annotations are matched. Epochs are used for x-axis where 1 epoch is equivalent to 391 iterations in Figure 6. 110 layer networks are not tested.

## Best test error

| Architecture | #layers | % Test Error (original paper) | % Test Error (this implementation)  |
| --- | --- | --- | --- |
| Plain Net | 20 | 9.5\* | 10.62 |
| Plain Net | 32 | 10\* | 10.85 |
| Plain Net | 44 | 12\* | 12.42 |
| Plain Net | 56 | 13.5\* | 14.22 |
| ResNet | 20 | 8.75 | 10.98 |
| ResNet | 32 | 7.51 | 10.73 |
| ResNet | 44 | 7.17 | 11.10 |
| ResNet | 56 | 6.97 | 8.96 |

\* These figures are approximate readings from Figure 6. as they aren't provided by the original paper.

Classification error on the CIFAR-10 test set. All methods are with data augmentation. The lowest test error achieved across all training epochs is reported for this implementation.  

The test errors reproduced for the plain network baselines are approximately 1% higher than the original paper. Note that this comparison is inaccurate because exact figures are not provided by the paper, and so we rely on readings from Figure 6, left-hand panel.

## Analysis

The best test error reproduced for residual networks is 1.99% higher than cited in the original paper, with the disparity for 32 and 42 layer networks being higher at around 3.5%.

The key observation of the original paper is that residual layers enable deep networks to outperform shallower networks. That observation is reproduced here with the deepest 56 layer residual network outperforming all other networks tested, whilst the equivalent 56 layer plain network performed the worst.

The original paper also reported that residual layers improved the performance of smaller networks, for example in Figure 6. the 20 layer ResNet outperforms its 'plain' counterpart. That result is not reproduced here. Instead the plain 20 layer network slightly outperforms the residual equivalent with test errors of 10.6% and 11% respectively.

Finally higher test errors are observed for all experiments with a discrepancy of around 1% for plain networks and 2-4% for residual networks. I ran each experiment once and so the variance between experiments is unknown, however the paper's authors report a standard deviation of 0.16% for the 110 layer experiment suggesting a the discrepancies observed here are likely significant. 

This discrepancy could be attributed to the cropping algorithm chosen in data augmentation, or perhaps a difference in the implementation of batch normalisation. Another possibility is that the authors might have implemented pooling following the first layer, which is ambiguous in the paper and not implemented here.

## Implementation uncertainties

