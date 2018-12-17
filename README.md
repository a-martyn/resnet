# Deep Residual Learning for Image Recognition: CIFAR-10 

This repository provides a PyTorch implementation of the paper *Deep Residual Learning for Image Recogniton* by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Han Sun which achived state of the art in 2015 by winning the ILSVRC and COCO challenges. The experiment from section 4.2 of the paper, based on the CIFAR10 dataset, is reproduced here.

The key insight provided by the paper is that "shortcuts" between network layers allow layers to model the residual function and improve the performance of deeper networks with more than 20 layers. This result is reproduced here, with 2% higher test error for all results.


## Training results from the original paper

![Figure 6.](./assets/fig6.png)

Figure 6. (from original paper) Training on CIFAR-10. Dashed lines denote training error, and bold lines denote testing error. **Left**: plain networks. The error of plain-110 is higher than 60% and not displayed. **Right**: ResNets.

## Training results from this implementation

![Figure 6. Recreation](./assets/fig6_recreation.png)

A recreation of Figure 6. showing the results from this implementation for comparison. All annotations are matched. Epochs are used for x-axis where 1 epoch is equivalent to 391 iterations in Figure 6.

## Best test error

| Architecture | #layers | % Test Error (original paper) | % Test Error (this implementation)  |
| --- | --- | --- | --- |
| Plain Net | 20 | | 10.62 |
| Plain Net | 32 | | 10.85 |
| Plain Net | 44 | | 12.42 |
| Plain Net | 56 | | 14.22 |
| ResNet | 20 | 8.75 | 10.98 |
| ResNet | 32 | 7.51 | 10.73 |
| ResNet | 44 | 7.17 | 11.10 |
| ResNet | 56 | 6.97 | 8.96 |

Classification error on the CIFAR-10 test set. All methods are with data augmentation. The lowest test error achieved across all training epochs is reported for this implementation.

## Conclusion

The key observation of the original paper is that residual layers enable deep networks to outperform shallower networks. That observation is reproduced here with the deepest 56 layer residual network outperforming all other networks tested, whilst the equivalent 56 layer plain network performed the worst.

The original paper also reported that residual layers improved the performance of smaller networks too, for example in Figure 6. the 20 layer ResNet outperforms its 'plain' counterpart. That result is not reproduced here. Instead the plain 20 layer network slightly outperforms the residual equivalent with test errors of 10.6% and 11% respectively.

Finally I observed higher test errors for all experiments with a discrepency of around 2%. I ran each experiment once and so the variance between experiments is unknown, however the paper's authors report a standard deviation of 0.16% for the 110 layer experiment suggesting a 2% discrepency is likely significant. This discrepency could be attributed to the cropping algorithm chosen in data augmentation, or perhaps a difference in the implementation of batch normalisation. Another possibility is that the authors might have implemented pooling following the first layer, which is ambiguous in the paper and not implemented here.
