# Deep Residual Learning for Image Recognition: CIFAR-10 

This repository provides a PyTorch implementation of the paper *Deep Residual Learning for Image Recogniton* by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Han Sun which achived state of the art in 2015 by winning the ILSVRC and COCO challenges. The experiment from section 4.2 of the paper, based on the CIFAR10 dataset, is reproduced here.

The key insight provided by the paper is that "shortcuts" between network layers allow layers to model the residual function and improve the performance of deeper networks with more than 20 layers. This result is reproduced here, with 2% higher test error for all results.


## Results from the original paper

![Figure 6.](./assets/fig6.png)

Figure 6. (from original paper) Training on CIFAR-10. Dashed lines denote training error, and bold lines denote testing error. **Left**: plain networks. The error of plain-110 is higher than 60% and not displayed. **Right**: ResNets.

## Results from this implementation

![Figure 6. Recreation](./assets/fig6_recreation.png)

A recreation of Figure 6. showing the results from this implementation for comparison. All annotations are matched.

