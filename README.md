# DenseNetFCN3D-pytorch
PyTorch implementation of Full Convolutional DenseNet for 3D image segmentation

Based on:
The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
https://arxiv.org/abs/1611.09326

The default values for DenseNetFCN are as per instructions for DenseNet103

## Fully Convolutional DenseNet Architecture

All images and caption are taken directly from the paper cited above.

![image](https://user-images.githubusercontent.com/56860757/220510062-aee040f7-1efa-4195-b9e3-2c5c2e0abdaf.png)

Figure 1. Diagram of our architecture for semantic segmentation.
Our architecture is built from dense blocks. The diagram is composed of a downsampling path with 2 Transitions Down (TD) and
an upsampling path with 2 Transitions Up (TU). A circle represents concatenation and arrows represent connectivity patterns in
the network. Gray horizontal arrows represent skip connections,
the feature maps from the downsampling path are concatenated
with the corresponding feature maps in the upsampling path. Note
that the connectivity pattern in the upsampling and the downsampling paths are different. In the downsampling path, the input to
a dense block is concatenated with its output, leading to a linear
growth of the number of feature maps, whereas in the upsampling
path, it is not.

### DenseBlock

![image](https://user-images.githubusercontent.com/56860757/220510248-c9ae8512-3200-46d2-9a47-718f37563edb.png)

Figure 2. Diagram of a dense block of 4 layers. A first layer is applied to the input to create k feature maps, which are concatenated
to the input. A second layer is then applied to create another k
features maps, which are again concatenated to the previous feature maps. The operation is repeated 4 times. The output of the
block is the concatenation of the outputs of the 4 layers, and thus
contains 4 ∗ k feature maps

### Details of components

![image](https://user-images.githubusercontent.com/56860757/220510522-4083c187-1440-4b77-b7f0-c3f06a61fbce.png)

Table 1. Building blocks of fully convolutional DenseNets. From left to right: layer used in the model, Transition Down (TD) and Transition
Up (TU). See text for details.

## FCN DenseNet103

Below is the table from the paper where the number of features are listed upon exiting each denseblock:

![image](https://user-images.githubusercontent.com/56860757/220510754-3abacc3e-cfd5-4c4b-bd7a-401b36140335.png)

Table 2. Architecture details of FC-DenseNet103 model used in
our experiments. This model is built from 103 convolutional layers. In the Table we use following notations: DB stands for Dense
Block, TD stands for Transition Down, TU stands for Transition
Up, BN stands for Batch Normalization and m corresponds to the
total number of feature maps at the end of a block. c stands for the
number of classes.


I found there to be an error in the m value listed. In my code you will be able to see where m is extracted from. The number of features after existing each down-block, then the bottleneck, and then each of the 5 upblocks are: (48 is the number of features after the initial convoluation layer) 112, 192, 304, 464, 656, 896, 1088, 816, 576, 384, 256.


