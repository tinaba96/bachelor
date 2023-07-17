# Bachelor
# Quantization of Convolutional Neural Network for Motion Estimation for FPGA Implementation




This is a Pytorch implementation of quantized network for flownet implemented in https://github.com/NVIDIA/flownet2-pytorch.git

The algorithm I used for this quantization is similar to https://github.com/jiecaoyu/XNOR-Net-PyTorch


- - -
## Background
- - -
Recently, the technology of convolutional neural network (CNN) is applied to image recognition. Especially, a network for semantic segmentation, a network for motion estimation and a network for distance estimation are applied to in-vehicle cameras for automatic driving and contribute to driving control. However, these networks need massive calculation and time to process the data. 



- - -
## Objectives
- - -
Since it increases the circuit size of a hardware, I consider the quantization of the weights and activations of CNN for motion estimation, hence that I can reduce the circuit size when I implement the network on FPGA.
The purpose of this research is to quatize the model to reduce the circuit size without losing the accuracy.

- - -
## Approach
- - -
In order to quantize the CNN for motion estimation, I quantized the activations, which are the inputs of the convolutional layer, and weights of the network. 
As a motion estimation CNN model, I target the [FlowNetSimple](https://github.com/NVIDIA/flownet2-pytorch.git) shown below.

![image](https://github.com/tinaba96/master_old/assets/57109730/db7cf0ce-665b-419d-b0fd-7d541b278f41)
![image](https://github.com/tinaba96/Quantization-Flownet2-Pytorch/assets/57109730/f904e965-eefa-48cf-b091-d373ae26d36b)

### Quantization for the activations
When quantizing the activations, which are the outputs of the activation function, in the contracting part, I applied batch normalization and used ReLU as the activation function. That is, with batch normalization, the distribution of activation was made `0` on average and `1` on variance. In addition, the value less than `0` was made `0` by the activation function ReLU. Here, I quantized the activations in the range of `0` to `4` (`4σ`) (`σ` indicates the standard deviation). Thanks to applying the normalization and activation function ReLU before the quantization, the range of the activations to be quantized can be fixed.

 In addition, before applying the quantization of the expanding part, the expanding part of the unquantized network FlowNerSimple was modified to be easy to quantize as follows. In the unquantized network FlowNetSimple, the
 
refinement is performed by using upconvolution and the activation function LeakyReLU. Therefore, we first replace upconvolution with convolution and upsampling using bilinear interpolation. Then we perform batch normalization after convolution to determine the range to be quantized. By modifying in this way, it became possible to quantize the expanding part as same as the contracting part.
In addition, since batch normalization cannot be applied before flow prediction, flow prediction is not performed in each layer of the expanding part in the modified network. Hence, flow prediction is performed only in the output layer. That is, the concatenation in each layer of the expanding part is not the output of layers of the contracting part, the output of the previous layer and the predicted flow, but only the output of layers of the contracting part and the output of the previous layers
 

### Quantization for the weights
In the quantization of weights, since it is known that the average of the weights distribution is `0` and the standard deviation is smaller than `1`, I set the quantization range from `-1` to `1`. As the process flow, I first quantize weights and save unquantized weights. Next, forward and backward calculations are executed. After carrying out forward and backward calculations with quantized weights, the unquantized weights are restored. Finally, the weights are updated using unquantized weights. That is, at the time of learning, inference and error back propagation are performed using the quantized weights, and the weights before quantization are updated using the obtained gradients.



- - -
## Result & Conclusion
- - -



- - -
- - -

# How To

