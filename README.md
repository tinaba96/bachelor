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


- DanQ: Target Hybrid (CNN & RNN) Model for genomics data

![image](https://github.com/tinaba96/master/assets/57109730/cf69250c-abe9-4c8f-b234-f7555692fd1c)


- Mutli-FPGA Implementation using AWS F1 Instance

The overview of the FPGA Implementation

![image](https://github.com/tinaba96/master/assets/57109730/909df276-43a6-47c1-b23b-2854795d3a9c)

By separating the process on multiple FPGAs the way they can process individually, I can obtain faster training time.
I also indicate the result we obtained by implementing the DanQ model on FPGAs. We show three different implementations that are implemented on single FPGA, dual FPGA and 8 FPGAs.

![image](https://github.com/tinaba96/master/assets/57109730/e1f0325d-7e3e-4d4a-9477-5c5fa2e9032b)

I focused on the BiLSTM layer, which consumes about 48% of the whole training time. BiLSTM is time-consuming because there are two LSTM to process. Also, a BiLSTM layer is relatively easy to divide the process which leads to the availability of parallel processing. By dividing these two BiLSTM process into two parts, the processing time can be decreased: the LSTM which reads the input forward is implemented on an FPGA while the other LSTM which reads the input backward is implemented on another FPGA. In addition, we process the BiLSTM layer in parallel using 8 FPGAs by dividing each LSTM layers into 4 parts independently.


The overview of the Dual FPGA Implementation

![image](https://github.com/tinaba96/master/assets/57109730/644eec14-1ca4-40a9-8798-63bbcc6f6423)

The overview of the 8 FPGA Implementation

![image](https://github.com/tinaba96/master/assets/57109730/7971036b-98c5-4125-83b8-94a70dd50177)


- Cloud Optimization under Give Costs
My implementation can change the instance size by saving the parameters which are needed to continue the training, such as weights. This enables to control the training time with the instance usage fee. I propose a system to optimize the usage fee depending on the training time given by users. The usage fee fluctuates each time for using an instance in real cloud services such as AWS. Therefore, I introduce to change the instance size during the training according to the usage fee at that time.

![image](https://github.com/tinaba96/master/assets/57109730/1c169898-44f5-4ec8-ae9e-3fc82d9764bf)

- - -
## Result & Conclusion
- - -
- FPGA Implemetaion
There is a big task for improving the training time for deep learning, especially in the field of genomics. There are huge datasets of DNA sequences to estimate the chromatin effect.
Therefore, it is necessary to accelerate the training time, and we proposed a method of using an FPGA. We focused on BiLSTM Layer and implemented it on AWS EC2 F1 Instances.
As a result, we could accelerate the DanQ model by using a single FPGA by 1.05x compared to our CPU implementation. Besides, our implementation on 8 FPGAs gets 2.87x faster than the dual FPGA implementation and 6.00x faster than the CPU implementation.

![image](https://github.com/tinaba96/master/assets/57109730/8ca56459-6f4a-4a55-87a7-13505f2d9d32)

- Cloud Optimization
There are so many cloud users who concern the trade-off between the cloud usage fee and the execution time. This is because the cloud usage fee always changes each time.
Therefore, changing the instance size during the training depending on the cloud usage fee at that time leads to a better result in terms of the training time and the cloud instance usage fee.
Comparing a case of using 8 FPGAs for all time and a case in which we optimized the number of FPGAs during the training with our model, I obtained the result that we can save the cloud usage fee for 56.28% by only taking 16.00% extra time. Therefore, I can optimize the training time as well as the instance usage fee depending on the user’s needs.

![image](https://github.com/tinaba96/master/assets/57109730/bd51c099-913d-4744-a838-b8bdf598d4b7)



- - -
- - -

# How To

