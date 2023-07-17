# Research in Kanazawa University
# Quantization of Convolutional Neural Network for Motion Estimation for FPGA Implementation

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

 In addition, I first replace upconvolution with convolution and upsampling using bilinear interpolation. Then I perform batch normalization after convolution to determine the range to be quantized. By modifying in this way, it became possible to quantize the expanding part as same as the contracting part. Moreover, since batch normalization cannot be applied before flow prediction, flow prediction is not performed in each layer of the expanding part in the modified network. Hence, flow prediction is performed only in the output layer. That is, the concatenation in each layer of the expanding part is not the output of layers of the contracting part, the output of the previous layer and the predicted flow, but only the output of layers of the contracting part and the output of the previous layer.
 

### Quantization for the weights
In the quantization of weights, since it is known that the average of the weights distribution is `0` and the standard deviation is smaller than `1`, I set the quantization range from `-1` to `1`. As the process flow, I first quantize weights and save unquantized weights. Next, forward and backward calculations are executed. After carrying out forward and backward calculations with quantized weights, the unquantized weights are restored. Finally, the weights are updated using unquantized weights. That is, at the time of learning, inference and error back propagation are performed using the quantized weights, and the weights before quantization are updated using the obtained gradients.



- - -
## Result
- - -
The Table shows the average of L1Loss and EPE for the entire dataset when inferred. L1Loss and EPE (Endpoint Error) were used as the evaluation index of the experimental results. L1Loss represents the pixel average of the difference absolute value with the ground truth flow. EPE, on the other hand, is a pixel average representing Euclidean distance between the estimated flow and the ground truth flow.
A comparison of the (unquantized) FlowNetSimple with the modified (unquantized) FlowNetSimple shows that the modified network is slightly less accurate. In addition, it was found that when the modified network was quantized to the bit-width from 8 bits to 6 bits, the accuracy did not decrease compared to the unquantized networks. On the other hand, it was found that when quantization was performed with the bit- width of 5, the precision dropped sharply. It turned out that the performances of the CNN with the bit-width of 4 or lower were very poor and the estimated flows were also extremely bad in quality that are impossible to recognize the figure shown in the ground truth flow.

![image](https://github.com/tinaba96/master_old/assets/57109730/333ebbf9-33d0-49dc-affc-9c56e2557950)

The results of the predicted flows using a set of input images as an example is shown in the below figure. The numerical values of L1Loss and EPE described in the each predicted flow represents the error between the predicted flow and the ground truth flow. The figure illustrates that the deterioration was rapid for the predicted flow with the bit-width of 5. However, there were no big differences between the predicted flows with the bit-width of 8 to 6 and (unquantized) FlowNetSimple as well as the modified (unquantized) FlowNetSimple.

![image](https://github.com/tinaba96/Quantization-Flownet2-Pytorch/assets/57109730/b024da22-c39a-4a5f-8d4c-ebfec0b169cc)


- - -
## Conclusion
- - -
It is known that networks for motion estimation need a large amount of calculations and take time to process. Therefore, I quantized the network for motion estimation. The network FlowNetSimple is modified to be easy to quantize, and the weights and activations of the network are quantized in the range of −1 to 1 and 0 to 4, respectively. As a result of training the quantized network, the precision deteriorated rapidly with the bit-width of 5 and remained in poor performance with the bit- width of 4 or lower. From this result, it was found that motion estimation requires more bits than semantic segmentation which has high precision up to the bit-width of 2.



- - -
- - -

# How To

## Install
First, please install by running
```
#install custom layers
bash install.sh
```

## Requirement
Please prepare following packages
```
- numpy
- PyTorch ( == 0.4.1, for <= 0.4.0 see branch python36-PyTorch0.4)
- scipy
- scikit-image
- tensorboardX
- colorama, tqdm, setproctitle
```

## Training and validation
To train the model and validate using MPISintel, please run 
```
# Example on MPISintel Final and Clean, with L1Loss on FlowNet2 model
python main.py --batch_size 8 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 --training_dataset MpiSintelFinal --training_dataset_root /path/to/mpi-sintel/final/dataset --validation_dataset MpiSintelClean --validation_dataset_root /path/to/mpi-sintel/clean/dataset

# Example on MPISintel Final and Clean, with MultiScale loss on FlowNet2C model 
python main.py --batch_size 8 --model FlowNet2C --optimizer=Adam --optimizer_lr=1e-4 --loss=MultiScale --loss_norm=L1 --loss_numScales=5 --loss_startScale=4 --optimizer_lr=1e-4 --crop_size 384 512 --training_dataset FlyingChairs --training_dataset_root /path/to/flying-chairs/dataset --validation_dataset MpiSintelClean --validation_dataset_root /path/to/mpi-sintel/clean/dataset
```

## Inference
To infrence the model, please run
```
# Example on MPISintel Clean
python main.py --inference --model FlowNet2 --save_flow --inference_dataset MpiSintelClean --inference_dataset_root /path/to/mpi-sintel/clean/dataset --resume /path/to/checkpoints
```


note: This is a Pytorch implementation of quantized network for flownet implemented in (here)[https://github.com/NVIDIA/flownet2-pytorch.git]  
note: The algorithm I used for this quantization is similar to (here)[https://github.com/jiecaoyu/XNOR-Net-PyTorch]
