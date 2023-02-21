# Deep-learning-elements


The upsample and transpose convolution layers increase the dimension of input matrix.

**Upsample**

![image](https://user-images.githubusercontent.com/64680838/215352704-b7840780-e901-4922-9fe1-015c22882131.png)


Example:
```
Input= tensor([[[[11., 12.],
          [13., 14.]]]])
          
Upscale by factor 2=  tensor([[[[11., 11., 12., 12.],
                          [11., 11., 12., 12.],                        
                          [13., 13., 14., 14.],                          
                          [13., 13., 14., 14.]]]])     
```                          
                          
Upsampling has no trainable parameters.

**Transposed Convolution Layer**

![image](https://user-images.githubusercontent.com/64680838/215352749-6392cb74-472e-49a6-9dda-f69bcb14674f.png)


Example:
```
Input=  tensor([[[[1., 2.],
          [3., 4.]]]])
          
Kernel=  tensor([[[[1., 0.],
          [0., 0.]]]])
          
Transpose Convolution with stride 2=   tensor([[[[1., 0., 2., 0.],
                                          [0., 0., 0., 0.],
                                          [3., 0., 4., 0.],
                                          [0., 0., 0., 0.]]]], grad_fn=<SlowConvTranspose2DBackward>)
                                          
```

Transpose convolution has trainable parameters.

Transpose= convolution + Upsample


**1 X 1 Convolution**

As we increase layers in deep neural network, width, height and depth of feature map increases. The pooling layer downsamples feature maps by reducing width and height. The depth of feature map indicates number of feature maps which is reduced using 1 x 1 convolution layer.

In convolution,

* Each filter creates one feature map.
* Filter has same depth or number of channels as input.

1 x 1 convolution layer is called channelwise pooling. It is used for **projecting feature map** when number of filters are same as depth of previous layer, **reducing no of feature map** when number of filters are less than previous layer depth, and **increasing no of feature maps** when number of filters are more than previous layer.

![image](https://user-images.githubusercontent.com/64680838/220333299-2de9d042-3856-4081-82de-c4000606cad2.png)


