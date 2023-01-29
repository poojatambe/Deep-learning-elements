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

Transpose= convolution + Upssample
