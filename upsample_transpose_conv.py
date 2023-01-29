import torch

########################################################## upsample #####################################################
input = torch.arange(11, 15, dtype=torch.float32).view(1,1,2,2)
print('Input: ',input)
n = torch.nn.Upsample(scale_factor=2, mode='nearest')
print('upsample (nearest) and factor 2: \n',n(input))
b = torch.nn.Upsample(scale_factor=2, mode='bilinear')
print('upsample (bilinear) and factor 2: \n',b(input))
b1 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
print('upsample (bilinear) and factor 2 with align_corners: \n',b1(input))

################################################### convtransposed2d ##############################################
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).view(1,1,2,2)
K = torch.tensor([[1.0, 0.0], [0.0, 0.0]]).view(1,1,2,2)
tconv = torch.nn.ConvTranspose2d(1, 1, kernel_size=1, bias=False)
tconv.weight.data = K
print('input: \n', X)
print('kernel: \n', K)
print('Transposed convolution stride 1 no padding: \n',tconv(X))

tconv = torch.nn.ConvTranspose2d(1, 1, kernel_size=1, padding=1, bias=False)
tconv.weight.data = K
print('Transposed convolution stride 1 padding 1:\n',tconv(X))

tconv = torch.nn.ConvTranspose2d(1, 1, kernel_size=1, stride=2, bias=False)
tconv.weight.data = K
print('Transposed convolution stride 2 no padding:\n ',tconv(X))

X = torch.rand(size=(1, 10, 16, 16))
conv = torch.nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = torch.nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
# print(conv(X))
print(tconv(conv(X)).shape == X.shape)
print(conv(X).shape)
print(tconv(conv(X)).shape)