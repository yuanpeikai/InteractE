import torch
import torch.nn as nn

class A(nn.Module):
    def __init__(self):
        super(A,self).__init__()
        self.dorp=nn.Dropout2d(0.5)

    def forward(self,x):
        return self.dorp(x)

model=A()
model.train()
x=torch.tensor([[1.,2.,3.,4.,5.],[1.,2.,3.,4.,5.]])
x=model(x)
print(x)
model.eval()
y=torch.tensor([[1.,2.,3.,4.,5.],[1.,2.,3.,4.,5.]])
y=model(y)
print(y)