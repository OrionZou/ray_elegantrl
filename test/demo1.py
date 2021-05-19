import torch
import torch.optim as optim
from torch.autograd import Variable

w = Variable(torch.randn(3, 5), requires_grad=True) # 这里变量的定义需要注意, 是需要梯度的
b = Variable(torch.randn(3), requires_grad=True)
x = Variable(torch.randn(5))
y = Variable(torch.randn(3))

optimizer = optim.SGD([w,b], lr=0.01) # 我们要优化的是w和b两个参数

num_epochs = 100
for epoch in range(num_epochs):

    y_pred = torch.mv(w, x) + b # torch.mv表示矩阵与向量相乘
    loss = ((y_pred-y)**2).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(loss)

y_pred = torch.mv(w, x) + b
print('y: \n', y)
print('y_pred: \n', y_pred)