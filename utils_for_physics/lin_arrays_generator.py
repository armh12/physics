import random
import numpy as np

f1 = lambda x, al1, al2: al1 * np.sin(al2 * x)
f2 = lambda x, al1, al2: al1 * np.exp(-al2 * x)
f3 = lambda x, al1, al2: al1 * np.exp(-al2 * x)
f4 = lambda x, al1, al2: al1 * x + al2
f5 = lambda x, al1, al2, al3: al1 * x * x + al2 * x + al3

pi = np.pi
xmin=0.
xmax=10.

xmin1=10.
xmax1=12.
smin=-0.05
smax=0.1
delmin=0.0025
delmax=0.005
ntot=1000

xx1=[]
yy1=[]
eyy1=[]


x1=np.random.random([1,ntot])
x1=xmin+(xmax-xmin)*x1
x2=np.random.random([1,ntot])
x2=xmin1+(xmax1-xmin1)*x2
x3=np.random.random([1,ntot])
x4=np.random.random([1,ntot])
y1=f5(x1,0.1,3.,5.)*(1.+delmin+delmax*x3)
y2=f5(x2,0.1,3.,5.)*(1.+delmin+delmax*x4)
ey1=0.05*abs(y1)
ey2=0.05*abs(y2)
ey3=0.15*abs(y1)
print(x1)
print(y1)

for j in range(ntot):
    xx1.append(xmin+(xmax-xmin)*random.random())
    xx1.sort
    yy1.append(f5(xx1[j],0.01,3.,5.)*(1.+delmin+delmax*random.random()))
    eyy1.append(0.05*f5(xx1[j],0.01,3.,5.))

