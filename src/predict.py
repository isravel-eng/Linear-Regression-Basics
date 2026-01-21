# a predicting model
def model(m,x,c):
  return m*x+c
# Day 1
import matplotlib.pyplot as plt

x=[0,1,2,3,4,5,6,7,8]
y=[20,40,50,55,60,65,70,80,90]

def predict(m,x,c):
    return m*x+c

m,c=10,20

y_pred=[predict(m,i,c) for i in x]

plt.subplot(1,2,1)
plt.plot(x,y)
plt.subplot(1,2,2)
plt.plot(x,y_pred)
plt.savefig("day1_line.png")
plt.show()
