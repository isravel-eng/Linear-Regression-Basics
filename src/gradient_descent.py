def predict(w,x,b):
    return w*x+b

def compute_loss(y_true,y_pred):
    total_error=0
    n=len(y_true)

    for i in range(n):
        error=y_pred[i]-y_true[i]
        total_error+=error**2

    return total_error/n

def compute_gradients(w,b,x,y_true,y_pred):
    dw,db=0,0
    n=len(x)

    for i in range(n):
        error=y_pred[i]-y_true[i]
        dw+=error*x[i]
        db+=error

    dw=(2/n)*dw
    db=(2/n)*db

    return dw,db

def update_parameters(w,b,dw,db,lr):
    w=w-lr*dw
    b=b-lr*db

    return w,b

x=[0,1,2,3,4,5,6,7,8]
y=[20,40,50,55,60,65,70,80,90]

w,b=0,0
lr=0.01

losses=[]

for i in range(20):
    y_pred=[predict(w,j,b) for j in x]
    loss=compute_loss(y,y_pred)
    losses.append(loss)

    dw,db=compute_gradients(w,b,x,y,y_pred)
    w,b=update_parameters(w,b,dw,db,lr)
    

import matplotlib.pyplot as plt

plt.title("Training loss over iterations")
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Mean squared error")
plt.grid(True)
plt.savefig("day5_loss_labelled.png")
plt.show()