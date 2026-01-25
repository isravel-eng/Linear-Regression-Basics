# Linear Regression 
# import library 
import matplotlib.pyplot as plt 

# define functions 
def predict(x, weight, bias):
    return weight*x + bias

def compute_loss(y_true, y_pred):
    return sum((y_pred[i]-y_true[i])**2 for i in range(len(y_true))) / len(y_true)

def compute_gradients(x, y_true, y_pred):
    n = len(x)
    dw = sum((y_pred[i]-y_true[i]) * x[i] for i in range(n)) * (2/n)
    db = sum((y_pred[i]-y_true[i]) for i in range(n)) * (2/n)
    return dw, db


def update_parameters(weight,bias,dw,db,learning_rate):
    weight=weight-dw*learning_rate
    bias=bias-db*learning_rate
    return weight,bias

# dataset
x=[0,1,2,3,4,5,6,7,8]
y=[20,40,50,55,60,65,70,80,90]

# starts with bad weight, bias 
w,b=0,0
lr=0.01
losses=[] 

# model training loop
for i in range(30):
    y_pred=[predict(j,w,b) for j in x]
    loss=compute_loss(y,y_pred)
    losses.append(loss)
    dw,db=compute_gradients(x,y,y_pred)
    w,b=update_parameters(w,b,dw,db,lr)

#ploting the graphs
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.title("Linear Regression : Actual vs Prediction (after training)")
plt.plot(x,y,label="Actual")
plt.plot(x,y_pred,label="Predicted")
plt.xlabel("Independent variable - input")
plt.ylabel("Dependent variable - output")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.title("Training loss over iterations")
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.tight_layout()
plt.savefig("day6_subplots.png")
plt.show()