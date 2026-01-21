import matplotlib.pyplot as plt 

def predict(w,x,b):
    return w*x+b

def compute_loss(y_true,y_pred):
    total_error=0
    n=len(y_true)

    for i in range(n):
        error=y_pred[i]-y_true[i]
        total_error+=error**2

    return total_error/n

x=[0,1,2,3,4,5,6,7,8]
y=[20,40,50,55,60,65,70,80,90]

w,b=10,20

y_pred=[predict(w,i,b) for i in x]

mse=compute_loss(y,y_pred)
print("Mean squared error is",round(mse,2))

plt.plot(x,y,label="Actual")
plt.plot(x,y_pred,label="Predicted")
plt.legend()
plt.savefig("plots/day2_prediction_vs_actual.png")
plt.show()