### AIM : 






Manually compute forward and backward pass for a single sigmoid neuron and update 
weights. 









### ALGORITHM :  







Compute gradient of cross-entropy loss and update weights via gradient descent.








### THEORY: 









Back propagation is chain rule application in neural nets; cross-entropy + sigmoid 
simplifies gradients.









### INPUT :



import numpy as np

# Single sample x, target y

x = np.array([0.5, -1.2])

y = 1.0

w = np.array([0.1, -0.2]); b = 0.0

lr = 0.1

def sigmoid(z): return 1/(1+np.exp(-z))

# Forward pass

z = np.dot(w, x) + b

a = sigmoid(z)

loss = - (y*np.log(a) + (1-y)*np.log(1-a))

# Backward pass (gradients)

dz = a - y

dw = dz * x

db = dz

# Update weights

w_new = w - lr * dw

b_new = b - lr * db

print('a (output):', round(a,4), 'loss:', round(loss,4))

print('dw:', np.round(dw,4), 'db:', round(db,4))

print('Updated w:', np.round(w_new,4), 'Updated b:', round(b_new,4))





### OUTPUT :



a (output): 0.572 loss: 0.5586

dw: [-0.214   0.5136] db: -0.428

Updated w: [ 0.1214 -0.2514] Updated b: 0.0428









 ### RESULT : 









Learners observe numeric gradients and weight updates; demystifies 
backpropagation.
