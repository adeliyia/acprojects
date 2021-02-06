import numpy as np
import pandas as pd





#Import Train Data, Define Data------------------------------------------------------------------
MNIST = pd.read_csv('mnist_train_0_4.csv', header=None)
#Input, remove label column 784 cols(12665.784)
input_vals = MNIST.drop(MNIST.columns[[0]], axis=1)
#NOrmalize
X = input_vals.to_numpy()/255






#Expected Output, Dimension = (12665,1)
output_vals = MNIST.iloc[:, 0]
output_col = output_vals.to_numpy()
#converting to 2-d array
Y = np.reshape(output_col, (-1, 1))

#dividing train outout data by 4
Y = Y/4



#--------------------------------------------------------------------------------------------

#Import Test Data
MNIST_test = pd.read_csv('mnist_test_0_4.csv', header=None)
#Input, Dimension = (12664,784), put inouts into a list and Normalize
input_vals_test = MNIST_test.drop(MNIST_test.columns[[0]], axis=1)
#NOrmalize
X_test = input_vals_test.to_numpy()/255
print ('\n Input:')
print(X_test)

#Expected Output, Dimension = (12664,1)
output_vals_test = MNIST_test.iloc[:, 0]
output_col_test = output_vals_test.to_numpy()
#converting to 2-d array
Y_test = np.reshape(output_col_test, (-1, 1))
print ('\n Actual Output:')
print(Y_test)

#------------------------------------------------------------------------------------------------
# defining the Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

# derivative of Sigmoid Function
def sigmoid_der(x):
    return x * (1 - x)



def predict(x):
    xtw = x.dot(weight1) + bias1 
    
    output1 = sigmoid(xtw) 
      
    # Output layer 
    z = output1.dot(weight2) + bias2
    output2 = sigmoid(z) 
    return(output2)


def accuracy(x,y):
    k = 0

    for i, j in zip(x,y):
        if i == j:
            k+=1
        


    return ((k/len(x))*100)

#-------------------------------------------------------------------------------------------------------
# initializing the variables
epoch=1000 # number of training iterations
alpha= 0.001 # learning rate
inputlayer_neurons = X.shape[1] # number of features in data set
hiddenlayer_neurons = 4 # number of hidden layers neurons
output_neurons = 1 # number of neurons at output layer

# initializing weight and bias
weight1=np.random.uniform(low = -1, high =1,size=(inputlayer_neurons,hiddenlayer_neurons))

weight2=np.random.uniform(low = -1, high =1, size=(hiddenlayer_neurons,output_neurons))

bias1=np.random.uniform(size=(1,hiddenlayer_neurons))

bias2= np.random.uniform(size=(1,output_neurons))

# training the model------------------------------------------
for i in range(epoch):

    #Forward Propogation

    xtw = X.dot(weight1) + bias1
 
    output1 = sigmoid(xtw)
   

    z = output1.dot(weight2) + bias2
    output2 = sigmoid(z)

    #Backpropagation
    
    rawerror = Y-output2

    slope_hidden = sigmoid_der(output1)

    slope_out = sigmoid_der(output2)
    
    delta_output = rawerror  *  slope_out

    
    error_hidden = delta_output.dot(weight2.transpose())

    
    delta_hidden = error_hidden * slope_hidden
    
    #Weight Update
    w1_adj = X.transpose().dot(delta_hidden)
    weight1 = weight1+(alpha*(w1_adj))
    
    w2_adj = output1.transpose().dot(delta_output)
    weight2 = weight2+(alpha*(w2_adj))

    #Bias Update
    bias1 += np.sum(delta_hidden, axis=0,keepdims=True) *alpha
    bias2 += np.sum(delta_output, axis=0,keepdims=True) *alpha
    

#Prediction----------------------------------------------------------------
predicted_vals = predict(X_test)
print('\n Predicted (UnRounded) Output:',predicted_vals)

#Times 4
new_vals = predicted_vals*4
print('\n Predicted (UnRounded) Output times 4:',new_vals)

#Round the predictions up or down as needed
rounded_vals = np.round(new_vals)

print('\n Predicted (Rounded) Output:', rounded_vals)

#Accuracy vs actual output Y
print(' \n Accuracy:', accuracy(rounded_vals, Y_test), '%')

