import numpy as np
import cv2
import os
import sys


filename = sys.argv[1] 

#Network Parameters

epochs = 500
alpha = 0.005

n_input = 34*34
n_hidden1 = 256
n_hidden2 = 100
n_output = 5
n_train = 1000


#Activation Function 
def sigmoid(array):
 sig = 1/(1+(np.exp(-array)))
 return sig

#Random initialization of weights
def network_init(n_input,n_hidden1,n_hidden2,n_output):
 epsilon_init = 0.12
 w1 = (np.random.rand(n_input+1,n_hidden1)*2*epsilon_init)-epsilon_init
 w2 = (np.random.rand(n_hidden1+1,n_hidden2)*2*epsilon_init)-epsilon_init
 w3 = (np.random.rand(n_hidden2+1,n_output)*2*epsilon_init)-epsilon_init
 
 return w1,w2,w3

#Conversion of images into stacked numpy arrays (matrix form)
def data_init():
 dict_list = ['A','B','C','D','E']

 X_train = np.empty([n_output,int(n_train/n_output),34,34])
 Y_train = np.empty([n_output,int(n_train/n_output),n_output])
 iteration = 0
 for i in dict_list:
   folder = (filename + "/{0}/".format(i))
   count = 0
   for file in os.listdir(folder):
     img = cv2.imread(os.path.join(folder,file),0)
     if img is not None and count<200:
       X_train[iteration,count] = np.array(img)
       Y_train[iteration,count] = 0
       Y_train[iteration,count,iteration] = 1
     count += 1
   iteration += 1

 X_train = X_train/255
 X_train = X_train.reshape([n_train,n_input])
 Y_train = Y_train.reshape([n_train,n_output])
 return X_train,Y_train

# Feed forward using the trained weights
def predict(w1,w2,w3,X):
 X_bias = np.insert(X,0,[1],axis=1)
 hidden = np.matmul(X_bias,w1)
 hidden_final = sigmoid(hidden)

 hidden_final_bias = np.insert(hidden_final,0,[1],axis=1)
 output1 = np.matmul(hidden_final_bias,w2)
 output1 = sigmoid(output1)
 
 output1_bias = np.insert(output1,0,[1],axis=1)
 output2 = np.matmul(output1_bias,w3)
 final_output = sigmoid(output2)
 
 print("The likelihood that the test character is A is {0}%" .format(final_output.item(0)*100/final_output.sum()))
 print("The likelihood that the test character is B is {0}%" .format(final_output.item(1)*100/final_output.sum()))
 print("The likelihood that the test character is C is {0}%" .format(final_output.item(2)*100/final_output.sum()))
 print("The likelihood that the test character is D is {0}%" .format(final_output.item(3)*100/final_output.sum()))
 print("The likelihood that the test character is E is {0}%" .format(final_output.item(4)*100/final_output.sum()))

################################################# Training ##############################################################################

#After the training is executed once, the weights are stored in weightsN.npy.
# In the next execution, the training algo checks for these weights mentioned in File1, File2, File3 variable
# If weights are detected, then testing is implemented.

def training():
  File1 = (filename + "/weights11.npy")
  File2 = (filename + "/weights22.npy")
  File3 = (filename + "/weights33.npy")

  X,Y = data_init()
  w1,w2,w3 = network_init(n_input,n_hidden1,n_hidden2,n_output)

  if not ((os.path.exists(File1)) and (os.path.exists(File2)) and (os.path.exists(File3))):
    for i in range(epochs):
      #feed forward
      X_bias = np.insert(X,0,[1],axis=1)
      hidden = np.matmul(X_bias,w1)
      hidden_final = sigmoid(hidden)

      hidden_final_bias = np.insert(hidden_final,0,[1],axis=1)
      output1 = np.matmul(hidden_final_bias,w2)
      output1 = sigmoid(output1)
 
      output1_bias = np.insert(output1,0,[1],axis=1)
      output2 = np.matmul(output1_bias,w3)
      final_output = sigmoid(output2)

      w3_nobias = np.delete(w3,1,axis=0)
      w2_nobias = np.delete(w2,1,axis=0)
      w1_nobias = np.delete(w1,1,axis=0)
 
      #Backpropogation
      delta_output = (final_output -Y)*final_output*(1-final_output)
      w3 = w3 - alpha*np.matmul(output1_bias.transpose(),delta_output)

      delta_temp = np.matmul(delta_output,w3_nobias.transpose())
      delta_hidden_2 = delta_temp*(output1)*(1-output1)
      w2 = w2 - alpha*np.matmul(hidden_final_bias.transpose(),delta_hidden_2)

      delta_temp = np.matmul(delta_hidden_2,w2_nobias.transpose())
      delta_hidden_1 = delta_temp*(hidden_final)*(1-hidden_final)
      w1 = w1 - alpha*np.matmul(X_bias.transpose(),delta_hidden_1)

      cost = np.square(final_output-Y)/n_input
      cost = (cost.sum())
      print(cost)
  
  #save trained weights
    np.save('weights11', w1)
    np.save('weights22', w2)
    np.save('weights33', w3)

###################################################### Testing #############################################################################
def testing():
  #Update the location of w1,w2,w3 according to your PC location
  w1 = np.load(filename + "/weights11.npy")
  w2 = np.load(filename + "/weights22.npy")
  w3 = np.load(filename + "/weights33.npy")

  test = cv2.imread(filename + "A/4955.jpg",0)
  X_test = np.array(test.reshape([1,n_input]))
  predict(w1,w2,w3,X_test)

def main():
  training()
  testing()
  

if __name__ == "__main__":
    main()
