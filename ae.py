# AutoEncoders
'''
An Autoencoder is a model for unsupervised learning.

Choose an autoencoder when:

1. You have an unsupervised learning problem,

2. You need to do dimensionality reduction on large dimensional input data.

3. You need  feature extraction

4. You want to do train generative models on the dataset.
'''


# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', sep = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', sep = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, number_of_nodes): # number_of_nodes - number of input nodes and numbeê ùà output nodes should be the same for Auto-Encoder
        super(SAE, self).__init__()  #  built-in “super” function returns a proxy object to delegate method calls to a class 
        self.fc1 = nn.Linear(number_of_nodes, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, number_of_nodes)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
		
		
sae = SAE(nb_movies)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # The SAE class inherits method parameters from the nn.Module class 

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # PyTorch doesn't accept a single 1d vector as  input so add another dimension (a fake dimension/batch) so we can create this new dimension in PyTorch with Variable()  and we also have to use .unsqueeze to specify the index of the dimension
        target = input.clone()
		#  if an observation contains only zeros,  i.e. the user didn't rate any movies, skip this observation
		# target.data is older ratings of this user here at the loop right now
        if torch.sum(target.data > 0) > 0: 
		    # SAE class inherits from the nn.Module that has a __call__ subroutine
			# Like __init__ it is a special kind of subroutine - it executes anytime a class is constructed
			# Inside __call__ there is a forward() call, which will get executed as is, unless there is own forward() in SAE class
            output = sae(input) #  ~ output = sae.forward(input) , where input - existed real ratings, output - vector of predicted ratings
            target.require_grad = False # no need to compute the gradient with respect to the target ( to save memory and optimize code)
            output[target == 0] = 0 # to not penalize users, that haven't seen a movie, the output is set to 0
			#  take the same indexes of the ratings that were equal to zero in the input vector (i.e. user didn't rate those movies)  and set these indexes to 0 to exclude non-rating movies from the calculation of error so they won't have impact on the updates of the different weights after having measured the error
        
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # compute MSE, your denominator is sample size(n), and in this case,  number of movie is total number of movies. Yet, for each user, he or she definitely didn't watch all 1682 movies, so the real sample size is the number of movies which he or she haven watched
            loss.backward() # backward decides the direction to which the weight will be updated (increased or decreased)
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
			# optimizer.zero_grad() 
            optimizer.step() # apply the optimizer to update the weights, i.e. optimizer step decides intensity of the weight updates while direction is controlled by loss.backward()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)  # target is  real ratings of the test set
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False 
        output[target == 0] = 0 
		loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))


# Making predictions
target_user_id = 123
target_movie_id = 327
input = Variable(training_set[target_user_id-1]).unsqueeze(0)
output = sae(input)
output_numpy = output.data.numpy()
print (''+ str(output_numpy[0,target_movie_id-1]))