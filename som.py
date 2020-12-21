# Self Organizing Maps

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
df = pd.read_csv("Credit_Card_Applications.csv")

print(df)

"""So first of all, all these customers are the input,
these customers are the inputs of our neural network.
And then what happens is that these input points
are going to be mapped to a new output space.
And between the input space and the output space,
we have this neural network composed of neurons,
each neuron being initialized as a vector of weights
that is the same size as the vector of customer
that is a vector of 15 elements,
because we have the customer ID plus 14 attributes.
And so for each observation point that is for each customer,
the output of this customer, will be the neuron
that is the closest to the customer.
So basically, in the network, we pick the neuron
that is the closest to the customer.
And remember, this neuron is called the winning node,
for each customer, the winning node is the most similar
neuron to the customer, then, you know,
we use a neighborhood function like
the galch neighborhood function,
to update the weight of the neighbors of the winning node
to move them closer to the point.
And we do this for all the customers in the input space.
And we'll repeat that again.
We'll repeat all this many times.
And each time we'll repeat it, the output space decreases
and loses dimensions.
It reduces its dimension little by little.
And then it reaches a point where the neighborhood
stops decreasing, where the output space stops decreasing.
And that's the moment where we obtained our
self organizing map in two dimensions
with all the winning nodes that were eventually identified.
"""

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

X = sc.fit_transform(X)
 
# Training the SOM
from minisom import MiniSom 
# Here for the dimensions of the grid, there are not much customers i.e. not much observations so we make a 10 X 10 grid.
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)

# Initialize the weights
som.random_weights_init(X)

#Train the SOM on X
som.train_random(X, 100)
# Executes quickly as the dataset is small

# 2 dimensional grid that will contain all the final winning nodes, for each of this winning nodes we will get MID- Mean Interneuron Distance

"""MID of a specific winning node is the mean of the distances of all the neurons around the winning node
inside a neighborhood defined,sigma here, which is the radius of this neighborhood.
Higher is the MID the more the winning node will be far away from it's neighbors, inside a neighborhood.
Therefore the higher the MID,the more the winning node is an outlier. And since in some way
the majority of the winning nodes represent the rules that are respected.
If far from this majority of neurons, then far from the general rules,and that is how ouliers are detected
that is this the frauds. Because for each neuron we get the MID, so we simply need to take the winning nodes
that have the highest MID."""

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)

"""Different colors corresponding to the different range values of the Mean Interneuron Distances.
And to do this use the pcolor function, and inside this pcolor function add all the values of the Mean Interneuron Distances
for all the winning nodes of our self-organizing map. And to get these mean distances,
use the method Distance Map Method, which returns all the Mean Interneuron Distancein one matrix.
So this will return the matrix of all these distances for all the winning nodes.
The pcolor function, we need to take the transpose of this matrix returned by the Distance Map Method.
To take the transpose, just add here .T, the transpose of this MID matrix.
Scale to the right is the range of values of the MID (Mean Interneuron Distances).
These are normalized values, that means that the values were scaled from zero to one, and therefore the highest MIDs,
correspond to the white color, and on the other hand, the smallest Mean Interneuron Distances
correspond to the dark colors. Dark colors are close to each other because their MID is pretty low.
So that means that all the winning nodes in the neighborhood of one winning node are close to this
winning node at the center and therefore that creates clusters of winning nodes
all close to each other. But, these winning nodes here have large MIDs and therefore they're outliers and accordingly potential frauds, the white blocks"""

# Red color - no approval
# Green color - approved

colorbar()
markers = ['o','s']
colors = ['r','g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2) #each winning node is represented by square in the self organizing map
show()

"""We have the Mean Interneuron Distances but also we get the information
of whether the customers got approval or didn't get approval for each of the winning nodes.
For example if we have a look at the winning node, 4th row from top 7th column from left, it can be seen that the customers associated
to this winning node didn't get approval.
However, if we look at 3rd row from top 7th column from left winning node we can see
that the customers associated to this winning node got approval, but the color is fine its around the level
So it doesn't indicate a high risk of fraud. However, looking at our outliers, the obvious outliers that are the white square winning node.
Here, absolutely no doubt the Mean Interneuron Distance is almost equal to one or perhaps equal to one.
Which clearly indicates that there is a high risk of fraud for these customers associated to these two winning nodes.
So now, we have to is catch these potential cheaters in the winning nodes but in priority those who got approval
because it's of course much more relevant to the bank to catch the cheaters who got away with this."""

# Finding the frauds
mappings = som.win_map(X) # only the data on which the som was trained

frauds = np.concatenate((mappings[(4,8)],mappings[(1,5)]), axis = 0) # list of all fraud customers

frauds = sc.inverse_transform(frauds)