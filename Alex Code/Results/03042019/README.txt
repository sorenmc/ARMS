This one is an attempt to make a deeper (streamlined) network. Before the sequence pooling layer,
there is a convolutional layer with a smaller window size (5 for example) and max pooling with the same size, to
create a smaller sequence. The idea is that this is the equivalent of trying larger and larger numbers of filters
in the one-layer model. 

Here I am using three-fold cross validation