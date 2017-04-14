# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
#from numpy import genfromtxt


RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

#This method reads from the 'X' and 'Y' file and gives in the input as an array of arrays (aka if the input dim is 5 and there are 10 training sets, the input is a 10X 5 array)
#a a a a a       3 3 3 3 3 
#b b b b b       4 4 4 4 4
#c c c c c       5 5 5 5 5

def get_data(test_file="test.csv",file_val="test_val.csv"):
    #trainX = np.array([])
    train_X = np.reshape(np.transpose(np.genfromtxt('test_val.csv', delimiter=',')),(-1,1))

    train_Y = np.transpose(np.genfromtxt('test.csv', delimiter=','))
    print("X shape: " , train_X.shape)
    print("Y shape: " , train_Y.shape)

    #train_X = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5]])
    #train_Y = np.array([[2,2],[3,3],[4,4],[5,5],[6,6]])
    return train_X, train_Y 

def main():
    train_X, train_Y = get_data()
    print("Train_X: " , train_X)

    #train_X, test_X, train_y, test_y = get_iris_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 200                # Number of hidden nodes
    y_size = train_Y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    
    # Backward propagation
    cost = tf.reduce_mean(tf.square(y-yhat))
    #Output float values)
    #cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0005, decay=0.5).minimize(cost)
    #updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    n_batch = 5

    n_iter = 100000000000
    step = 0

    numEpochs=5

    curEpoch=0
    #print("Train x shape: " , train_X.shape)

    while curEpoch < numEpochs:


        # Train with each example
        #for i in range(len(train_X)):

        #Now I need to deal with epochs. So
        

        batch_x = train_X[step * n_batch : (step+1) * n_batch]
        batch_y = train_Y[step * n_batch : (step+1) * n_batch]
        #print("Batch x: " , batch_x)

        sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})

        loss = sess.run(cost,feed_dict={X:batch_x,y:batch_y})
        step += 1
        if step == int(train_X.shape[0]/n_batch):
            step = 0
            numEpochs +=1
            print("Epoch: " , numEpochs)

            print("Step = %d, train loss = %.2f"
              % (step + 1, loss))
            if (numEpochs % 500 == 0 or numEpochs == 1):
                myvals0 = sess.run(yhat,feed_dict={X:train_X[0:1],y:train_Y[0:1]})[0]

                myvals1 = sess.run(yhat,feed_dict={X:train_X[1:2],y:train_Y[1:2]})[0]

                myvals2 = sess.run(yhat,feed_dict={X:train_X[-2:-1],y:train_Y[-2:-1]})[0]

                print("Outputing trained results")
                output_file = "save_vals" +str(numEpochs) + ".txt"

                f = open(output_file, 'w')
                f.write("XValue\nActual\nPredicted\n")
                #f.write("Train_X:")
                f.write(str(train_X[0][0]))
                f.write("\n")
                #f.write("\n")
                for item in list(train_Y[0]):
                    f.write(str(item) + ",")
                f.write("\n")
                for item in list(myvals0):
                    f.write(str(item) + ",")
                f.write("\n")

                #f.write("Train_X:")
                f.write(str(train_X[1][0]))
                #f.write("\n")
                f.write("\n")
                for item in list(train_Y[1]):
                    f.write(str(item) + ",")
                f.write("\n")
                for item in list(myvals1):
                    f.write(str(item) + ",")
                f.write("\n")

                #f.write("Train_X:")
                f.write(str(train_X[-1][0]))
                f.write("\n")
                for item in list(train_Y[-1]):
                    f.write(str(item) + ",")
                f.write("\n")
                for item in list(myvals2):
                    f.write(str(item) + ",")
                f.write("\n")

                f.flush()
                f.close()
                #os.exit()

    sess.close()

if __name__ == '__main__':
    main()