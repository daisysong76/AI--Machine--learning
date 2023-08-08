import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)
        #self.w.transpose(); DotProduct;self.get_weights()
        #nn.DotProduct(features, weights)
    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        node = self.run(x)
        sign = nn.as_scalar(node)
        if sign >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        updateOrnot = 1#true
        while (updateOrnot):
            updateOrnot = 0
            for x, y in dataset.iterate_once(batch_size):
            #an entire pass over the data se
            #(a) Classify the sample using the current weights,
            #let y be the class predicted by your current w:
            #Compare the predicted label y to the true label y
                predict_y = self.get_prediction(x)
                if nn.as_scalar(y) != predict_y:
                #do nothing
                #parameter.update(direction, multiplier)
                    self.w.update(x, nn.as_scalar(y))
                    #weights←weights+direction⋅multiplier
                    #The direction argument is a Node with the same shape as the parameter,
                    #and the multiplier argument is a Python scalar.
                    updateOrnot = 1
           #print(x)#print(y) #break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hiddenLayerSize = 50 # randomly choose 10-400
        self.learningRate = 0.001 # bigger then process will faster
        self.weight1 = nn.Parameter(1,50) #batch_size is 1
        self.weight2 = nn.Parameter(50,1)
        self.hiddenLayerNum = 2
        self.bias1 = nn.Parameter(1,50)
        self.bias2 = nn.Parameter(1,1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = nn.Linear(x, self.weight1) #wx
        x = nn.AddBias(x,self.bias1) #
        x = nn.ReLU(x)#
        x = nn.Linear(x, self.weight2)
        x = nn.AddBias(x,self.bias2)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y,y) # this loss function to prodict if the y value correct or not

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        updateOrnot = 1#true
        while (updateOrnot):
            updateOrnot = 0
            for x, y in dataset.iterate_once(batch_size):
                predict_y = self.run(x)
                loss = self.get_loss(x,y)
                # parameters: a list (or iterable) containing Parameter nodes
                # Output: a list of Constant objects, representing the gradient of the loss
                # with respect to each provided parameter.
                gradient_weight1,gradient_weight2,gradient_bias1, gradient_bias2 = nn.gradients(loss, [self.weight1,self.weight2,self.bias1,self.bias2])

                #Your implementation will receive full points if it gets a loss of 0.02 or
                #better, averaged across all examples in the dataset. You may use the training
                #loss to determine when to stop training (use nn.as_scalar to convert a loss node
                # to a Python number). Note that it should take the model a few minutes to train.
                #if w not converged do: #????
                if nn.as_scalar(loss) >= 0.02:
                #m.update(grad_wrt_m, multiplier)
                    self.weight1.update(gradient_weight1, -self.learningRate)
                    self.weight2.update(gradient_weight2, -self.learningRate)
                    self.bias1.update(gradient_bias1, -self.learningRate)
                    self.bias2.update(gradient_bias2, -self.learningRate)
                    updateOrnot = 1


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.a = nn.Parameter(784, 150) # 784 because 784 dimensiaonal vector; randomly chose 150 layers
        self.b = nn.Parameter(1, 150) # weight1's bias
        self.c = nn.Parameter(150, 10) # weight 2
        self.d = nn.Parameter(1, 10)#10 because 10 digits; weight2's bias

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        relu = nn.ReLU(nn.AddBias(nn.Linear(x, self.a), self.b))
        return nn.AddBias(nn.Linear(relu, self.c), self.d)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y) #DigitClassificationModel to find max probability

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        threshold = 0.98
        accuracy = dataset.get_validation_accuracy()
        while accuracy < threshold:
            for x, y in dataset.iterate_once(500):
                temp = [self.a, self.b, self.c, self.d]
                loss = self.get_loss(x, y)
                a_gradient, b_gradient, c_gradient, d_gradient = nn.gradients(loss, temp)
                self.a.update(a_gradient, -0.1)
                self.b.update(b_gradient, -0.1)
                self.c.update(c_gradient, -0.1)
                self.d.update(d_gradient, -0.1)
            accuracy = dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.weight1 = nn.Parameter(47, 150)# 26 letters, The hidden size should be sufficiently large
        self.bias1 = nn.Parameter(1, 150)
        self.weight2 = nn.Parameter(150, 5) # 5 kinds of language
        self.weight_hiden = nn.Parameter(150, 150) # 5 kinds of language
        self.bias2 = nn.Parameter(1, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #the first layer of f initial will begin by multiplying the vector x0 by some weight
        #matrix W to product z0 = x0*w
        first_layer = nn.Linear(xs[0], self.weight1)
        #for subsequent letters, you should replace this with zi = xi*w + hi * W_hidden
        for i in range(len(xs)):
            second_layer = nn.Linear(first_layer, self.weight_hiden)
            i_layer = nn.Linear(xs[i], self.weight1)
            sum_layer = nn.Add(i_layer, second_layer)
            Z_i = nn.ReLU(nn.AddBias(sum_layer, self.bias1))
            first_layer = Z_i

        H_i = nn.Linear(first_layer, self.weight2)
        return nn.AddBias(H_i, self.bias2)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        threshold = 0.85
        accuracy = dataset.get_validation_accuracy()
        learningRate = 0.02
        while accuracy < threshold:
            for x, y in dataset.iterate_once(500):
                temp = [self.weight1, self.bias1, self.weight2, self.bias2]
                loss = self.get_loss(x, y)
                weight1_gradient, bias1_gradient, weight2_gradient, bias2_gradient = nn.gradients(loss, temp)
                self.weight1.update(weight1_gradient, -learningRate)
                self.bias1.update(bias1_gradient, -learningRate)
                self.weight2.update(weight2_gradient, -learningRate)
                self.bias2.update(bias2_gradient, -learningRate)
            accuracy = dataset.get_validation_accuracy()
