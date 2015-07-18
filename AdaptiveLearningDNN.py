"""
A deep neural network.
"""

import numpy
import theano
import sys
import math
from theano import tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict

BATCH_SIZE = 100

def relu_f(vec):
    """ Wrapper to quickly change the rectified linear unit function """
    return (vec + abs(vec)) / 2.


def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
            name=name, borrow=True)


class Linear(object):
    """ Basic linear transformation layer (W.X + b) """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W_values *= 4  # This works for sigmoid activated networks!
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b = build_shared_zeros((n_out,), 'b')
        self.input = input
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.output = T.dot(self.input, self.W) + self.b

    def __repr__(self):
        return "Linear"


class SigmoidLayer(Linear):
    """ Sigmoid activation layer (sigmoid(W.X + b)) """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        super(SigmoidLayer, self).__init__(rng, input, n_in, n_out, W, b)
        self.pre_activation = self.output
        self.output = T.nnet.sigmoid(self.pre_activation)


class ReLU(Linear):
    """ Rectified Linear Unit activation layer (max(0, W.X + b)) """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if b is None:
            b = build_shared_zeros((n_out,), 'b')
        super(ReLU, self).__init__(rng, input, n_in, n_out, W, b)
        self.pre_activation = self.output
        self.output = relu_f(self.pre_activation)


class DatasetMiniBatchIterator(object):
    """ Basic mini-batch iterator """
    def __init__(self, x, y, batch_size=BATCH_SIZE, randomize=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        from sklearn.utils import check_random_state
        self.rng = check_random_state(42)

    def __iter__(self):
        n_samples = self.x.shape[0]
        if self.randomize:
            for _ in xrange(n_samples / BATCH_SIZE):
                if BATCH_SIZE > 1:
                    i = int(self.rng.rand(1) * ((n_samples+BATCH_SIZE-1) / BATCH_SIZE))
                else:
                    i = int(math.floor(self.rng.rand(1) * n_samples))
                yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size],
                       self.y[i*self.batch_size:(i+1)*self.batch_size])
        else:
            for i in xrange((n_samples + self.batch_size - 1)
                            / self.batch_size):
                yield (self.x[i*self.batch_size:(i+1)*self.batch_size],
                       self.y[i*self.batch_size:(i+1)*self.batch_size])


class LogisticRegression:
    """Multi-class Logistic Regression
    """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if W != None:
            self.W = W
        else:
            self.W = build_shared_zeros((n_in, n_out), 'W')
        if b != None:
            self.b = b
        else:
            self.b = build_shared_zeros((n_out,), 'b')

        # P(Y|X) = softmax(W.X + b)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.output = self.y_pred
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    #My TODO: Any particular reason we're using sum instead of mean?
    def negative_log_likelihood_sum(self, y):
        return -T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def training_cost(self, y):
        """ Wrapper for standard name """
        return self.negative_log_likelihood_sum(y)

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            print("!!! y should be of int type")
            return T.mean(T.neq(self.y_pred, numpy.asarray(y, dtype='int')))


class NeuralNet(object):
    """ Neural network (not regularized, without dropout) """
    def __init__(self, numpy_rng, theano_rng=None, 
                 n_ins=40*3,
                 layers_types=[Linear, ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=[1024, 1024, 1024, 1024],
                 n_outs=62 * 3,
                 rho=0.9, eps=1.E-6,
                 debugprint=False, mu=0.9):
        """
        Basic feedforward neural network.
        """
        self.layers = []
        self.params = []
        self.pre_activations = [] # SAG specific
        self.n_layers = len(layers_types)
        self.layers_types = layers_types
        assert self.n_layers > 0
        self._rho = rho  # ``momentum'' for adadelta
        self._eps = eps  # epsilon for adadelta
        self._mu = mu
        self._accugrads = []  # for adagrad
        self._accuDeltragrads = []  # for adadelta
        self._accudeltas = []  # for adadelta
        self._momentum = [] # for momentum
        self._sag_gradient_memory = []  # for SAG
        self._nag = [] # for nag

        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.fmatrix('x')
        self.y = T.ivector('y')
        
        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]
        
        layer_input = self.x
        
        for layer_type, n_in, n_out in zip(layers_types,
                self.layers_ins, self.layers_outs):
            this_layer = layer_type(rng=numpy_rng,
                    input=layer_input, n_in=n_in, n_out=n_out)
            assert hasattr(this_layer, 'output')
            self.params.extend(this_layer.params)
            #self.pre_activations.extend(this_layer.pre_activation)# SAG specific TODO 
            self._accugrads.extend([build_shared_zeros(t.shape.eval(),
                'accugrad') for t in this_layer.params])
            self._accuDeltragrads.extend([build_shared_zeros(t.shape.eval(),
                'accuDeltragrads') for t in this_layer.params])
            
            self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
                'accudelta') for t in this_layer.params])
            self._momentum.extend([build_shared_zeros(t.shape.eval(),
                'momentum') for t in this_layer.params])
            self._nag.extend([build_shared_zeros(t.shape.eval(),
                'nag') for t in this_layer.params])

            self._sag_gradient_memory.extend([build_shared_zeros(tuple([(x_train.shape[0]+BATCH_SIZE-1) / BATCH_SIZE] + list(t.shape.eval())), 'sag_gradient_memory') for t in this_layer.params])
            #self._sag_gradient_memory.extend([[build_shared_zeros(t.shape.eval(), 'sag_gradient_memory') for _ in xrange(x_train.shape[0] / BATCH_SIZE + 1)] for t in this_layer.params])
            #print self._accugrads[0].shape.eval()
            self.layers.append(this_layer)
            layer_input = this_layer.output

        assert hasattr(self.layers[-1], 'training_cost')
        assert hasattr(self.layers[-1], 'errors')
        # TODO standardize cost
        self.mean_cost = self.layers[-1].negative_log_likelihood(self.y)
        self.cost = self.layers[-1].training_cost(self.y)
        if debugprint:
            theano.printing.debugprint(self.cost)

        self.errors = self.layers[-1].errors(self.y)

    def __repr__(self):
        dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
                                    zip(self.layers_ins, self.layers_outs))
        return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
                            zip(self.layers_types, dimensions_layers_str)))


    def get_SGD_trainer(self):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        # using mean_cost so that the learning rate is not too dependent
        # on the batch size
        gparams = T.grad(self.mean_cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * learning_rate

        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y),
                                           theano.Param(learning_rate)],
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y})

        return train_fn
    

    def get_adagrad_trainer(self):
        """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.mean_cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, param, gparam in zip(self._accugrads, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            
            agrad = accugrad + gparam * gparam
            dx = - (learning_rate / T.sqrt(agrad + self._eps)) * gparam
            updates[param] = param + dx
            updates[accugrad] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=self.mean_cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn

    def get_adadelta_trainer(self):
        """ Returns an Adadelta (Zeiler 2012) trainer using self._rho and
        self._eps params.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.mean_cost, self.params)

        # compute list of weights updates
        updates = OrderedDict()
        for accuDeltragrads, accudelta, param, gparam in zip(self._accuDeltragrads,
                self._accudeltas, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accuDeltragrads + (1 - self._rho) * gparam * gparam
            dx = - T.sqrt((accudelta + self._eps)
                          / (agrad + self._eps)) * gparam
            updates[accudelta] = (self._rho * accudelta
                                  + (1 - self._rho) * dx * dx)
            updates[param] = param + dx
            updates[accuDeltragrads] = agrad

        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y)],
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y})

        return train_fn


    def get_momentum_trainer(self):
        """Stochastic Gradient Descent (SGD) updates with momentum.
        Generates update expressions of the form:
        * ``velocity := momentum * velocity - learning_rate * gradient``
        * ``param := param + velocity``
        Notes
        -----
        Higher momentum also results in larger update steps. To counter that,
        you can optionally scale your learning rate by `1 - momentum`.
        See Also
        --------
        apply_momentum : Generic function applying momentum to updates
        nesterov_momentum : Nesterov's variant of SGD with momentum
        """

        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.mean_cost, self.params)
        # compute list of weights updates
        updates = OrderedDict()
        
        for momentum, param, gparam in zip(self._momentum, self.params, gparams):
            currMomentum = self._mu * momentum - learning_rate * gparam
            updates[param] = param + currMomentum 
            updates[momentum] = currMomentum      
        
        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=self.mean_cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn


    def get_nag_trainer(self):
        """Stochastic Gradient Descent (SGD) updates with momentum.
        Generates update expressions of the form:
        * ``velocity := momentum * velocity - learning_rate * gradient``
        * ``param := param + velocity``
        Notes
        -----
        Higher momentum also results in larger update steps. To counter that,
        you can optionally scale your learning rate by `1 - momentum`.
        See Also
        --------
        apply_momentum : Generic function applying momentum to updates
        nesterov_momentum : Nesterov's variant of SGD with momentum
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.mean_cost, self.params)
        # compute list of weights updates
        updates = OrderedDict()        
        
        for nag, param, gparam in zip(self._nag, self.params, gparams):
            currNag = self._mu * nag - (learning_rate * gparam)
            updates[param] = self._mu*currNag + param  - (learning_rate * gparam)
            updates[nag] = currNag                    
            
        
        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=self.mean_cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})

        return train_fn


    def score_classif(self, given_set):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x),
                                        theano.Param(batch_y)],
                                outputs=self.errors,
                                givens={self.x: batch_x, self.y: batch_y})

        def scoref():
            """ returned function that scans the entire set given as input """
            return [score(batch_x, batch_y) for batch_x, batch_y in given_set]

        return scoref


class RegularizedNet(NeuralNet):
    """ Neural net with L1 and L2 regularization """
    def __init__(self, numpy_rng, theano_rng=None,
                 n_ins=100,
                 layers_types=[ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=[1024, 1024, 1024],
                 n_outs=2,
                 rho=0.9, eps=1.E-6,
                 L1_reg=0.,
                 L2_reg=0.,
                 debugprint=False, mu=0.9):
        """
        Feedforward neural network with added L1 and/or L2 regularization.
        """
        super(RegularizedNet, self).__init__(numpy_rng, theano_rng, n_ins,
                layers_types, layers_sizes, n_outs, rho, eps, debugprint, mu)

        L1 = shared(0.)
        for param in self.params:
            L1 += T.sum(abs(param))
        if L1_reg > 0.:
            self.cost = self.cost + L1_reg * L1
        L2 = shared(0.)
        for param in self.params:
            L2 += T.sum(param ** 2)
        if L2_reg > 0.:
            self.cost = self.cost + L2_reg * L2

    def fit(self, x_train, y_train, x_dev=None, y_dev=None, x_test=None, y_test=None,
            max_epochs=40, early_stopping=True, split_ratio=0.1, # TODO 100+ epochs
            method='adadelta', verbose=False, plot=False):
        """
        Fits the neural network to `x_train` and `y_train`. 
        If x_dev nor y_dev are not given, it will do a `split_ratio` cross-
        validation split on `x_train` and `y_train` (for early stopping).
        """
        import time, copy
        if x_dev == None or y_dev == None:
            from sklearn.cross_validation import train_test_split
            x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                    test_size=split_ratio, random_state=42)
        if method == 'sgd':
            train_fn = self.get_SGD_trainer()
        elif method == 'adagrad':
            train_fn = self.get_adagrad_trainer()
        elif method == 'adadelta':
            train_fn = self.get_adadelta_trainer()
        elif method == 'momentum':
            train_fn = self.get_momentum_trainer()
        elif method == 'nag':
            train_fn = self.get_nag_trainer()
        elif method == 'sag':
            #train_fn = self.get_SAG_trainer(R=1+numpy.max(numpy.sum(x_train**2, axis=1)))
            if BATCH_SIZE > 1:
                line_sums = numpy.sum(x_train**2, axis=1)
                train_fn = self.get_SAG_trainer(R=numpy.max(numpy.mean(
                    line_sums[:(line_sums.shape[0]/BATCH_SIZE)*BATCH_SIZE].reshape((line_sums.shape[0]/BATCH_SIZE,
                        BATCH_SIZE)), axis=1)),
                    alpha=1./x_train.shape[0])
            else:
                train_fn = self.get_SAG_trainer(R=numpy.max(numpy.sum(x_train**2,
                    axis=1)), alpha=1./x_train.shape[0])
        train_set_iterator = DatasetMiniBatchIterator(x_train, y_train)
        if method == 'sag':
            sag_train_set_iterator = DatasetMiniBatchIterator(x_train, y_train, randomize=True)
        dev_set_iterator = DatasetMiniBatchIterator(x_dev, y_dev) 
        test_set_iterator = DatasetMiniBatchIterator(x_test, y_test)        
        train_scoref = self.score_classif(train_set_iterator)

        dev_scoref = self.score_classif(dev_set_iterator)
        test_scoref = self.score_classif(test_set_iterator)
        best_dev_loss = numpy.inf
        best_test_loss = numpy.inf
        epoch = 0
        # TODO early stopping (not just cross val, also stop training)
        if plot:
            verbose = True
            self._costs = []
            self._train_errors = []
            self._dev_errors = []  
            self._test_errors = []          
            self._updates = []

        seen = numpy.zeros(((x_train.shape[0]+BATCH_SIZE-1) / BATCH_SIZE,), dtype=numpy.bool)
        n_seen = 0

        while epoch < max_epochs:
            if not verbose:
                sys.stdout.write("\r%0.2f%%" % (epoch * 100./ max_epochs))
                sys.stdout.flush()
            avg_costs = []
            timer = time.time()
            if method == 'sag':
                for ind_minibatch, x, y in sag_train_set_iterator:
                    if not seen[ind_minibatch]:
                        seen[ind_minibatch] = 1
                        n_seen += 1
                    avg_cost = train_fn(x, y, ind_minibatch, n_seen)
                    if type(avg_cost) == list:
                        avg_costs.append(avg_cost[0])
                    else:
                        avg_costs.append(avg_cost)
            else:
                for x, y in train_set_iterator:
                    if method == 'sgd' or method == 'adagrad' or method == 'momentum' or method == 'nag':
                        lr = numpy.asarray(1.E-2, dtype='float32')
                        #lr = numpy.asarray(learningRate, dtype='float32')
                        avg_cost = train_fn(x, y, lr=lr)
                        #avg_cost = train_fn(x, y, lr=1.E-2)
                    elif method == 'adadelta':
                        avg_cost = train_fn(x, y)
                    if type(avg_cost) == list:                        
                        avg_costs.append(avg_cost[0])
                    else:
                        avg_costs.append(avg_cost)
            if verbose:
                mean_costs = numpy.mean(avg_costs)
                mean_train_errors = numpy.mean(train_scoref())
                
                #print('  epoch %i took %f seconds' %
                #      (epoch, time.time() - timer))
                #print('  epoch %i, avg costs %f' %
                #      (epoch, mean_costs))
                #print('  epoch %i, training error %f' %
                #      (epoch, mean_train_errors))
                
                if plot:
                    self._costs.append(mean_costs)
                    self._train_errors.append(mean_train_errors)
            dev_errors = numpy.mean(dev_scoref())
            test_errors = numpy.mean(test_scoref())
            #print('  epoch %i, training error: %f, validation error: %f, test error: %f' %
            #          (epoch, mean_train_errors, dev_errors, test_errors))            
                    
            if plot:
                self._dev_errors.append(dev_errors)
                self._test_errors.append(test_errors)
            if dev_errors < best_dev_loss:
                best_dev_loss = dev_errors
                best_params = copy.deepcopy(self.params)
                if test_errors < best_test_loss:
                    best_test_loss = test_errors
                '''
                if verbose:
                    print('!!!  epoch %i, validation error of best model: %f, test error of best model: %f' %
                          (epoch, dev_errors, test_errors))
                '''
            print ".",
            epoch += 1
        print ""
        print('  epoch %i, training error: %f, validation error: %f, test error: %f' %
                      (epoch, mean_train_errors, dev_errors, test_errors))
            
        if not verbose:
            print("")
        for i, param in enumerate(best_params):
            self.params[i] = param

    def score(self, x, y):
        """ error rates """
        iterator = DatasetMiniBatchIterator(x, y)
        scoref = self.score_classif(iterator)
        return numpy.mean(scoref())        



if __name__ == "__main__":
    #add_fit_and_score(DropoutNet)
    #add_fit_and_score(RegularizedNet)

    #My TODO:  Why nudge?
    def nudge_dataset(X, Y):
        """
        This produces a dataset 5 times bigger than the original one,
        by moving the 8x8 images in X around by 1px to left, right, down, up
        """
        from scipy.ndimage import convolve
        direction_vectors = [
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [1, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]]]
        shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                      weights=w).ravel()
        X = numpy.concatenate([X] +
                              [numpy.apply_along_axis(shift, 1, X, vector)
                                  for vector in direction_vectors])
        Y = numpy.concatenate([Y for _ in range(5)], axis=0)
        return X, Y

    from sklearn import datasets, svm, naive_bayes
    from sklearn import cross_validation, preprocessing
    DIGITS = True
    FACES = True
    TWENTYNEWSGROUPS = True
    VERBOSE = True
    SCALE = True
    PLOT = True

    def train_models(x_train, y_train, x_test, y_test, n_features, n_outs,
            use_dropout=False, n_epochs=50, numpy_rng=None,
            svms=False, nb=False, deepnn=True, name=''):
  
        if deepnn:
            import warnings
            warnings.filterwarnings("ignore")  # TODO remove


            def new_regNet_dnn(muVal, epsVal, rhoVal):
                    print("Simple (regularized) DNN")
                    return RegularizedNet(numpy_rng=numpy.random.RandomState(123), n_ins=n_features,
                        #layers_types=[LogisticRegression],
                        #layers_sizes=[],
                        #layers_types=[ReLU, ReLU, ReLU, LogisticRegression],
                        #layers_sizes=[1000, 1000, 1000],
                        layers_types=[ReLU, LogisticRegression],
                        layers_sizes=[200],
                        n_outs=n_outs,
                        #L1_reg=0.001/x_train.shape[0],
                        #L2_reg=0.001/x_train.shape[0],
                        L1_reg=0.,
                        L2_reg=1./x_train.shape[0],
                        debugprint=0,
                        mu=muVal,
                        eps= epsVal,
                        rho=rhoVal)

            import matplotlib.pyplot as plt
            plt.figure(1)
            
            ax1 = plt.subplot(221)
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(223)
            ax4 = plt.subplot(224)  # TODO updates of the weights
            
            #epsVals = [1.E-2, 1.E-4, 1.E-6, 1.E-8]
            #muVals = [0.9, 0.99, 0.999]
            muVal = 0.9
            rhoVal = 0.999
            eps = 1.E-8
            muVals = [0.0, 0.9, 0.99, 0.999]            
            methods = ['sgd', 'momentum', 'nag', 'adagrad', 'adadelta']

            
            for method in methods:
                print method    
                labelStr = ""                
                dnn = new_regNet_dnn(muVal, eps, rhoVal)
                print dnn
                print labelStr
                dnn.fit(x_train, y_train, x_test=x_test, y_test=y_test, max_epochs=n_epochs, method=method, verbose=VERBOSE, plot=PLOT)                
                print("score: %f" % (1. - numpy.mean(dnn._train_errors)))
                plt.figure(1)
                ax1.plot(numpy.log10(dnn._costs), label=method+labelStr)
                ax2.plot(dnn._train_errors, label=method+labelStr)                
                ax3.plot(dnn._dev_errors, label=method+labelStr)                
                ax4.plot(dnn._test_errors, label=method+labelStr)
                #ax4.plot([test_error for _ in range(10)], label=method)
            plt.figure(1)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Cost (log10)')
            ax1.set_title('Cost(log10) vs Epoch')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Training Error')
            ax2.set_title('Training Error vs Epoch')

            ax3.set_xlabel('Epoch')            
            ax3.set_ylabel('Validation Error')
            ax3.set_title('Validation Error vs Epoch')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Test Error')
            ax4.set_title('Test Error vs Epoch')
            plt.legend()
            
            plt.legend(loc='upper right', bbox_to_anchor=(1, 2.4), fontsize="small")
            plt.tight_layout()
            plt.savefig(name+'_graph.png')


    if DIGITS:
        digits = datasets.load_digits()
        data = numpy.asarray(digits.data, dtype='float32')
        target = numpy.asarray(digits.target, dtype='int32')
        nudged_x, nudged_y = nudge_dataset(data, target)
        if SCALE:
            nudged_x = preprocessing.scale(nudged_x)
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                nudged_x, nudged_y, test_size=0.2, random_state=42)
        train_models(x_train, y_train, x_test, y_test, nudged_x.shape[1],
                     len(set(target)), numpy_rng=numpy.random.RandomState(123),
                     name='digits')

    if FACES:
        import logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s')
        lfw_people = datasets.fetch_lfw_people(min_faces_per_person=50,
                                               resize=0.4)
        X = numpy.asarray(lfw_people.data, dtype='float32')
        if SCALE:
            X = preprocessing.scale(X)
        y = numpy.asarray(lfw_people.target, dtype='int32')
        target_names = lfw_people.target_names
        print("Total dataset size:")
        print("n samples: %d" % X.shape[0])
        print("n features: %d" % X.shape[1])
        print("n classes: %d" % target_names.shape[0])
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                    X, y, test_size=0.2, random_state=42)

        train_models(x_train, y_train, x_test, y_test, X.shape[1],
                     len(set(y)), numpy_rng=numpy.random.RandomState(123),
                     name='faces')

    if TWENTYNEWSGROUPS:
        from sklearn.feature_extraction.text import TfidfVectorizer
        newsgroups_train = datasets.fetch_20newsgroups(subset='train')
        vectorizer = TfidfVectorizer(encoding='latin-1', max_features=10000)
        #vectorizer = HashingVectorizer(encoding='latin-1')

        x_train = vectorizer.fit_transform(newsgroups_train.data)
        x_train = numpy.asarray(x_train.todense(), dtype='float32')
        y_train = numpy.asarray(newsgroups_train.target, dtype='int32')
        newsgroups_test = datasets.fetch_20newsgroups(subset='test')
        x_test = vectorizer.transform(newsgroups_test.data)
        x_test = numpy.asarray(x_test.todense(), dtype='float32')
        y_test = numpy.asarray(newsgroups_test.target, dtype='int32')
        print "Train: "
        print("n samples: %d" % x_train.shape[0])
        print("n features: %d" % x_train.shape[1])
        print("n classes: %d" % len(newsgroups_train.target_names))
        print "Test: "
        print("n samples: %d" % x_test.shape[0])
        print("n features: %d" % x_test.shape[1])
        print("n classes: %d" % len(newsgroups_test.target_names))
        

        train_models(x_train, y_train, x_test, y_test, x_train.shape[1],
                     len(set(y_train)),
                     numpy_rng=numpy.random.RandomState(123),
                     svms=False, nb=True, deepnn=True,
                     name='20newsgroups')
