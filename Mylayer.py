from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K
import numpy as np

try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations

class MyLayer(Layer):

    def __init__(self, weights=None, axis=-1, beta_init = 'zero', gamma_init = 'one', momentum = 0.9, **kwargs):
        # 参数**kwargs代表按字典方式继承父类
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.Zeros()
        self.gamma_init = initializations.Ones()
        self.initial_weights = weights
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        # 1：InputSpec(dtype=None, shape=None, ndim=None, max_ndim=None, min_ndim=None, axes=None)
        #Docstring:     
        #Specifies the ndim, dtype and shape of every input to a layer.
        #Every layer should expose (if appropriate) an `input_spec` attribute:a list of instances of InputSpec (one per input tensor).
        #A None entry in a shape is compatible with any dimension
        #A None shape is compatible with any shape.

        # 2:self.input_spec: List of InputSpec class instances
        # each entry describes one required input:
        #     - ndim
        #     - dtype
        # A layer with `n` input tensors must have
        # an `input_spec` of length `n`.

        shape = (int(input_shape[self.axis]),)

        # Compatibility with TensorFlow >= 1.0.0
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))

        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        out = np.array(out)*np.array(out)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(MyLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def get_output_shape_for(self,input_shape):
        return(input_shape[0]+65)





class MyLayerW(Layer):

    def __init__(self, weights=None, axis=-1, beta_init = 'zero', gamma_init = 'one', momentum = 0.9, **kwargs):
        # 参数**kwargs代表按字典方式继承父类
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.Zeros()
        
        self.gamma_init = initializations.Ones()
        self.initial_weights = weights
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        # 1：InputSpec(dtype=None, shape=None, ndim=None, max_ndim=None, min_ndim=None, axes=None)
        #Docstring:     
        #Specifies the ndim, dtype and shape of every input to a layer.
        #Every layer should expose (if appropriate) an `input_spec` attribute:a list of instances of InputSpec (one per input tensor).
        #A None entry in a shape is compatible with any dimension
        #A None shape is compatible with any shape.

        # 2:self.input_spec: List of InputSpec class instances
        # each entry describes one required input:
        #     - ndim
        #     - dtype
        # A layer with `n` input tensors must have
        # an `input_spec` of length `n`.

        shape = (int(input_shape[self.axis]),)

        # Compatibility with TensorFlow >= 1.0.0
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))

        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(MyLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))






























