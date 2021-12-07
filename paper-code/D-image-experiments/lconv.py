from tensorflow import reduce_sum, concat, reduce_max
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import deserialize

from numpy import newaxis,prod


class L_module(Layer):
    def __init__(self, n_L, out_dim = None, hidden_units = [], activation = 'linear', **kws):
        super(L_module, self).__init__(**kws) 
        self.params = dict(n_L = n_L, activation = activation, hidden_units = hidden_units)
        self.out_dim = out_dim

        
    def build(self, input_shape):
        # print(input_shape)
        d = prod(input_shape[1:])
        
        space_dim, feature_dim = input_shape[-2:]
        out_dim = self.out_dim or space_dim #int(space_dim/self.stride)
        
        n_L = self.params['n_L']
        hidden_units = self.params['hidden_units']
        
        self.hidden_layers = []
        in_dim = space_dim 
        for i,u in enumerate(hidden_units):
            self.hidden_layers += [self.add_weight(shape=(n_L, u, in_dim), 
                               initializer='glorot_normal', trainable=True, name='Lh_%d' %i)]
            in_dim = u
        
        self.L = self.add_weight(shape=(n_L, out_dim, in_dim), initializer='glorot_normal', trainable=True, name='L')
        
    def call(self, inputs):
        n_L = self.params['n_L']
        
        x = inputs
        
        for l in self.hidden_layers:
            x = l @ x
        x = self.L @ x
        act = deserialize(self.params['activation'])
        return act(x)
    
class L_Conv(Model):
    def __init__(self, 
                 num_filters: int, 
                 kernel_size: int, 
                 #stride: int = 1,
                 activation = 'relu',
                 L_hid =[], L_act = 'linear',
                ):
        """Assumes channel last input x: (batch, space, features). 
        Uses stride to to scale space dimension: out_dim = int(space/stride).
        call: L @ (x @ W + b) 
        """
        super(L_Conv, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = 1 # stride
        self.activation = activation
        self.L_params = dict(n_L = kernel_size-1, hidden_units = L_hid, activation = L_act)

    def get_L(self, input_shape):
        # assume channel last
        space_dim, feature_dim = input_shape[-2:]
        out_dim = int(space_dim/self.stride)
        
        # num_L = kernel-1 b/c original input will be concat
#         L = self.add_weight(shape=(self.kernel_size - 1, out_dim, space_dim), 
#                                initializer='glorot_normal',
#                                trainable=True, name='L')
        L = L_module(out_dim = out_dim, **self.L_params)
        return L
        
    def build(self, input_shape):
        self.L = self.get_L(input_shape)
        self.w = self.add_weight(shape=(self.kernel_size, input_shape[-1], self.num_filters),
                               initializer='glorot_normal',
                               trainable=True, name = 'w')
        self.b = self.add_weight(shape=(self.kernel_size,1, self.num_filters),
                               initializer='zeros',
                               trainable=True, name = 'b')
        self.activation_layer = deserialize(self.activation)

    def call(self, inputs):
        x0 = inputs[:,newaxis]

        # (batch, space, features) --> (batch, 1, space, features) 
        #x = self.L @ x0
        x = self.L(x0)
        # (batch, 1, space, features) --> (batch, kernel_size, space/stride, features) 

        x = concat([x, x0], axis = 1) # add back the original 

        x = x @ self.w + self.b
        # (batch, kernel_size, space, features) --> (batch, kernel_size, space, num_filters) 

        x = reduce_sum(x, axis = 1)

        return self.activation_layer(x)
    
class L_Conv_max(L_Conv):
    def __init__(self, stride: int = 1, **kws):
        """Does max_i(L_i L_j x)
        """
        super(L_Conv_max, self).__init__(**kws)
    
    def call(self, inputs):
        x0 = inputs[:,newaxis]
        
        #print(x0.shape)
        # (batch, space, features) --> (batch, 1, space, features) 
        #x = self.L @ x0
        x = self.L(x0)
        # (batch, 1, space, features) --> (batch, kernel_size, space, features) 
        
        #print(x.shape)
        x = concat([x, x0], axis = 1) # add back the original 
        
        #print(x.shape)
        # apply L again 
        x1 = self.L(x[:,:,newaxis])
        # (batch, kernel_size, space, features) -->  (batch, kernel_size, kernel_size, space, features) 
        
        #print(x1.shape, x.shape)
        x = concat([x1, x[:,:,newaxis]], axis = 2) # add back the original 
        
        x = x @ self.w + self.b
        # (batch, kernel_size, kernel_size, space, features) --> (batch, kernel_size, kernel_size, space, num_filters) 
        
        x = reduce_sum(x, axis = 1)
        # (batch, kernel_size, space, features)
        
        # max pooling
        x = reduce_max(x, axis = 1)
        # (batch, space, features)

        return self.activation_layer(x)
    
    
class L_Conv_strided(L_Conv):
    def __init__(self, stride: int = 1, **kws):
        """Assumes channel last input x: (batch, space, features). 
        Uses stride to to scale space dimension: out_dim = int(space/stride).
        call: L @ (x @ W + b) 
        """
        super(L_Conv_strided, self).__init__(**kws)
        self.stride = stride
        self.L_params['n_L'] += 1 # to make up for lack of residual conn.

    def call(self, inputs):
        x0 = inputs[:,newaxis]

        # (batch, space, features) --> (batch, 1, space, features) 
        #x = self.L @ x0
        x = self.L(x0)
        # (batch, 1, space, features) --> (batch, kernel_size, space/stride, features) 

#         x = tf.concat([x, x0], axis = 1) # add back the original 

        x = x @ self.w + self.b
        # (batch, kernel_size, space, features) --> (batch, kernel_size, space, num_filters) 

        x = reduce_sum(x, axis = 1)

        return self.activation_layer(x)