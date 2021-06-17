## This file implements all the code for pi-VAE model.

import tensorflow as tf
#print(tf.__version__)
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from keras.models import Model
from keras import losses
from keras.layers.core import Lambda
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.activations import softplus
import numpy as np
from keras.callbacks import LearningRateScheduler

eps = 1e-7;

############# Section I: define utility functions #############
def slice_func(x, start, size):
    """Utility function. We use it to take a slice of tensor start from 'start' with length 'size'. Search tf.slice for detailed use of this function.
    
    """
    return tf.slice(x, [0,start],[-1,size])

def perm_func(x, ind):
    """Utility function. Permute x with given indices. Search tf.gather for detailed use of this function.
    """
    return tf.gather(x, indices=ind, axis=-1);

squeeze_func = Lambda(lambda x: K.squeeze(x, 1));

############# Section II: define encoders and decoders #############
## The following three functions define GIN volume preserving flow
def first_nflow_layer(z_input, dim_x):
    """Define the first layer in GIN flow, which maps z to the cancatenation of z and t(z), t is parameterized by NN. 
    This is equivalent to GIN model with input as z1:dim_z padding dim_x - dim_z zeros.
    
    # Arguments
        z_input: latents z
        dim_x: dimension of observations x
        
    # Important hyperparameters to adjust:
        gen_nodes: (integer) number of node used in each hidden layer
        act_func: (list) activation functions used in the hidden layers and output layer
        
    # Returns
        output (tensor): output of the first layer in GIN flow.
    """
    gen_nodes = dim_x//4;
    dim_z = z_input.shape.as_list()[-1];
    n_nodes = [gen_nodes, gen_nodes, dim_x-dim_z];
    act_func = ['relu', 'relu', 'linear'];
    n_layers = len(n_nodes);
    output = z_input;
    
    for ii in range(n_layers):
        output = layers.Dense(n_nodes[ii], activation=act_func[ii])(output);
    
    output = layers.concatenate([z_input, output], axis=-1);
    return output

def affine_coupling_layer(layer_input, dd=None):
    """Define each affine_coupling_layer, which maps input x to [x_{1:dd}, x_{dd+1:n} * exp(s(x_{1:dd})) + t(x_{1:dd})].
    
    # Arguments
        layer_input: input of affine_coupling_layer.
        dd: dimension of input which keeps the same after applying this layer. Default is None, which will set dd as half of the layer_input dimension if the input dimension is an even number, or set dd as the closest integer if the input dimension is an odd number.
        
    # Important hyperparameters to adjust:
        n_nodes: (list) number of nodes in the hidden layer and output layer of functions s and t. Both s and t map from dd to dim(layer_input)-dd.
        act_func: (list) activation functions used in the hidden layers and output layer of functions s and t.
        
    # Returns
        output (tensor): output of an affine_coupling_layer.
    """
    DD = layer_input.shape.as_list()[-1];
    if dd is None:
        dd = (DD//2);
    
    ## define some lambda functions
    clamp_func = Lambda(lambda x: 0.1*tf.tanh(x));
    trans_func = Lambda(lambda x: x[0]*tf.exp(x[1]) + x[2]);
    sum_func = Lambda(lambda x: K.sum(-x, axis=-1, keepdims=True));
    
    ## compute output for s and t functions, both from dd to DD-dd
    x_input1 = Lambda(slice_func, arguments={'start':0,'size':dd})(layer_input);
    x_input2 = Lambda(slice_func, arguments={'start':dd,'size':DD-dd})(layer_input);
    st_output = x_input1;
    
    n_nodes = [DD//4, DD//4, 2*(DD-dd)-1];
    act_func = ['relu', 'relu', 'linear'];
    for ii in range(3):
        st_output = layers.Dense(n_nodes[ii], activation = act_func[ii])(st_output);
    s_output = Lambda(slice_func, arguments={'start':0,'size':DD-dd-1})(st_output);
    t_output = Lambda(slice_func, arguments={'start':DD-dd-1,'size':DD-dd})(st_output);
    s_output = clamp_func(s_output); ## make sure output of s is small
    s_output = layers.concatenate([s_output, sum_func(s_output)], axis=-1); ## enforce the last layer has sum 0
    
    ## perform transformation
    trans_x = trans_func([x_input2, s_output, t_output]);
    output = layers.concatenate([trans_x, x_input1], axis=-1);
    return output

def affine_coupling_block(x_output, dd=None):
    """Define affine_coupling_block, which contains two affine_coupling_layer.
    
    # Returns
        output (tensor): output of a GIN block (affine_coupling_block).
    """
    for _ in range(2):
        x_output = affine_coupling_layer(x_output, dd);
    return x_output

## decoder
## decoder using GIN flow, used in our paper
def decode_nflow_func(z_input, n_blk, dim_x, mdl, dd=None):
    """Define mean(p(x|z)) using GIN volume preserving flow.
    
    # Arguments
        z_input: latents z
        n_blk: number of affine_coupling_block used in normalizing flow
        dim_x: dimension of observations x
        mdl: observation model. If 'poisson', add a softplus transformation to the output; if 'gaussian', directly return output.
        dd: dimension which keeps the same after applying affine_coupling_layer. Check affine_coupling_layer function for more details. Default is None, will set it in affine_coupling_layer function.
        
    # Important hyperparameters to adjust:
        n_blk
        dd
        
    # Returns
        output (tensor): output of the decoder network, i.e. mean(p(x|z)).
    """
    
    ## generate permutation indices
    permute_ind = [];
    for ii in range(n_blk):
        np.random.seed(ii);
        permute_ind.append(tf.convert_to_tensor(np.random.permutation(dim_x)));
    
    ## Get output through first_nflow_layer 
    output = first_nflow_layer(z_input, dim_x);
    
    ## First permute the input before passing it to each GIN block (affine_coupling_block). Repeat this procedure n_blk times.
    for ii in range(n_blk):
        output = Lambda(perm_func, arguments={'ind':permute_ind[ii]})(output);
        output = affine_coupling_block(output, dd);
    
    ## Get the final output, if 'poisson' observation, the output is the firing rate as a function of z_input; if 'gaussian' observation, the output is the mean of gaussian as a function of z_input.
    if mdl == 'poisson':
        softplus_func = Lambda(lambda x: softplus(x));
        output = softplus_func(output)
        
    return output

## decoder using monotone increasing nn [initial try, didn't use it eventually]
def decode_func(z_input, gen_nodes, dim_x, mdl):
    
    n_nodes = [gen_nodes, gen_nodes, dim_x];
    if mdl == 'poisson':
        act_func = ['tanh', 'tanh', 'softplus'];
    else:
        act_func = ['tanh', 'tanh', 'linear'];
    n_layers = len(n_nodes);
    output = z_input;
    
    for ii in range(n_layers):
        output = layers.Dense(n_nodes[ii], activation=act_func[ii])(output)
    
    return output

## encoder
def encode_func(x_input, gen_nodes, dim_z): 
    """Define mean or log of variance of q(z|x).
    
    # Arguments
        x_input: observations x
        gen_nodes: number of nodes in the hidden layer of encoder network
        dim_z: dimension of latents z
        
    # Important hyperparameters to adjust:
        act_func: (list) activation functions used in the hidden layers and output layer
        n_layers: number of hidden layers + output layer, now fix as 3 which means that we have 2 hidden layers and 1 output layer
    # Returns
        output (tensor): output of the encoder network. We can get the mean of q(z|x) and log of variance of q(z|x) separately by calling this function twice.
    
    """
    n_nodes = [gen_nodes, gen_nodes, dim_z];
    act_func = ['tanh', 'tanh', 'linear'];
    n_layers = len(n_nodes);
    output = x_input;
    #output = layers.concatenate([x_input, u_input], axis=-1);
    
    for ii in range(n_layers):
        output = layers.Dense(n_nodes[ii], activation=act_func[ii])(output)
    
    return output

## The following two functions define the prior of p(z|u) for continuous u and discrete u respectively.
def z_prior_nn(u_input, dim_z):
    """Compute the prior mean and log of variance of prior p(z|u) for continuous u. 
    We assume p(z|u) as gaussian distribution with mean and log of variance parameterized by feed-forward neural network as a function of u.
    
    # Arguments
        u_input (tensor): input labels
        dim_z: dimension of latent z
    
    # Important hyperparameters to adjust:
        n_hidden_nodes_in_prior: number of nodes used in the hidden layer of prior network, now fix as 20 (mean and log of variance share these 2 hidden layers)
        act_func: (list) activation functions used in the hidden layers and output layer
        n_layers: number of hidden layers + output layer, now fix as 3 which means that we have 2 hidden layers and 1 output layer
        
    # Returns
        mean and log of variance of prior p(z|u) (tensors)
    """
    n_hidden_nodes_in_prior = 20
    dim_u = u_input.shape.as_list()[-1];
    n_nodes = [n_hidden_nodes_in_prior, n_hidden_nodes_in_prior, 2*dim_z];
    act_func = ['tanh', 'tanh', 'linear'];
    n_layers = len(n_nodes);
    output = u_input;
    
    for ii in range(n_layers):
        output = layers.Dense(n_nodes[ii], activation=act_func[ii])(output)
        
    ## split the last layer as lam_mean and lam_log_var
    lam_mean = Lambda(slice_func, arguments={'start':0,'size':dim_z})(output);
    lam_log_var = Lambda(slice_func, arguments={'start':dim_z,'size':dim_z})(output);
    return lam_mean, lam_log_var

def z_prior_disc(u_input, dim_z, num_u):
    """Compute the prior mean and log of variance of prior p(z|u) for discrete u. 
    We assume p(z|u) as gaussian distribution with mean and log of variance treated as different real numbers for different u.
    
    # Arguments
        u_input (tensor): input labels
        dim_z: dimension of latent z
        num_u: number of different labels u
        
    # Returns
        mean and log of variance of prior p(z|u) (tensors)
    """
    lam_mean = squeeze_func(layers.Embedding(num_u, dim_z, input_length=1)(u_input))
    lam_log_var = squeeze_func(layers.Embedding(num_u, dim_z, input_length=1)(u_input))
    return lam_mean, lam_log_var

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of q(z|x)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def compute_posterior(args):
    """Compute the full posterior of q(z|x, u). We assume that q(z|x, u) \prop q(z|x)*p(z|u). Both q(z|x) and p(z|u) are gaussian distributed.
    
    # Arguments
        args (tensor): mean and log of variance of q(z|x) and p(z|u)
        
    # Returns
        mean and log of variance of q(z|x, u) (tensor)
    """
    z_mean, z_log_var, lam_mean, lam_log_var = args;
    # q(z) = q(z|x)p(z|u) = N((mu1*var2+mu2*var1)/(var1+var2), var1*var2/(var1+var2));
    post_mean = (z_mean/(1+K.exp(z_log_var-lam_log_var))) + (lam_mean/(1+K.exp(lam_log_var-z_log_var)));
    post_log_var = z_log_var + lam_log_var - K.log(K.exp(z_log_var) + K.exp(lam_log_var));
    
    return [post_mean, post_log_var]

############# Section III: define pi-vae model #############
def vae_mdl(dim_x, dim_z, dim_u, gen_nodes, n_blk=None, mdl='poisson', disc=True, learning_rate=5e-4):
    """Define pi-vae model.
    
    # Arguments
        dim_x: dimension of observations x.
        dim_z: dimension of latents z.
        dim_u: dimension of input labels u.
        gen_nodes: number of nodes in the hidden layer of encoder (which maps x to z).
        n_blk: number of flow blocks used in the decoder (which maps z to x).
        mdl: type of observations. Currently support 'poisson' and 'gaussian'.
        disc: Boolean. Whether the input labels are discrete (True) or continuous (False).
        learning_rate: learning_rate used to optimze the loss function of the pi-vae model.
        
    # Returns
        vae: (tensorflow model) the pi-vae model.
    """
    ### discrete u, or continuous u (one-hot) as input

    ### define input layer
    x_input = layers.Input(shape=(dim_x,));
    z_input = layers.Input(shape=(dim_z,));

    ### define prior distribution p(z|u) [gaussian]
    if disc:
        u_input = layers.Input(shape=(1,));
        lam_mean, lam_log_var = z_prior_disc(u_input, dim_z, dim_u);
    else:
        u_input = layers.Input(shape=(dim_u,));
        lam_mean, lam_log_var = z_prior_nn(u_input, dim_z);
    
    ### define encoder model
    z_mean = encode_func(x_input, gen_nodes, dim_z);
    z_log_var = encode_func(x_input, gen_nodes, dim_z);

    post_mean, post_log_var = Lambda(compute_posterior)([z_mean, z_log_var, lam_mean, lam_log_var]);
    z_sample = Lambda(sampling)([post_mean, post_log_var]);
    encoder = Model(inputs = [x_input, u_input], outputs = [post_mean, post_log_var, z_sample,z_mean,z_log_var,lam_mean, lam_log_var], name='encoder')

    ### define decoder model
    if n_blk is not None: # use nflow
        fire_rate = decode_nflow_func(z_input, n_blk, dim_x, mdl);
    else: # this else part has been deprecated 
        fire_rate = decode_func(z_input, gen_nodes, dim_x, mdl);
    if mdl == 'poisson': # clip the value of fire_rate to make it more numerically stable.
        clip_func = Lambda(lambda x: K.clip(x, min_value=1e-7, max_value=1e7));
        fire_rate = clip_func(fire_rate);
    
    decoder = Model(inputs = [z_input], outputs = [fire_rate], name='decoder')
    
    ### run encoder and decoder to define vae
    post_mean, post_log_var, z_sample, z_mean, z_log_var, lam_mean, lam_log_var = encoder([x_input, u_input])
    fire_rate = decoder([z_sample])
    if mdl == 'gaussian':
        # if gaussian observation, set the observation noise level as different real numbers and optimize it in loss function.
        one_tensor = layers.Input(tensor=(tf.ones((1,1))))
        obs_log_var = layers.Dense(dim_x, activation='linear', use_bias=False, name='obs_noise')(one_tensor);
        #obs_log_var = clip_func(obs_log_var);
        vae = Model(inputs = [x_input, u_input, one_tensor], outputs = [post_mean, post_log_var, z_sample,fire_rate, lam_mean, lam_log_var, z_mean, z_log_var, obs_log_var], name='vae')
    elif mdl == 'poisson':
        vae = Model(inputs = [x_input, u_input], outputs = [post_mean, post_log_var, z_sample,fire_rate, lam_mean, lam_log_var, z_mean, z_log_var], name='vae')

    ### define loss function to optimize
    # min -log p(x|z) + E_q log(q(z))-log(p(z|u))
    # cross entropy
    # q (mean1, var1) p (mean2, var2)
    # E_q log(q(z))-log(p(z|u)) = -0.5*(1-log(var2/var1) - (var1+(mean2-mean1)^2)/var2)
    # E_q(z|x,u) log(q(z|x,u))-log(p(z|u)) = -0.5*(log(2*pi*var2) + (var1+(mean2-mean1)^2)/var2)
    # p(z) = q(z|x) = N(f(x), g(x)) parametrized by nn;
    
    if mdl == 'poisson':
        obs_loglik = K.sum(fire_rate - x_input*tf.log(fire_rate), axis=-1)
    elif mdl == 'gaussian':
        obs_loglik = K.sum(K.square(fire_rate - x_input)/(2*tf.exp(obs_log_var)) + (obs_log_var/2), axis=-1);
    
    kl_loss = 1 + post_log_var - lam_log_var - ((K.square(post_mean-lam_mean) + K.exp(post_log_var))/K.exp(lam_log_var));
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(obs_loglik + kl_loss)
    vae.add_loss(vae_loss)
    
    ### define optimizer
    optimizer = Adam(lr = learning_rate);
    vae.compile(optimizer=optimizer)
    
    print(vae.summary())
    return vae

def custom_data_generator(x_all, u_one_hot):
    while True:
        for ii in range(len(x_all)):
            yield ([x_all[ii], u_one_hot[ii]], None)
            
############# Section IV: simulate data #############
########### Code below is to simulate the data used in our paper. TODO: separate the code below to another file ################
def realnvp_layer(x_input):
    DD = x_input.shape.as_list()[-1]; ## DD needs to be an even number
    dd = (DD//2);
    
    ## define some lambda functions
    clamp_func = Lambda(lambda x: 0.1*tf.tanh(x));
    trans_func = Lambda(lambda x: x[0]*tf.exp(x[1]) + x[2]);
    sum_func = Lambda(lambda x: K.sum(-x, axis=-1, keepdims=True));
    
    ## compute output for s and t functions
    x_input1 = Lambda(slice_func, arguments={'start':0,'size':dd})(x_input);
    x_input2 = Lambda(slice_func, arguments={'start':dd,'size':dd})(x_input);
    st_output = x_input1;
    
    n_nodes = [dd//2, dd//2, DD];
    act_func = ['relu', 'relu', 'linear'];
    for ii in range(len(act_func)):
        st_output = layers.Dense(n_nodes[ii], activation = act_func[ii])(st_output);
    s_output = Lambda(slice_func, arguments={'start':0,'size':dd})(st_output);
    t_output = Lambda(slice_func, arguments={'start':dd,'size':dd})(st_output);
    s_output = clamp_func(s_output); ## keep small values of s
    
    ## perform transformation
    trans_x = trans_func([x_input2, s_output, t_output]);
    output = layers.concatenate([trans_x, x_input1], axis=-1);
    return output

def realnvp_block(x_output):
    for _ in range(2):
        x_output = realnvp_layer(x_output);
    return x_output

def simulate_data(length, n_cls, n_dim):
    ## simulate 2d z
    np.random.seed(888);
    mu_true = np.random.uniform(-5,5,[2,n_cls]);
    var_true = np.random.uniform(0.5,3,[2,n_cls]);
    
    u_true = np.array(np.tile(np.arange(n_cls), int(length/n_cls)), dtype='int');
    z_true = np.vstack((np.random.normal(mu_true[0][u_true], np.sqrt(var_true[0][u_true])),
                        np.random.normal(mu_true[1][u_true], np.sqrt(var_true[1][u_true])))).T;

    z_true = np.hstack((z_true, np.zeros((z_true.shape[0],n_dim-2))));
    
    ## simulate mean
    dim_x = z_true.shape[-1];
    permute_ind = [];
    n_blk = 4;
    for ii in range(n_blk):
        np.random.seed(ii);
        permute_ind.append(tf.convert_to_tensor(np.random.permutation(dim_x)));
    
    x_input = layers.Input(shape=(dim_x,));
    x_output = realnvp_block(x_input);
    for ii in range(n_blk-1):
        x_output = Lambda(perm_func, arguments={'ind':permute_ind[ii]})(x_output);
        x_output = realnvp_block(x_output);
    
    realnvp_model = Model(inputs=[x_input], outputs=x_output);
    mean_true = realnvp_model.predict(z_true)
    lam_true = np.exp(2*np.tanh(mean_true));
    return z_true, u_true, mean_true, lam_true

def simulate_cont_data(length, n_dim):
    ## simulate 2d z
    np.random.seed(777);
    
    u_true = np.random.uniform(0,2*np.pi,size = [length,1]);
    mu_true = np.hstack((u_true, 2*np.sin(u_true)));
    z_true = np.random.normal(0, 0.6, size=[length,2])+mu_true;
    z_true = np.hstack((z_true, np.zeros((z_true.shape[0],n_dim-2))));
    
    ## simulate mean
    dim_x = z_true.shape[-1];
    permute_ind = [];
    n_blk = 4;
    for ii in range(n_blk):
        np.random.seed(ii);
        permute_ind.append(tf.convert_to_tensor(np.random.permutation(dim_x)));
    
    x_input = layers.Input(shape=(dim_x,));
    x_output = realnvp_block(x_input);
    for ii in range(n_blk-1):
        x_output = Lambda(perm_func, arguments={'ind':permute_ind[ii]})(x_output);
        x_output = realnvp_block(x_output);
    
    realnvp_model = Model(inputs=[x_input], outputs=x_output);
    mean_true = realnvp_model.predict(z_true)
    lam_true = np.exp(2.2*np.tanh(mean_true));
    return z_true, u_true, mean_true, lam_true

def simulate_cont_data_diff_var(length, n_dim):
    ## simulate 2d z
    np.random.seed(777);
    
    u_true = np.random.uniform(0,2*np.pi,size = [length,1]);
    mu_true = np.hstack((u_true, 2*np.sin(u_true)));
    var_true = 0.15*np.abs(mu_true);
    var_true[:,0] = 0.6-var_true[:,1];
    z_true = np.random.normal(0, 1, size=[length,2])*np.sqrt(var_true)+mu_true;
    z_true = np.hstack((z_true, np.zeros((z_true.shape[0],n_dim-2))));
    
    ## simulate mean
    dim_x = z_true.shape[-1];
    permute_ind = [];
    n_blk = 4;
    for ii in range(n_blk):
        np.random.seed(ii);
        permute_ind.append(tf.convert_to_tensor(np.random.permutation(dim_x)));
    
    x_input = layers.Input(shape=(dim_x,));
    x_output = realnvp_block(x_input);
    for ii in range(n_blk-1):
        x_output = Lambda(perm_func, arguments={'ind':permute_ind[ii]})(x_output);
        x_output = realnvp_block(x_output);
    
    realnvp_model = Model(inputs=[x_input], outputs=x_output);
    mean_true = realnvp_model.predict(z_true)
    lam_true = np.exp(2.2*np.tanh(mean_true));
    return z_true, u_true, mean_true, lam_true

