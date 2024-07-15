import numpy as np
import matplotlib.pyplot as plt
import glob
import os

CLASS_NUM_FM = 10
CLASS_NUM_M = 10
NUM_TARGET_IMG = 2222
NUM_FILLER_IMG = 8042


# Class for doing sparse coding
class OlshausenFieldModel:
    def __init__(self, num_inputs, num_units, batch_size, Phi=None,
                 lr_r=1e-2, lr_Phi=1e-2, lmda=1e-3):
        self.lr_r = lr_r # learning rate of r
        self.lr_Phi = lr_Phi # learning rate of Phi
        self.lmda = lmda # regularization parameter
        
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.batch_size = batch_size
        
        # Weights
        if Phi is None:
            Phi = np.random.randn(self.num_inputs, self.num_units).astype(np.float32)
            self.Phi = Phi * np.sqrt(1/self.num_units)
        else:
            self.Phi = Phi

        # activity of neurons
        self.r = np.zeros((self.batch_size, self.num_units))
    
    def initialize_states(self):
        self.r = np.zeros((self.batch_size, self.num_units))
        
    def normalize_rows(self):
        self.Phi = self.Phi / np.maximum(np.linalg.norm(self.Phi, 
            ord=2, axis=0, keepdims=True), 1e-8)

    # thresholding function of S(x)=|x|
    def soft_thresholding_func(self, x, lmda):
        return np.maximum(x - lmda, 0) - np.maximum(-x - lmda, 0)

    def calculate_total_error(self, error, include_sparsity=True):
        #recon_error = np.mean(error**2)
        recon_error = np.mean(error**2, axis=1)
        sparsity_r = 0
        if include_sparsity:
            sparsity_r = self.lmda*np.mean(np.abs(self.r)) 
        return recon_error + sparsity_r
        
    def __call__(self, inputs, training=True):
        # Updates                
        error = inputs - self.r @ self.Phi.T
        
        r = self.r + self.lr_r * error @ self.Phi
        self.r = self.soft_thresholding_func(r, self.lmda)

        if training:  
            error = inputs - self.r @ self.Phi.T
            dPhi = error.T @ self.r
            self.Phi += self.lr_Phi * dPhi
            
        return error, self.r

    def get_prediction(self):
        return self.r @ self.Phi.T

    def get_Phi(self):
        return self.Phi

    def get_r(self):
        return self.r

    def clone(self, batch_size):
        cloned_model = OlshausenFieldModel(self.num_inputs, self.num_units, 
            batch_size, self.Phi, self.lr_r, self.lr_Phi, self.lmda)
        return cloned_model

    def clone_with_different_lambda(self, batch_size, lmda):
        cloned_model = OlshausenFieldModel(self.num_inputs, self.num_units, 
            batch_size, self.Phi, self.lr_r, self.lr_Phi, lmda)
        return cloned_model


# Helper function to train or evaluate dictionary
def run_model(model, inputs, mode='train', num_iter=1000, nt_max=1000, 
    batch_size=250, eps=1e-2, plot=False, verbose=False, save_path='default'):
    if mode == 'train':
        if verbose:
            print('Start training')
    else:
        assert(mode == 'validation' or mode == 'test')
        if verbose:
            print('Start evaluating on', mode, 'data')

    assert(len(inputs.shape) == 2)

    error_list = []
    batch_error_list = []
    num_samples = len(inputs)
    if mode != 'train':
        prediction = np.zeros((num_iter*batch_size, inputs.shape[1]))
    for iter_ in range(num_iter):
        #print(model.r[:5,])
        #print('--------------')
        if mode == 'train':
            index = np.random.choice(num_samples, batch_size)
            batch_inputs = inputs[index] - np.mean(inputs[index])
        else:
            batch_inputs = inputs[iter_*batch_size:(iter_+1)*batch_size] - \
            np.mean(inputs[iter_*batch_size:(iter_+1)*batch_size])
        
        model.initialize_states() # reset states
        if mode == 'train':
            model.normalize_rows() # normalize weights
    
        # Input a new batch until latent variables are converged 
        r_tm1 = model.r # set previous r (t minus 1)

        for t in range(nt_max):
            # Update r without update weights 
            error, r = model(batch_inputs, training=False)
            dr = r - r_tm1 

            # Compute norm of r
            dr_norm = np.linalg.norm(dr, ord=2) / (eps + np.linalg.norm(r_tm1, ord=2))
            r_tm1 = r # update r_tm1
        
            # Check convergence of r, then update weights
            if dr_norm < eps:
                if mode == 'train':
                    error, r = model(batch_inputs, training=True)
                break
        
            # If failure to convergence, break and print error
            if t >= nt_max-2: 
                print("Error at patch:", iter_)
                print(dr_norm)
                break
        errors = model.calculate_total_error(error, include_sparsity=False)
        error_list.extend(errors) # Append errors
        batch_error_list.append(np.mean(errors))

        # Print moving average error or save prediction
        if mode == 'train':
            if iter_ % 100 == 99 and verbose:  
                print("Iter: {}/{}, moving error:{:.3g}".format(iter_+1, num_iter, 
                    np.mean(batch_error_list[iter_-99:iter_])))
        else:
            prediction[iter_*batch_size:(iter_+1)*batch_size,:] = model.get_prediction()
    if verbose:
        print('Average reconstruction error on {} is {:.3g}'.format(mode, 
            np.mean(batch_error_list)))
    if mode == 'train' and plot:
        fig = plt.figure(figsize=(10, 6))
        plt.ylabel("Error")
        plt.xlabel("Iterations")
        plt.title('Sparse Coding Train Error Plot')
        plt.plot(np.arange(len(batch_error_list)), np.array(batch_error_list))
        plt.tight_layout()
        plt.show()
        fig.savefig(save_path, bbox_inches='tight')
    if mode == 'train':
        return error_list, batch_error_list,
    else:
        return error_list, batch_error_list, prediction

def compute_pairwise_spc_distance(layer, dir_list, target_img_list, codeword_dim,
    num_units, num_iter=1000, nt_max=1000, batch_size=250, eps=1e-2, lr_r=1e-2, 
    lr_Phi=1e-2, lmda=1e-3, prefix='', plot_spc_train=True, include_filler=False,
    train_with_filler_only=True):
    folder_list = ['Targets']
    if include_filler:
        folder_list = ['Targets', 'Fillers']
    act_list = []
    # need to make sure directory is first targets then fillers
    example_img = np.load(os.path.join(dir_list[0], '1.npy'))

    # Sample a subset of units
    indices_sampled = np.random.choice(example_img.reshape((1,-1)).shape[1], 
        codeword_dim, replace=False)
    # Load targets
    files = glob.glob(os.path.join(dir_list[0], '*.npy'))

    for directory in dir_list:
        files = glob.glob(os.path.join(directory,'*.npy'))
        for i in range(1, len(files)+1):
            tmp = np.load(os.path.join(directory, str(i)+'.npy')) # e.g. (1, 112, 112, 64)
            tmp_sampled = tmp.reshape((1,-1))[:,indices_sampled]
            act_list.append(tmp_sampled)

    act_flat = np.asarray(act_list).squeeze(axis=1)
    act_flat = act_flat / np.quantile(np.abs(act_flat), 0.95)
    num_img = act_flat.shape[0]
    train_act_flat = act_flat
    if train_with_filler_only:
        train_act_flat = act_flat[NUM_TARGET_IMG:]
    
    model_train = OlshausenFieldModel(num_inputs=codeword_dim, 
        num_units=num_units, batch_size=batch_size, Phi=None, lr_r=lr_r, 
        lr_Phi=lr_Phi, lmda=lmda)

    fig_save_dir = os.path.join('figures', 'mem', '_'.join(folder_list), prefix)
    if not os.path.isdir(fig_save_dir):
        os.makedirs(fig_save_dir, exist_ok=True)

    error_train, batch_error_train = run_model(model_train, train_act_flat, 'train', 
        num_iter, nt_max, batch_size, eps, plot=plot_spc_train, verbose=False, 
        save_path=os.path.join(fig_save_dir,layer))

    model_test = model_train.clone_with_different_lambda(batch_size=num_img,
        lmda=[lmda])
    error_list, batch_error_list, pred_test = run_model(model_test, act_flat, 'test', 1, nt_max,
        num_img, eps, verbose=False)

    distance_rec_quality = np.zeros((len(target_img_list), 1))
    for i, target_img in enumerate(target_img_list):
        distance_rec_quality[i,:] = error_list[target_img-1]

    return distance_rec_quality





