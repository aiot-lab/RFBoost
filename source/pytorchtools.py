import numpy as np
import torch
import time
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, best_path='checkpoint.pt', best_acc_path='', end_model_path='', trace_func=print, es_enabled=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            best_path (str): Path for the Best model to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: ear            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_path = best_path
        self.best_acc_path = best_acc_path
        self.best_acc = 0
        self.best_f1 = 0
        self.trace_func = trace_func
        self.es_enabled = es_enabled

        self.best_acc = None
        self.best_f1 = None

        self.best_ep_loss = None
        self.best_ep_acc = None
        # start timer
        self.time_start = time.time()
        
        self.end_model_path = end_model_path
        
    def __call__(self, val_loss, val_acc, val_f1, ep, model):

        score = -val_loss

        if self.best_acc is None:
            self.best_acc = val_acc
            self.best_f1 = val_f1
            self.best_ep_acc = ep
            self.save_checkpoint(None, model, self.best_acc_path) 
        elif self.best_acc < val_acc:
            self.best_acc = val_acc
            self.best_f1 = val_f1 
            self.best_ep_acc = ep  
            self.save_checkpoint(None, model, self.best_acc_path)         

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.best_path)
            self.best_ep_loss = ep
        elif score < self.best_score + self.delta:
            if self.es_enabled:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.trace_func("No update on validset. But Early stopping is disabled")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.best_path)
            self.best_ep_loss = ep
            self.counter = 0
            
        # self.save_checkpoint(, model, self.end_model_path)



    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if val_loss is not None:
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss
        else:
            self.trace_func(f'Better Acc. Saving model ...')
        torch.save(model.state_dict(), path)
        