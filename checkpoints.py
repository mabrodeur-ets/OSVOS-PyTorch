import torch
import shutil

def save_checkpoint(state, is_best, path_checkpoint, path_best_model):
    """
    Parameters
    ----------
    state : Object 
        Contains the [model_state_dict, optimizer_state_dict, valid_loss_min, epoch] to save as checkpoint
    is_best : bool
        If this is the best model, it will be saved at path_best_model
    path_checkpoint : str
        The path where to save the checkpoint
    path_best_model : str
        The path where to save the best model, if applicable (i.e. is_best is True)
    """
    f_path = path_checkpoint
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)

    # if it is a best model, min validation loss
    if is_best:
        best_fpath = path_best_model
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_checkpoint(path_checkpoint, model, optimizer):
    """
    Parameters
    ----------
    path_to_checkpoint : str 
        The path to the saved checkpoint
    model : torch.nn.modules
        The model which state must be restored using checkpoint to continue training 
    optimizer : torch.optim
        The optimizer which state must be restored using checkpoint to continue training 

    Returns
    ----------
    model, optimizer, epoch, valid_loss_min.item()
    """
    # load check point
    checkpoint = torch.load(path_checkpoint)
    
    # load checkpoint.model into model
    model.load_state_dict(checkpoint['model_state_dict'])
   
    # load checkpoint.optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # load checkpoint.valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']

    # load epoch to start from (previous + 1)
    epoch = checkpoint['epoch']

    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, epoch, valid_loss_min.item()