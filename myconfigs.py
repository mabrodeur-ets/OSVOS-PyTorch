
class Configs():
    # Name of the model to start the fine-tuning from ... default: 'parent'
    parentModelName = 'parent'

    # Epoch of the model to start the fine-tuning from ... default: 240
    parentEpoch = 240

    # Name of the sequence of images (video) ... default: 'blackswan'
    sequence_name = 'blackswan'

    # Create a checkpoint every <nth> epoch
    checkpoint_every = 100

    # Visualize the network [0, 1] ... default: 0              
    vis_network = 0

    # Visualize the results [0, 1] ... default: 0
    vis_results = 0

    # Average the gradient every <nth> iterations
    avg_gradient_every = 5    

    # Train train the model/network for a total of <number> epochs              
    nb_epochs = 2000 * avg_gradient_every

    # Parameters in p are used for the name of the model
    p = {
        'trainBatch': 1,  # Number of Images in each mini-batch
        }

    # Used For Testing Phase
    seed = 0                                
