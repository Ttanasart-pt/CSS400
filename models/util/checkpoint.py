import torch
import os.path

def save_checkpoint(filename, model):
    checkpoint = { "state_dict": model.state_dict() }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model):
    if(not os.path.isfile(filename)):
        return
    
    print("Loading " + filename)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    print("Checkpoint loaded")
