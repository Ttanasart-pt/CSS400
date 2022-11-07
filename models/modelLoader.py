def loadFromConfig(config):
    if config['model'] == "colorHandPose_handSeg":
        from colorHandPose.handSegNet import HandSegNet as Model
    elif config['model'] == "colorHandPose_postNet":
        from colorHandPose.posenet import PoseNet as Model
    elif config['model'] == "homebrew":
        from homebrew.CNN import Model
    elif config['model'] == "hourglass":
        from hourglass.model import Model
    elif config['model'] == "hourglass_def":
        from hourglass.hrglass import create_hourglass_net
        return create_hourglass_net()
    elif config['model'] == "mobilenet":
        from mobilenet.model import Model
    elif config['model'] == "FPN":
        from handSegment.model import PetModel
        return PetModel("FPN", "resnet34", in_channels=3, out_classes=1)
        
    return Model()