import torch
import network_train
import applications.mnist

def main():
    """
    1) prepare a dataset
    2) train (or provide) a network
    3) generate (or provide) an approximated network
    ---
    4) simplify networks (to ReLU & linear layers + normalize)
    5) combine them into an evaluation network
    6) perform (parallel) linear optimimization to find maxima in polytopes
    7) report results
    
    AppMax
    input: original network, approximated network, data samples
    output: reported errors (single sample × polytope; maximum × average)
    """
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    args = network_train.TrainingArgs(device, 64, 10)
    model = applications.mnist.SmallDenseNet(args)
    data_split = applications.mnist.MnistSplit()
    MODEL_FILE = "models/small_dense.pth"
    
    if False:
        model.fit(data_split.train, data_split.dev)
        model.save(MODEL_FILE)
    else:
        model.load(MODEL_FILE)
        loader_dev = torch.utils.data.DataLoader(data_split.dev, batch_size=64)
        print(model.evaluate(loader_dev))



if __name__ == '__main__':
    main()
