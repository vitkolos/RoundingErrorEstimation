import torch
import dataset_prepare
import network_train

def main():
    """
    1) prepare a dataset
    2) train a network
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
    model = network_train.FCNetwork().to(device)
    data_split = dataset_prepare.get_mnist_split()
    model.fit(data_split.train, data_split.dev)
    torch.save(model.state_dict(), "model.pth")
    
    # model = network_train.FCNetwork().to(device)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))



if __name__ == '__main__':
    main()
