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
    torch.manual_seed(42)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = applications.mnist.SmallDenseNet().to(device)
    data_split = applications.mnist.MnistSplit()
    MODEL_FILE = "models/small_dense.pth"
    
    if False:
        model.fit(data_split.train, data_split.dev)
        model.save(MODEL_FILE)
    else:
        model.load(MODEL_FILE)
        loader_dev = torch.utils.data.DataLoader(data_split.dev, batch_size=64)
        print(model.evaluate(loader_dev))

        model_approx = applications.mnist.SmallDenseNet().to(device)
        model_approx.load(MODEL_FILE)
        model_approx.round(bits=16)
        print(model_approx.evaluate(loader_dev))

        max_err, avg_err = network_train.TrainableModel.compute_error(model, model_approx, loader_dev)
        print(max_err, avg_err)


if __name__ == '__main__':
    main()
