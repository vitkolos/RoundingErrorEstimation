import tqdm
import torch

from network import SmallConvNet
from sparse_utils import left_padding, right_padding, SparseLinear

class MaxNetwork(torch.nn.Module):

    def __init__(self, shape, mask):
        super().__init__()

        
        #        self.mask = torch.zeros(shape)
        #        for i in mask:
        #            self.mask[i] = 1
        x, y = mask
        self.x, self.y = x, y
        assert len(shape) == 1
        self.n_inputs = shape[0] 

        #        print("n_inputs", self.n_inputs)
        
        self.layer1 = torch.nn.Linear(self.n_inputs, 2, bias=False)
        #print(self.layer1.weight.data.shape)
        self.layer1.weight.data = torch.zeros(self.layer1.weight.data.shape).double()
        self.layer1.weight.data[0, x] = 1.0
        self.layer1.weight.data[0, y] = -1.0
        self.layer1.weight.data[1, y] = 1.0  # is this correct?
        # max(x,y) = ReLU(x-y)+y
        # but this would equal ReLU(x-y)+ReLU(y), right?
        
        #print(self.layer1.bias.data.shape)
        #self.layer1.bias.data = torch.zeros(self.layer1.bias.data.shape).double()

        self.layer2 = torch.nn.ReLU()

        self.layer3 = torch.nn.Linear(2, 1, bias=False)
        #print(self.layer3.weight.data.shape)        
        #print(self.layer3.bias.data.shape)
        self.layer3.weight.data = torch.ones(self.layer3.weight.data.shape).double()
        #self.layer3.bias.data = torch.zeros(self.layer3.bias.data.shape).double()

        self.layer4 = torch.nn.ReLU()

        
    def forward(self, x): 
        output = self.layer1(x)      # Linear
        output = self.layer2(output) # ReLU
        output = self.layer3(output) # Linear
        output = self.layer4(output) # ReLU
        return output
        
def stack_max_networks(list_of_networks):


    n = len(list_of_networks)
    net = list_of_networks[0]
    n_inputs = net.layer1.weight.data.shape[1]
    layer1 = torch.nn.Linear(n_inputs, 2*n, bias=False).double()#.cuda()
    layer2 = torch.nn.ReLU()#.cuda()
    layer3 = torch.nn.Linear(2*n, n, bias=False).double()#.cuda()
    layer4 = torch.nn.ReLU()#.cuda()

    layer1.weight.data = torch.vstack([
        net.layer1.weight.data
        for net in list_of_networks
        ])
    #layer1.bias.data = torch.zeros(layer1.bias.data.shape).double()

    layer3.weight.data = torch.vstack([
        torch.hstack([
            torch.zeros((1, 2), dtype=torch.float64)#.cuda()
            if i != j else net.layer3.weight.data
            for j in range(n)
        ])
        for i, net in enumerate(list_of_networks)
    ])
    #layer3.bias.data = torch.zeros(layer3.bias.data.shape).double()

    return torch.nn.Sequential(layer1, layer2, layer3, layer4)#.cuda()
    
class MyMaxPool(torch.nn.Module):

    def create_maximum_layer(self, mask):
        #print("mask.shape", mask.shape)

        indexes_of_ones = [] 
        for i, bit in enumerate(mask):
            if bit:
                indexes_of_ones.append(i)

        #        print(indexes_of_ones)
        assert len(indexes_of_ones) % 2 == 0
        
        num = len(indexes_of_ones) // 2
        new_masks = []
        for i in range(num):
            new_mask = (indexes_of_ones[i*2], indexes_of_ones[i*2+1])
            new_masks.append(new_mask)

            
        networks = []
        for m in new_masks:
            networks.append(MaxNetwork(mask.shape, m))
        
        net = stack_max_networks(networks)
        return net

    def vstack_networks(self, net1, net2):

        layers = []
        for l in net1.children():
            layers.append(l)
        for l in net2.children():
            layers.append(l)
        return torch.nn.Sequential(*layers)

        

    def create_maximum(self, mask):
        net = self.create_maximum_layer(mask)
        n_outputs = net[-2].weight.data.shape[0] # -1 is ReLU, -2 is last linear 

        while n_outputs > 1:
            mask2 = torch.ones((n_outputs,))
            top_net = self.create_maximum_layer(mask2)
            net = self.vstack_networks(net, top_net)
            n_outputs = net[-2].weight.data.shape[0]

        return net
        
    def __init__(self, layer, input_shape, output_shape):
        super().__init__()

        self.layer = layer 
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride

        
        assert len(output_shape) == 3
        n_outputs = output_shape[0]*output_shape[1]*output_shape[2]
        self.n_outputs = n_outputs
        
        masks = []
        for i in range(n_outputs):
            mask = torch.zeros(self.input_shape, dtype=torch.bool)
            plane = i // (self.output_shape[1] * self.output_shape[2])
            x = (i % (self.output_shape[1] * self.output_shape[2])) // self.output_shape[1]
            y = (i % (self.output_shape[1] * self.output_shape[2])) % self.output_shape[1]

            for xx in range(self.kernel_size):
                for yy in range(self.kernel_size):
                    mask[plane, x*self.stride+xx, y*self.stride+yy] = True
            masks.append(mask)
            
        self.masks = masks

        max_nets = []
        i = 0
        for mask in tqdm.tqdm(self.masks):
            max_nets.append(self.create_maximum(mask.flatten()).cpu())
            i += 1
            #if i > 10:
            #    break

#        print("concatenating max networks")
        self.max_nets = max_nets
        #self.net = self.hstack_networks(max_nets)

        
    def forward(self, x):
        print("evaluating resulting max network")
        
        out = [max_(x.flatten()) for max_ in self.max_nets]
        
        return torch.tensor(out).reshape(self.output_shape)

class ListOfNetworks(torch.nn.Module):
    
    def __init__(self, network_list, output_shape):
        super().__init__()

        self.network_list = [x.cuda() for x in network_list]
        self.output_shape = output_shape

    def forward(self, x):
        out = [net(x.flatten()) for net in self.network_list]        
        return torch.tensor(out).reshape((-1, *self.output_shape))
        
        
    

def max_pooling_to_dense(layer, input_shape, output_shape):

    my = MyMaxPool(layer, input_shape, output_shape)
    #    return my.max_net
    return ListOfNetworks(my.max_nets, output_shape)
    
def hstack_networks(netlist):

    def padding(w, left, right):
        w = w.to_sparse()
        w = left_padding(w, left)
        w = right_padding(w, right)
        return w
    
    layers = []

    for child in netlist[0].children():
        if isinstance(child, torch.nn.Linear):
            layers.append(
                [child.weight.data]
            )
        elif isinstance(child, torch.nn.ReLU):
            layers.append(
                None
            )
        else:
            raise NotImplementedError

        
    for net in netlist[1:]:
        for i, child in enumerate(net.children()):
            if isinstance(child, torch.nn.Linear):
                layers[i].append(child.weight.data)
            elif isinstance(child, torch.nn.ReLU):
                assert layers[i] is None
            else:
                raise NotImplementedError

    final_layers = []
    first = True
    for i in range(len(layers)):
        print("layer", i)
        if layers[i] is None:
            final_layers.append(torch.nn.ReLU())
        else:
            list_of_layers = layers[i]

            if not first:
                pbar = tqdm.tqdm(total = len(list_of_layers))
                w_sizes = []
                for l in list_of_layers:
                    w_sizes.append(l.shape[1])

                paddings = []
                for i, size in enumerate(w_sizes):
                    left = sum(w_sizes[:i])
                    right = sum(w_sizes[i+1:])
                    paddings.append((left, right))
                    
                for i in range(len(list_of_layers)):
                    left, right = paddings[i]
                    list_of_layers[i] = padding(list_of_layers[i], left, right)
                    pbar.update(1)
                pbar.close()
            lin_layer = SparseLinear(
                torch.vstack(list_of_layers), # w
                None # bias
            )
            final_layers.append(lin_layer)
            first = False
            
    return torch.nn.Sequential(*final_layers)



if __name__ == "__main__":

    print("loading max layer")
    network = torch.load("tmp/max_layer.pt")
    print("done")

    input_ = torch.rand((64, 28, 28)).double().abs()

    print("evaluating max layer")
    output1 = network(input_)
    print("done")
    
    network_list = network.network_list 
    network2 = hstack_networks(network_list)

    output2 = network2(input_).cpu()

    print(torch.all(output1.flatten() == output2.flatten()))

    print("saving")
    torch.save(network2, "tmp/max_sparse_layer.pt")
    print("done")
    
    exit() 
    network_path = "mnist_conv_net.pt"
    
    net = SmallConvNet()
    net.load_state_dict(torch.load(network_path))
    net.eval()
    net.double()

    print(net)
    seq = next(net.children())
    
    print(seq)

    mp = seq[4]

    new_mp = max_pooling_to_dense(mp,
                                  input_shape=(64, 28, 28),
                                  output_shape=(64, 14, 14)
                                  )

    input_shape = (1, 1, 28, 28)
    input = torch.rand(input_shape).double()

    print(input.shape)
    output = input 
    for layer in seq[:4]:
        output = layer(output)

    print("Before maxpool")
    print(output.shape)


    print("After maxpool")
    print((seq[4](output)).shape)

    print("--------------------")

    input = torch.rand((64, 28, 28)).double().abs()
    
    out1 = mp(input)
    print(out1.flatten()[:10])

    out2 = new_mp(input)

    print(out1.shape)
    print(out2.shape)
    
    print(out1.flatten()[:10])
    print(out2.flatten()[:10])
    
    print(torch.all(out1.isclose(out2).flatten()))
    
    torch.save(new_mp, "tmp/max_layer.pt")
