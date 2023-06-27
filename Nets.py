import torch.nn as nn

class Risk_Net(nn.Module):
    def __init__(self, nf, hidden_layers,device, non_linearity='relu'):
        super().__init__()
        layers = []
        if non_linearity == "relu":
            activation = nn.ELU()
        for h in hidden_layers:
            #layers.append(nn.Dropout(p=0.3))
            layers.append(nn.Linear(nf, h))
            layers.append(activation)
            nf = h
        layers.pop()
        self.net = nn.Sequential(*layers).to(device)
    
    def forward(self, x):
        return self.net(x).reshape(-1,)
