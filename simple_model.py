import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            # nn.Flatten()
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    torch.manual_seed(14)
    x = torch.rand(1, 2)
    print(x)
    model = Model()
    y = model(x)
    print(y)

    # save the model 
    torch.save(model.state_dict(), 'very_simple_model.pth')