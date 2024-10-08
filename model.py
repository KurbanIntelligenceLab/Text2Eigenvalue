import torch
import torch.nn as nn

class Text2Everything(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super(HamiltonSolverTriple, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim // 4 + 3, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 8, hidden_dim // 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim // 16 + 3, output_dim),
        )

    def forward(self, x, a, b, c):
        x = self.layer1(x)
        x = torch.concat((x, a, b, c), dim=1)
        x = self.layer2(x)
        x = torch.concat((x, a, b, c), dim=1)
        x = self.layer3(x)

        return x

# Example usage:
# model = Text2Energy(input_dim=128, hidden_dim=256, output_dim=1, dropout=0.3)
# print(model)
