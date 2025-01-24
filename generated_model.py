import torch
import torch.nn as nn
import subprocess

class GeneratedModel(nn.Module):
    def __init__(self):
        super(GeneratedModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(3, 16, (3,), stride=(1,), padding=(1,)),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=3),
            nn.Linear(512, 10),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Write the generated PyTorch code to a separate file
    with open('run_model.py', 'w') as f:
        f.write('''import torch\nimport torch.nn as nn\n\nclass GeneratedModel(nn.Module):\n    def __init__(self):\n        super(GeneratedModel, self).__init__()\n        self.model = nn.Sequential(\n            nn.Conv1d(3, 16, (3,), stride=(1,), padding=(1,)),\n            nn.ReLU(),\n            nn.Flatten(start_dim=1, end_dim=3),\n            nn.Linear(512, 10),\n            nn.Linear(10, 1)\n        )\n\n    def forward(self, x):\n        return self.model(x)\n\nif __name__ == "__main__":\n    model = GeneratedModel()\n    print(model)\n    x = torch.randn(1, 3, 32)\n    output = model(x)\n    print("Output:", output)\n''')

    # Run the generated file as a subprocess
    subprocess.run(['python', 'run_model.py']) 