import torch
import torch.nn as nn

class ReLoss(nn.Module):
    def __init__(self, in_features=3, hidden_features=256,
                 out_features=None, act_layer=nn.ELU,
                pretrained=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Linear(hidden_features, 2*hidden_features),
            act_layer(),
            nn.Linear(2*hidden_features, hidden_features),
            act_layer(),
            nn.Linear(hidden_features, out_features),
        )

        # load pretrained weights
        if pretrained:
            ckpt_url = 'loss_module_deltae.ckpt'
            print(f'Load checkpoint of ReLoss from url: {ckpt_url}')
            state_dict = torch.load(ckpt_url)
            self.load_state_dict(state_dict)

    def forward(self, S0, Fuse):
        x = (S0 - Fuse)**2
        x = x.flatten(2).transpose(1, 2)
        x = self.mlp(x)
        loss = x.abs().mean()
        return loss
