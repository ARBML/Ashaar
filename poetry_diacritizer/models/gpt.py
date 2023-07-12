from typing import List
from torch import nn
import torch
from pathlib import Path
import json
from .gpt_model import Model, HParams


class GPTModel(nn.Module):
    def __init__(self, path, n_layer=-1, freeze=True, use_lstm=False):
        super().__init__()
        root = Path(path)

        params = json.loads((root / "params.json").read_text())
        hparams = params["hparams"]
        hparams.setdefault("n_hidden", hparams["n_embed"])
        self.model = Model(HParams(**hparams))
        state = torch.load(root / "model.pt", map_location="cpu")
        state_dict = self.fixed_state_dict(state["state_dict"])
        self.model.load_state_dict(state_dict)
        self.activation = {}
        self.freeze = freeze
        self.n_layer = n_layer
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.activation = {}
        self.use_lstm = use_lstm
        self.set_hook(self.n_layer)
        self.in_fc_layer = 512 if self.use_lstm else 768
        self.lstm1 = nn.LSTM(
            768,
            256,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            512,
            256,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm3 = nn.LSTM(
            512,
            256,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(self.in_fc_layer, 17)

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output[0].detach()

        return hook

    def set_hook(self, n_layer=0):
        self.model.blocks[n_layer].register_forward_hook(self.get_activation("feats"))

    def fixed_state_dict(self, state_dict):
        if all(k.startswith("module.") for k in state_dict):
            # legacy multi-GPU format
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        return state_dict

    def forward(self, src: torch.Tensor, lengths: torch.Tensor, target=None):

        # logits shape [batch_size, 256, 500]
        logits = self.model(src)["logits"]
        logits = self.activation["feats"]

        if self.use_lstm:
            x, (h, cn) = self.lstm1(logits)
            x, (h, cn) = self.lstm2(x)
            x, (h, cn) = self.lstm3(x)
        else:
            x = logits
        predictions = self.fc(x)

        output = {"diacritics": predictions}

        return output
