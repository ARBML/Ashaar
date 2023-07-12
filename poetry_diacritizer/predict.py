import os
from typing import Dict

from diacritization_evaluation import der, wer
import torch
from torch import nn
from torch import optim
from torch.cuda.amp import autocast
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.notebook import tqdm
from tqdm import trange
from diacritization_evaluation import util 

from .config_manager import ConfigManager
from .dataset import load_iterators
from .diacritizer import CBHGDiacritizer, Seq2SeqDiacritizer
from .options import OptimizerType
import gdown

class Trainer:
    def run(self):
        raise NotImplementedError


class GeneralTrainer(Trainer):
    def __init__(self, config_path: str, model_kind: str) -> None:
        self.config_path = config_path
        self.model_kind = model_kind
        self.config_manager = ConfigManager(
            config_path=config_path, model_kind=model_kind
        )
        self.config = self.config_manager.config
        self.losses = []
        self.lr = 0
        self.pad_idx = 0
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.set_device()

        self.config_manager.create_remove_dirs()
        self.text_encoder = self.config_manager.text_encoder
        self.start_symbol_id = self.text_encoder.start_symbol_id
        self.summary_manager = SummaryWriter(log_dir=self.config_manager.log_dir)

        self.model = self.config_manager.get_model()

        self.optimizer = self.get_optimizer()
        self.model = self.model.to(self.device)

        self.load_model(model_path=self.config.get("train_resume_model_path"))
        self.load_diacritizer()

        self.initialize_model()


    def set_device(self):
        if self.config.get("device"):
            self.device = self.config["device"]
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_diacritizer(self):
        if self.model_kind in ["cbhg", "baseline"]:
            self.diacritizer = CBHGDiacritizer(self.config_path, self.model_kind)
        elif self.model_kind in ["seq2seq", "tacotron_based"]:
            self.diacritizer = Seq2SeqDiacritizer(self.config_path, self.model_kind)

    def initialize_model(self):
        if self.global_step > 1:
            return
        if self.model_kind == "transformer":
            print("Initializing using xavier_uniform_")
            self.model.apply(initialize_weights)


    def load_model(self, model_path: str = None, load_optimizer: bool = True):
      with open(
          self.config_manager.base_dir / f"{self.model_kind}_network.txt", "w"
      ) as file:
          file.write(str(self.model))

      if model_path is None:
          last_model_path = self.config_manager.get_last_model_path()
          if last_model_path is None:
              self.global_step = 1
              return
      else:
          last_model_path = model_path

      print(f"loading from {last_model_path}")
      saved_model = torch.load(last_model_path, torch.device(self.config.get("device")))
      self.model.load_state_dict(saved_model["model_state_dict"])
      if load_optimizer:
          self.optimizer.load_state_dict(saved_model["optimizer_state_dict"])
      self.global_step = saved_model["global_step"] + 1

class DiacritizationTester(GeneralTrainer):
    def __init__(self, config_path: str, model_kind: str, model_path: str) -> None:
        # if config_path == 'config/test.yml' or config_path == "Arabic_Diacritization/config/test.yml":
        #   print("Exporting the pretrained models ... ")
        #   url = 'https://drive.google.com/uc?id=12aYNY7cbsLNzhdPdC2K3u1sgrb1lpzwO' 
        #   gdown.cached_download(url,'model.zip', quiet=False, postprocess=gdown.extractall)
        
        self.config_path = config_path
        self.model_kind = model_kind
        self.config_manager = ConfigManager(
            config_path=config_path, model_kind=model_kind
        )
        self.config = self.config_manager.config
        # print(self.config)
        self.pad_idx = 0
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.set_device()

        self.text_encoder = self.config_manager.text_encoder
        self.start_symbol_id = self.text_encoder.start_symbol_id

        self.model = self.config_manager.get_model()

        self.model = self.model.to(self.device)
        self.load_model(model_path=model_path, load_optimizer=False)
        self.load_diacritizer()
        self.diacritizer.set_model(self.model)
        self.initialize_model()

    def collate_fn(self, data):
        """
        Padding the input and output sequences
        """

        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        data.sort(key=lambda x: len(x[0]), reverse=True)

        # separate source and target sequences
        src_seqs, trg_seqs, original = zip(*data)

        # merge sequences (from tuple of 1D tensor to 2D tensor)
        src_seqs, src_lengths = merge(src_seqs)
        trg_seqs, trg_lengths = merge(trg_seqs)

        batch = {
            "original": original,
            "src": src_seqs,
            "target": trg_seqs,
            "lengths": torch.LongTensor(src_lengths),  # src_lengths = trg_lengths
        }
        return batch
    
    def get_batch(self, sentence):
      data = self.text_encoder.clean(sentence)
      text, inputs, diacritics = util.extract_haraqat(data)
      inputs = torch.Tensor(self.text_encoder.input_to_sequence("".join(inputs)))
      diacritics = torch.Tensor(self.text_encoder.target_to_sequence(diacritics))
      batch = self.collate_fn([(inputs, diacritics, text)])
      return batch
  
    def infer(self, sentence):
        self.model.eval()
        batch = self.get_batch(sentence)
        predicted = self.diacritizer.diacritize_batch(batch)
        return predicted[0]
