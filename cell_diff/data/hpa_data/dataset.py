# -*- coding: utf-8 -*-
import os
from typing import List

import numpy as np
import torch

from torch.utils.data import Dataset

from .collater import collate_fn
from cell_diff.data.hpa_data.vocabulary import Alphabet, convert_string_sequence_to_int_index
from cell_diff.data.hpa_data.sequence_masking import OAARDM_sequence_masking
from cell_diff.data.hpa_data.img_utils import RandomRotation

from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision import transforms

import json
import pickle
import io


class HPADataset(Dataset):
    def __init__(self, args, split_key, vae) -> None:
        super().__init__()
        self.args = args

        self.split_key = split_key
        self.data_path = self.args.data_path
        self.vocab = Alphabet()

        data_files = []
        for ensg in os.listdir(os.path.join(self.data_path, split_key)):
            if ensg.startswith('ENSG'):
                subdir = os.path.join(self.data_path, split_key, ensg)
                for filename in os.listdir(subdir):
                    if filename.endswith('png'):
                        data_files.append((os.path.join(subdir, filename), ensg))
        self.data_files = data_files

        self.img_crop_method = self.args.img_crop_method
        self.img_resize = self.args.img_resize
        self.img_crop_size = self.args.img_crop_size

        protein_sequences = {}
        this_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(this_dir, "protein_sequence.txt")) as f:
            for line in f.readlines():
                if len(line) > 0:
                    ensg, _, sequence = line[1:].strip().split("|")
                    protein_sequences[ensg] = sequence
        self.protein_sequences = protein_sequences
        self.vae = vae

    def __getitem__(self, index: int) -> dict:
        file_path, ensg_id = self.data_files[index]

        item = {}
        item['protein_img'], item['nucleus_img'], item['seg_img'] = self.get_img(file_path)

        protein_seq = self.protein_sequences[ensg_id]
        protein_seq_masked, protein_seq_mask, zm_label = OAARDM_sequence_masking(protein_seq, self.args.seq_zero_mask_ratio)

        """
        - convert string sequence to int index
        """
        protein_seq_token = convert_string_sequence_to_int_index(self.vocab, protein_seq)
        protein_seq_masked_token = convert_string_sequence_to_int_index(self.vocab, protein_seq_masked)

        if self.vocab.prepend_bos:
            protein_seq_mask = np.insert(protein_seq_mask, 0, False)
        if self.vocab.append_eos:
            protein_seq_mask = np.append(protein_seq_mask, False)

        item['zm_label'] = torch.Tensor([zm_label]).bool()
        item['protein_seq'] = torch.LongTensor(protein_seq_token)
        item['protein_seq_masked'] = torch.LongTensor(protein_seq_masked_token)
        item['protein_seq_mask'] = torch.from_numpy(protein_seq_mask).long()
        item['prot_id'] = ensg_id
        return item

    def get_img(self, file_path):
        image = Image.open(file_path)
        if image.width < self.img_crop_size or image.height < self.img_crop_size:
            scale = max(self.img_crop_size / image.width, self.img_crop_size / image.height)
            new_w = int(round(image.width * scale))
            new_h = int(round(image.height * scale))
            image = image.resize((new_w, new_h), Image.Resampling.NEAREST)
        image = to_tensor(image)
        # RGBA, R: microtubules, G: protein, B: nucleus, A: segmentation
        protein = image[1].unsqueeze(0)
        nucleus = image[2].unsqueeze(0)
        segmentation = image[3].unsqueeze(0)
        nucleus = nucleus * segmentation
        protein = protein * segmentation

        # To maintain the resolution.
        # First crop then resize.
        t_forms = []

        if self.img_crop_method == 'random':
            t_forms.append(transforms.RandomCrop(self.img_crop_size))
            t_forms.append(transforms.RandomHorizontalFlip(p=0.5))
            t_forms.append(RandomRotation([0, 90, 180, 270]))

        elif self.img_crop_method == 'center':
            t_forms.append(transforms.CenterCrop(self.img_crop_size))

        t_forms.append(transforms.Resize(self.img_resize, antialias=None))

        t_forms = transforms.Compose(t_forms)
        norm_fn = transforms.Normalize(mean=[0.5], std=[0.5])

        image = torch.stack([protein, nucleus, segmentation], dim=0)
        protein, nucleus, segmentation = t_forms(image)
        protein = norm_fn(protein)
        return protein, nucleus, segmentation

    def __len__(self) -> int:
        return len(self.data_files)

    def collate(self, samples: List[dict]) -> dict:
        return collate_fn(samples, self.vocab, self.args.max_protein_sequence_len, 0, self.vae)
