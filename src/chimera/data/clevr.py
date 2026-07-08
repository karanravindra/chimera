"""
CLEVR visual question answering DataModule for PyTorch Lightning.

The official CLEVR v1.0 archive is large (~18GB). By default this module expects
the extracted dataset at ``data_dir/CLEVR_v1.0`` and raises a clear error if it
is missing. Pass ``download=True`` to download and extract the official archive.

Expected layout:
    CLEVR_v1.0/
      images/{train,val,test}/*.png
      questions/CLEVR_{train,val,test}_questions.json

Usage:
    dm = CLEVRVQADataModule(data_dir="./data", batch_size=64)
    trainer.fit(model, datamodule=dm)
"""

import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLEVR_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize_question(text: str) -> list[str]:
    """Simple deterministic tokenizer for CLEVR questions."""
    return re.findall(r"[a-z0-9]+", text.lower())


class CLEVRQuestionDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        question_vocab: dict[str, int],
        answer_vocab: dict[str, int] | None,
        transform,
        max_question_len: int,
    ):
        self.root = root
        self.split = split
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab
        self.transform = transform
        self.max_question_len = max_question_len

        question_path = root / "questions" / f"CLEVR_{split}_questions.json"
        with question_path.open() as f:
            payload = json.load(f)
        self.questions = payload["questions"]

    def __len__(self) -> int:
        return len(self.questions)

    def _encode_question(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokenize_question(text)
        if not tokens:
            tokens = [UNK_TOKEN]
        ids = [
            self.question_vocab.get(token, self.question_vocab[UNK_TOKEN])
            for token in tokens[: self.max_question_len]
        ]
        length = len(ids)
        ids += [self.question_vocab[PAD_TOKEN]] * (self.max_question_len - length)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(length, dtype=torch.long)

    def __getitem__(self, idx: int):
        row = self.questions[idx]
        image_path = self.root / "images" / self.split / row["image_filename"]
        image = Image.open(image_path).convert("RGB")

        question, question_len = self._encode_question(row["question"])
        item = {
            "image": self.transform(image),
            "question": question,
            "question_len": question_len,
            "question_text": row["question"],
            "image_filename": row["image_filename"],
        }

        if "answer" in row:
            if self.answer_vocab is None:
                raise ValueError("answer_vocab is required for labeled CLEVR splits")
            answer = row["answer"]
            if answer not in self.answer_vocab:
                raise KeyError(f"answer {answer!r} is missing from the answer vocab")
            item["answer"] = torch.tensor(self.answer_vocab[answer], dtype=torch.long)
            item["answer_text"] = answer

        return item


class CLEVRVQADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        image_size: int = 128,
        max_question_len: int = 48,
        num_workers: int = 4,
        pin_memory: bool = True,
        download: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_question_len = max_question_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.question_vocab: dict[str, int] = {}
        self.answer_vocab: dict[str, int] = {}
        self.answer_names: list[str] = []

        self.clevr_train: Optional[Dataset] = None
        self.clevr_val: Optional[Dataset] = None
        self.clevr_test: Optional[Dataset] = None

    @property
    def root(self) -> Path:
        return self.data_dir / "CLEVR_v1.0"

    @property
    def vocab_size(self) -> int:
        return len(self.question_vocab)

    @property
    def num_answers(self) -> int:
        return len(self.answer_vocab)

    def prepare_data(self):
        if self.root.exists():
            return
        if not self.download:
            raise FileNotFoundError(
                f"CLEVR was not found at {self.root}. Download and extract the "
                f"official dataset from {CLEVR_URL}, or pass download=True."
            )

        self.data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_dir / "CLEVR_v1.0.zip"
        if not zip_path.exists():
            print(f"Downloading CLEVR from {CLEVR_URL} ...")
            req = urllib.request.Request(CLEVR_URL, headers={"User-Agent": "chimera"})
            with urllib.request.urlopen(req) as resp, zip_path.open("wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)

        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(self.data_dir)

    def _load_questions(self, split: str) -> list[dict]:
        question_path = self.root / "questions" / f"CLEVR_{split}_questions.json"
        with question_path.open() as f:
            return json.load(f)["questions"]

    def _ensure_vocabs(self):
        if self.question_vocab and self.answer_vocab:
            return

        train_questions = self._load_questions("train")
        tokens = sorted(
            {
                token
                for row in train_questions
                for token in tokenize_question(row["question"])
            }
        )
        self.question_vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.question_vocab.update(
            {token: idx for idx, token in enumerate(tokens, start=2)}
        )

        self.answer_names = sorted({row["answer"] for row in train_questions})
        self.answer_vocab = {
            answer: idx for idx, answer in enumerate(self.answer_names)
        }

    def _dataset(self, split: str, labeled: bool):
        return CLEVRQuestionDataset(
            root=self.root,
            split=split,
            question_vocab=self.question_vocab,
            answer_vocab=self.answer_vocab if labeled else None,
            transform=self.transform,
            max_question_len=self.max_question_len,
        )

    def setup(self, stage: Optional[str] = None):
        self._ensure_vocabs()

        if stage == "fit" or stage is None:
            self.clevr_train = self._dataset("train", labeled=True)
            self.clevr_val = self._dataset("val", labeled=True)

        if stage == "test" or stage is None:
            self.clevr_val = self._dataset("val", labeled=True)

        if stage == "predict" or stage is None:
            self.clevr_test = self._dataset("test", labeled=False)

    def train_dataloader(self):
        return DataLoader(
            self.clevr_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.clevr_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.clevr_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.clevr_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    dm = CLEVRVQADataModule(num_workers=0, pin_memory=False)
    dm.prepare_data()
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    print(f"question_vocab={dm.vocab_size} answers={dm.answer_names}")
    print(
        "train batch: "
        f"image={batch['image'].shape}, "
        f"question={batch['question'].shape}, "
        f"answer={batch['answer'].shape}"
    )
