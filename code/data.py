from pathlib import Path
from torch.utils.data import Dataset

class ShinraDataset(Dataset):
    def __init__(self, data_dir, tokenizer, is_train=True):
        self.is_train = is_train
        self.base_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self._read(self.base_dir)

    def _read(self, base_dir):
        dataset_path = "train.iob" if self.is_train else "test.iob"
        self.vocab = self._read_vocab(base_dir / "vocab.txt")
        self.tokens, self.labels, self.infos = self._read_iob(base_dir / dataset_path)
        self.label_vocab = self._create_label_vocab(self.labels)

    def _read_vocab(self, path):
        with open(path, 'r') as f:
            vocab = [v for v in f.read().split('\n') if v != '']
        return vocab

    def _read_iob(self, path):
        tokens = []
        labels = []
        infos = []
        with open(path, "r") as f:
            docs = [l for l in f.read().split('\n\n') if l != '']
        for doc in docs:
            sents = [l for l in doc.split('\n') if l != '']
            page_id = sents[0]
            line_id = sents[1]

            iobs = [s.split('\t') for s in sents[2:]]
            tokens.append([int(l[0]) for l in iobs])
            labels.append([l[3] for l in iobs])
            infos.append({"page_id": page_id, "line_id": line_id, "text_offset": [[l[1], l[2]] for l in iobs]})

        return tokens, labels, infos

    def _create_label_vocab(self, labels):
        label_vocab = {"O": 0}
        for label in labels:
            for l in label:
                if l not in label_vocab:
                    label_vocab[l] = len(label_vocab)
        return label_vocab

    def _preprocess(self, tokens, labels):
        tokens = ["[CLS]"] + [self.vocab[t] for t in tokens][:512]
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        labels = ["O"] + [self.label_vocab[l] for l in labels][:512]
        return tokens, labels

    def __getitem__(self, item):
        tokens, labels = self._preprocess(self.tokens[item], self.labels[item])
        return tokens, labels, self.infos[item]

    def __len__(self):
        return len(self.tokens)

def my_collate_fn(batch):
    tokens, labels, infos = list(zip(*batch))
    return tokens, labels, infos

