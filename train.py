import argparse

from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForTokenClassification, BertJapaneseTokenizer

from code.data import ShinraDataset, my_collate_fn

device = "cuda" if torch.cuda.is_available else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16, help="batch size during training")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--epoch", type=int, default=10, help="epoch")
    parser.add_argument("--model_path", type=str, default="./bert.model", help="path for saving model")
    parser.add_argument("--input_path", type=str, help="path for input path")

    args = parser.parse_args()
    return args


def train(model, dataset, lr=5e-5, batch_size=16, epoch=10):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for e in range(epoch):
        all_loss = 0
        for tokens, labels, infos in tqdm(dataloader):
            optimizer.zero_grad()

            input_x = pad_sequence([torch.tensor(token)
                                    for token in tokens], batch_first=True, padding_value=0).to(device)
            input_y = pad_sequence([torch.tensor(label)
                                    for label in labels], batch_first=True, padding_value=0).to(device)

            mask = input_x > 0
            output = model(input_x, labels=input_y, attention_mask=mask)
            loss = output[0]

            loss.backward()
            optimizer.step()
            all_loss += loss.item()

        losses.append(all_loss / len(dataloader))

    return losses


if __name__ == "__main__":
    # load argument
    args = parse_arguments()

    # load tokenizer
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

    # load dataset
    dataset = ShinraDataset(args.input_path, tokenizer)

    # load model
    model = BertForTokenClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=len(dataset.label_vocab)).to(device)

    losses = train(model, dataset, lr=args.lr, batch_size=args.batch_size, epoch=args.epoch)
    torch.save(model.state_dict(), args.model_path)

