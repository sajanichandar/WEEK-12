"""Train a character-level LSTM and generate text from a prompt."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import torch
from torch import nn


DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
DATA_PATH = Path(__file__).parent / "data" / "shakespeare.txt"


def load_text() -> str:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Downloading Tiny Shakespeare dataset to {DATA_PATH} …")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    text = DATA_PATH.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Dataset is empty: {DATA_PATH}")
    return text


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        states, _ = self.lstm(embedded)
        return self.output(states)


def generate(model: CharLSTM, seed: str, stoi: dict[str, int], itos: list[str],
             length: int, temperature: float, device: torch.device) -> str:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    unknown = [char for char in seed if char not in stoi]
    if unknown:
        raise ValueError(f"Seed contains characters not in the dataset: {unknown!r}")
    context = [stoi[char] for char in seed] or [0]
    result = list(seed)
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)[0, -1] / temperature
            next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
            result.append(itos[next_id])
            context.append(next_id)
    return "".join(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq-length", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", default="ROMEO:")
    parser.add_argument("--generate-length", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    args = parser.parse_args()

    if min(args.epochs, args.seq_length, args.hidden_size, args.batch_size) < 1:
        parser.error("epochs, seq-length, hidden-size, and batch-size must be positive")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = load_text()
    alphabet = sorted(set(text))
    stoi = {char: i for i, char in enumerate(alphabet)}
    itos = alphabet
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)

    # Non-overlapping chunks form next-character input/target pairs.
    usable = ((len(encoded) - 1) // args.seq_length) * args.seq_length
    if usable == 0:
        raise ValueError("Dataset must contain at least seq-length + 1 characters.")
    inputs = encoded[:usable].reshape(-1, args.seq_length)
    targets = encoded[1:usable + 1].reshape(-1, args.seq_length)

    model = CharLSTM(len(alphabet), args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    print(f"Characters: {len(text):,} | Vocabulary: {len(alphabet)} | Device: {device}")
    print(f"Training sequences: {len(inputs):,} | Sequence length: {args.seq_length}")

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(inputs))
        total_loss = 0.0
        count = 0
        for start in range(0, len(order), args.batch_size):
            batch_ids = order[start:start + args.batch_size]
            x = inputs[batch_ids].to(device)
            y = targets[batch_ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, len(alphabet)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_ids)
            count += len(batch_ids)
        print(f"Epoch {epoch + 1:>3}/{args.epochs} | loss: {total_loss / count:.4f}")

    sample = generate(model, args.seed, stoi, itos, args.generate_length,
                      args.temperature, device)
    output_path = Path(__file__).parent / "generated.txt"
    output_path.write_text(sample, encoding="utf-8")
    print("\n--- Generated sample ---\n")
    print(sample)
    print(f"\nSaved generated text to {output_path}")


if __name__ == "__main__":
    main()
