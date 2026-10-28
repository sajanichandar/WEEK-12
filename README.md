# Tutorial 12: LSTM Text Generation

Train a character-level LSTM on Shakespeare text, then generate new text from a seed phrase. The script also supports experiments with sequence length, hidden units, and epochs.

## Open in VS Code (Windows)

1. Open this project folder in VS Code (`File` → `Open Folder…`).
2. Open **Terminal** → **New Terminal**.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.
4. Install the dependency: `python -m pip install --upgrade pip` then `pip install -r requirements.txt`.
5. In VS Code, press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select `.venv`.
6. Run the starter experiment:

   ```powershell
   python train.py --epochs 10 --seq-length 100 --hidden-size 128 --seed "ROMEO:"
   ```

The script downloads the public-domain Tiny Shakespeare corpus on its first run and stores it in `data/`. Internet access is needed only for that first download. Generated text is saved to `generated.txt`.

## Experiments

Try changing one argument at a time and compare the printed loss and generated sample:

```powershell
python train.py --epochs 20 --seq-length 80 --hidden-size 128 --seed "JULIET:"
python train.py --epochs 20 --seq-length 120 --hidden-size 256 --seed "To be, or not to be:"
```

Short runs demonstrate the workflow; use more epochs for better output. This is character-level generation, so the model predicts the next character rather than the next word.

## Project files

- `train.py` — fetches and preprocesses text, builds/trains the LSTM, and generates text.
- `requirements.txt` — Python dependency.
- `data/` — downloaded training corpus (ignored by Git).

After you have run the implementation successfully, initialize Git in this folder and publish it as its own GitHub repository, following the same layout as the Week-04 example. Keep the report for a later step.
