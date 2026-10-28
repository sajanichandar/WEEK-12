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

Run a controlled comparison that changes sequence length, hidden units, and epochs. Each run writes its terminal output to a separate log in `experiment-results/`:

```powershell
.\run_experiments.ps1
```

The runner compares a baseline (2 epochs, sequence length 100, 128 hidden units) with one change at a time. These short runs demonstrate the effect of each setting; increase the epoch counts for better output. This is character-level generation, so the model predicts the next character rather than the next word.

## Project files

- `train.py` — fetches and preprocesses text, builds/trains the LSTM, and generates text.
- `run_experiments.ps1` — runs the controlled parameter comparisons and saves logs.
- `EXPERIMENT-RESULTS.md` — records the measured losses and sample comparisons.
- `requirements.txt` — Python dependency.
- `data/` — downloaded training corpus (ignored by Git).
- `experiment-results/` — saved training outputs from the comparison runner.

After you have run the implementation successfully, initialize Git in this folder and publish it as its own GitHub repository, following the same layout as the Week-04 example. Keep the report for a later step.
