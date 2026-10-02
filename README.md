# SpellEngine — Speech Recognition + Emotion Recognition for a Voice-Driven Game

The speech engine behind a spell-casting game: the player speaks an incantation into the
microphone, the engine transcribes it (**ASR**, automatic speech recognition) and reads the
emotion in the voice (**SER**, speech emotion recognition), and the two combine into a
**Spell Power** score.

Two models are trained on both tasks at once and compared:

| Model | Architecture | Role |
|---|---|---|
| **A** | Conformer encoder (from scratch) + CTC head + attention-pooling emotion head | Lightweight baseline, exported to ONNX |
| **B** | Whisper-Small + LoRA (PEFT) + emotion classification head | High-accuracy reference |

## Results

On the CREMA-D test split:

| Model | Trainable params | WER ↓ | CER ↓ | Emotion accuracy ↑ | Emotion macro-F1 ↑ |
|---|---|---|---|---|---|
| Conformer (from scratch) | 5.05 M | 11.28% | 5.14% | 53.83% | 0.5253 |
| **Whisper-Small + LoRA** | 2.07 M (of 243.5 M) | **0.15%** | **0.11%** | **75.84%** | **0.7543** |

Per-emotion results for Whisper-Small + LoRA:

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| Neutral | 0.77 | 0.84 | 0.80 |
| Happy | 0.77 | 0.80 | 0.78 |
| Sad | 0.70 | 0.71 | 0.70 |
| Anger | 0.81 | 0.89 | 0.85 |
| Fear | 0.75 | 0.71 | 0.73 |
| Disgust | 0.74 | 0.59 | 0.66 |

**Reading the WER:** CREMA-D contains only 12 distinct sentences, and with the default
file-level split the same actors appear in training and test data. The word error rate
therefore measures how well the models recognise a small, known vocabulary, which matches
the game's use case of a fixed set of incantations, not open-vocabulary transcription. The
notebook supports a stricter speaker-independent split.

## Spell Power

```
power = 100 × intent_weight(emotion) × emotion_confidence × (0.4 + 0.6 × clarity)
clarity = 1 − min(WER, 1)
```

`intent_weight` is a per-emotion weight, from 1.0 for anger down to 0.4 for neutral. Strong
emotion and clear diction give a stronger spell; a mumbled incantation still keeps 40% of
its power.

## How it works

1. **Data discovery** — walks the Kaggle input folder for CREMA-D and builds a manifest in
   which every clip serves as both an ASR example and an SER example.
2. **Lazy data pipeline** — one audio file read per item, log-Mel features, SpecAugment, and
   waveform augmentation (noise, gain, pitch) with Audiomentations.
3. **Model A: Conformer from scratch** — feed-forward, multi-head self-attention and
   depthwise-separable convolution blocks, with a CTC head for transcription and an
   attention-pooling head for emotion.
4. **Model B: Whisper-Small + LoRA** — only about 0.7% of the weights are updated. The
   encoder runs once per step and feeds both the decoder (ASR) and the emotion head.
5. **Multi-task loss** — Kendall uncertainty weighting: each task's loss weight is learned
   rather than hand-tuned.
6. **Training loop** — mixed precision (AMP), OneCycleLR, gradient accumulation, and a
   checkpoint manager that resumes interrupted Kaggle sessions.
7. **Evaluation** — WER and CER with `jiwer`, accuracy and macro-F1 for emotions, confusion
   matrices and a side-by-side comparison table.
8. **Export** — the Conformer to ONNX with dynamic axes and a parity check against PyTorch;
   the Whisper LoRA adapters merged into a standalone model.
9. **`cast_spell`** — the game loop in one function: transcribe, read the emotion, score
   clarity against the expected incantation, and return the Spell Power.

## Repository contents

| File | Description |
|---|---|
| `nlp-mini-projet.ipynb` | The full notebook (code only, without saved outputs) |
| `requirements.txt` | Python dependencies |

## Dataset

[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) (Crowd-Sourced Emotional
Multimodal Actors Dataset): 7,442 audio clips from 91 actors, 6 emotions (neutral, happy,
sad, anger, fear, disgust) and 12 fixed sentences. The dataset is not included in this
repository.

## Run it

The notebook is written for Kaggle:

1. **Settings → Accelerator → GPU T4 x2** (or P100).
2. **Settings → Internet → On**, to install packages and download the Whisper-Small weights.
3. **Add Input** → search for `crema-d` (for example `ejlok1/cremad`).
4. Run all cells. Checkpoints and exports are written under `/kaggle/working/` and training
   resumes from the last checkpoint if a session is interrupted.

## Tech stack

Python, PyTorch, torchaudio, Hugging Face Transformers, PEFT (LoRA), librosa,
Audiomentations, jiwer, scikit-learn, ONNX / ONNX Runtime, pandas, Matplotlib, Seaborn.

## About the project

Mini project for the **Natural Language Processing** module, 2SC IASD, École Supérieure en
Informatique de Sidi Bel Abbès (2025/2026). This notebook is the modeling part of a larger
project in which the engine is served through FastAPI to a Unity 3D game.

## References

- A. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition",
  *Interspeech*, 2020.
- A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision"
  (Whisper), 2022.
- E. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", *ICLR*, 2022.
- A. Kendall, Y. Gal and R. Cipolla, "Multi-Task Learning Using Uncertainty to Weigh Losses
  for Scene Geometry and Semantics", *CVPR*, 2018.
- H. Cao et al., "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset", *IEEE
  Transactions on Affective Computing*, 2014.
