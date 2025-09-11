# GEMINI.md

## Project Overview

This project is dedicated to building an Automatic Speech Recognition (ASR) system for Amdo Tibetan, a dialect of the Tibetan language. It utilizes the XBMU-AMDO31 speech corpus.

The project is structured as an `icefall` recipe, a framework for end-to-end speech recognition research. The core technologies employed are:

*   **[icefall](https://github.com/k2-fsa/icefall):** The main framework for training and decoding.
*   **[k2](https://github.com/k2-fsa/k2):** Used for finite-state automata (FSA) operations and for computing the RNN-T loss function.
*   **[lhotse](https://github.com/lhotse-speech/lhotse):** A library for speech and audio data handling, used here for data preparation and augmentation.
*   **PyTorch:** The deep learning framework used for model training.

The primary model architecture is a **Pruned RNN-T (Transducer)** with a **Conformer** encoder, as implemented in the `pruned_transducer_stateless5` and `pruned_transducer_stateless7` directories.

## Building and Running

The main workflow is divided into data preparation, model training, and decoding.

### 1. Data Preparation

The entire data preparation process is orchestrated by the `prepare.sh` script. This script handles:

*   Downloading the XBMU-AMDO31 dataset and supplementary materials like the MUSAN noise corpus and pre-trained language models.
*   Creating `lhotse` manifests for the training, development, and test sets.
*   Computing Fbank features using `local/compute_fbank_xbmu_amdo31.py`.
*   Preparing the language data, including building a BPE (Byte Pair Encoding) model and compiling language model (LM) graphs.

To run the data preparation, execute:

```bash
./prepare.sh
```

### 2. Model Training

The training process is handled by the `train.py` script within each model directory. For example, to train the `pruned_transducer_stateless5` model, you would use the following commands, as found in `RESULTS.md`:

```bash
# Ensure you are in the correct directory
# cd egs/xbmu_amdo31/ASR

# Set the visible CUDA device
export CUDA_VISIBLE_DEVICES="0"

# Run the training script
./pruned_transducer_stateless5/train.py
```

Key training parameters, such as the number of epochs, learning rate, and model dimensions, can be adjusted via command-line arguments in the `train.py` script.

### 3. Decoding

After a model is trained, you can perform inference (decoding) using the `decode.py` script in the corresponding model directory. The `RESULTS.md` file provides the exact commands for different decoding methods.

For example, to decode with the `pruned_transducer_stateless5` model using various beam search methods:

```bash
for method in greedy_search beam_search modified_beam_search;
do
./pruned_transducer_stateless5/decode.py \
    --epoch 28 \
    --avg 23 \
    --exp-dir ./pruned_transducer_stateless5/exp \
    --max-duration 600 \
    --decoding-method $method
done
```

This will generate recognition results in the experiment directory (`pruned_transducer_stateless5/exp/`).

## Development Conventions

*   **Data Management:** The project relies heavily on `lhotse` for data preparation. All data-related steps are encapsulated in `prepare.sh`, which generates manifests and features in the `data/` directory.
*   **Model-Specific Code:** Each ASR model (e.g., `pruned_transducer_stateless5`) is self-contained in its own directory, which includes scripts for training, decoding, and potentially model-specific data modules (`asr_datamodule.py`).
*   **Configuration:** Training and decoding scripts use `argparse` for configuration, allowing for easy modification of hyperparameters from the command line.
*   **Experiment Tracking:** Results, training commands, and pre-trained model links are documented in `RESULTS.md`. The training scripts log progress and save checkpoints in an `exp` directory within each model's folder.
*   **Dependencies:** The project's dependencies are managed by the `icefall` framework. Key Python libraries include `torch`, `k2`, `lhotse`, and `sentencepiece`.
