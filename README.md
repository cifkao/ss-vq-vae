Self-Supervised VQ-VAE for One-Shot Music Style Transfer
========================================================

This is the code repository for the ICASSP 2021 paper 
*Self-Supervised VQ-VAE for One-Shot Music Style Transfer*
by Ondřej Cífka, Alexey Ozerov, Umut Şimşekli, and Gaël Richard.

Copyright 2020 InterDigital R&D and Télécom Paris.

### Links
[:musical_note: Supplementary website](https://adasp.telecom-paris.fr/s/ss-vq-vae) with audio examples  
[:microphone: Demo notebook](https://colab.research.google.com/github/cifkao/ss-vq-vae/blob/main/experiments/colab_demo.ipynb)

Contents
--------

- `src` – the main codebase (the `ss-vq-vae` package); install with `pip install ./src`; usage details [below](#Usage)
- `data` – Jupyter notebooks for data preparation (details [below](#Datasets))
- `experiments` – model configuration, evaluation, and other experimental stuff

Setup
-----

```sh
pip install -r requirements.txt
pip install ./src
```

Usage
-----

To train the model, go to `experiments`, then run:
```sh
python -m ss_vq_vae.models.vqvae_oneshot --logdir=model train
```
This is assuming the training data is prepared (see [below](#Datasets)).

To run the trained model on a dataset, substitute `run` for `train` and specify the input and output paths as arguments (use `run --help` for more information).
Alternatively, see the [`colab_demo.ipynb`](./experiments/colab_demo.ipynb) notebook for how to run the model from Python code.

Datasets
--------
Each dataset used in the paper has a corresponding directory in `data`, containing a Jupyter notebook called `prepare.ipynb` for preparing the dataset:
- the entire training and validation dataset: `data/comb`; combined from LMD and RT (see below)
- [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) (LMD), rendered as audio using SoundFonts
  - the part used as training and validation data: `data/lmd/audio_train`
  - the part used as the 'artificial' test set: `data/lmd/audio_test`
  - both require [downloading](http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz) the raw data and pre-processing it using `data/lmd/note_seq/prepare.ipynb`
  - the following SoundFonts are required (available [here](https://packages.debian.org/buster/fluid-soundfont-gm) and [here](https://musescore.org/en/handbook/soundfonts-and-sfz-files#list)): `FluidR3_GM.sf2`, `TimGM6mb.sf2`, `Arachno SoundFont - Version 1.0.sf2`, `Timbres Of Heaven (XGM) 3.94.sf2`
- RealTracks (RT) from [Band-in-a-Box](https://www.pgmusic.com/) UltraPAK 2018 (not freely available): `data/rt`
- [Mixing Secrets](https://www.cambridge-mt.com/ms/mtk/) data
  - the 'real' test set: `data/mixing_secrets/test`
  - the set of triplets for training the timbre metric: `data/mixing_secrets/metric_train`
  - both require downloading and pre-processing the data using `data/mixing_secrets/download.ipynb`

Acknowledgment
--------------
This work has received funding from the European Union’s Horizon 2020 research and innovation
programme under the Marie Skłodowska-Curie grant agreement No. 765068.
