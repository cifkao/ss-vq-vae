---
title: Self-Supervised VQ-VAE for One-Shot Music Style Transfer
description: 'Supplementary material'
---

## Contents
{: .no_toc}
* TOC
{:toc}

## Examples

### Artificial inputs
The following is a random sample of the synthetic test set with outputs of our model and the two baselines (musaicing and U+L).
For U+L (Ulyanov and Lebedev), we include both the tuned version ($$\lambda_s=10^{-2.1}\lambda_c$$) and a version with a higher style weight ($$\lambda_s=10\lambda_c$$).

{% include audio_table.html table_data=site.data.examples_synth base_url="https://perso.telecom-paris.fr/ocifka/vqvae_examples/synth" %}

#### 'Good' outputs
{: .no_toc}
Here, we show a sample of the best outputs of our system (below the 5th percentile) according to the LSD metric.

{% include audio_table.html table_data=site.data.examples_synth_good base_url="https://perso.telecom-paris.fr/ocifka/vqvae_examples/synth" %}

#### 'Bad' outputs
{: .no_toc}
Similarly, here is a sample of the worst outputs (above the 95th percentile) according to the LSD metric.

{% include audio_table.html table_data=site.data.examples_synth_bad base_url="https://perso.telecom-paris.fr/ocifka/vqvae_examples/synth" %}

### Real inputs
The following are outputs on the 'Mixing Secrets' test set, first some cherry-picked ones and then a random sample.

#### Selection
{: .no_toc}
{% include audio_table.html table_data=site.data.examples_real_selected base_url="https://perso.telecom-paris.fr/ocifka/vqvae_examples/real" %}

#### Random sample
{: .no_toc}
{% include audio_table.html table_data=site.data.examples_real base_url="https://perso.telecom-paris.fr/ocifka/vqvae_examples/real" %}

## Additional information
This section contains details omitted from the paper for brevity.

### Artificial test set
The artificial test set was created from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/){:target="_blank"} using a set of files held out from the training set.
The audio was synthesized using the [Timbres Of Heaven](http://midkar.com/soundfonts/){:target="_blank"} SoundFont, which was not used for the training set.

We randomly draw 721 content-style input pairs and generate a corresponding ground-truth target for each pair by synthesizing the content input using the instrument (MIDI program) of the style input.
To avoid pairs of extremely different inputs (e.g. bass line + piccolo duet) for which the task would make little sense, we sort all instrument parts into 4 bins using two median splits: on the average pitch and on the average number of voices (simultaneous notes); we then form each pair by drawing two examples from the same bin.
To obtain a balanced distribution of instruments, we limit the total number of examples per MIDI program to 4.

### Timbre dissimilarity metric
The metric uses a sequence of MFCC vectors (only coefficients 2–13) as input and is trained using the triplet loss
(using the code from the [ISMIR 2020 metric learning tutorial](https://github.com/bmcfee/ismir2020-metric-learning)).
The training dataset consists of 7381 triplets (anchor, positive, negative) extracted from the [Mixing Secrets](https://www.cambridge-mt.com/ms/mtk/)
data so that the anchor and the positive example are from the same file and the negative example is from a different file.
The aim is to make the metric good at discriminating between different instruments, but largely pitch-independent.
