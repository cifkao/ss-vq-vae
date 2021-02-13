---
title: Self-Supervised VQ-VAE for One-Shot Music Style Transfer
custom_header: |
  <h1 class="project-name">Self-Supervised VQ-VAE for One-Shot Music Style Transfer</h1>
  <div class="buttons">
    <a href="https://arxiv.org/abs/2102.05749" class="btn" role="button">
      <i class="fa fa-file-pdf-o"></i>
      Paper
    </a>
    <a href="https://github.com/cifkao/ss-vq-vae" class="btn" role="button">
      <i class="fa fa-code"></i>
      Code
    </a>
    <a href="#examples" class="btn" role="button">
      <i class="fa fa-play-circle"></i>
      Examples
    </a>
  </div>
---

## Paper
{: .no_toc}

<blockquote>
  <p>
    <a href="https://scholar.google.cz/citations?user=Uz7XFWoAAAAJ" target="_blank">Ondřej Cífka</a>, <a href="https://scholar.google.com/citations?user=LnV-0z0AAAAJ" target="_blank">Alexey Ozerov</a>, <a href="https://scholar.google.com.tr/citations?user=CuArAkgAAAAJ" target="_blank">Umut Şimşekli</a> and <a href="https://scholar.google.fr/citations?user=xn70tPIAAAAJ" target="_blank">Gaël Richard</a>. <a href="https://arxiv.org/abs/2102.05749">"Self-Supervised VQ-VAE for One-Shot Music Style Transfer."</a> Accepted to the <em>2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)</em>, 2021.
  </p>
</blockquote>

<div class="badges">
<div data-badge-popover="bottom" data-badge-type="1" data-arxiv-id="2102.05749" data-hide-no-mentions="true" class="altmetric-embed"></div>
<a href="https://github.com/cifkao/ss-vq-vae"><img alt="GitHub repo stars" src="https://img.shields.io/github/stars/cifkao/ss-vq-vae?style=social"></a>
</div>

### Abstract
{: .no_toc}
Neural style transfer, allowing to apply the artistic style of one image to another, has become one of the most widely showcased computer vision applications shortly after its introduction. In contrast, related tasks in the music audio domain remained, until recently, largely untackled. While several style conversion methods tailored to musical signals have been proposed, most lack the 'one-shot' capability of classical image style transfer algorithms. On the other hand, the results of existing one-shot audio style transfer methods on musical inputs are not as compelling. In this work, we are specifically interested in the problem of **one-shot timbre transfer**. We present a novel method for this task, based on an extension of the **vector-quantized variational autoencoder** (VQ-VAE), along with a simple **self-supervised learning** strategy designed to obtain disentangled representations of timbre and pitch. We evaluate the method using a set of objective metrics and show that it is able to outperform selected baselines.

## Contents
{: .no_toc}
* TOC
{:toc}

## Examples

### Artificial inputs
The following is a random sample of the synthetic test set with outputs of our model and the two baselines (musaicing and U+L).
For U+L (Ulyanov and Lebedev), we include both the tuned version ($$\lambda_s=10^{-2.1}\lambda_c$$) and a version with a higher style weight ($$\lambda_s=10\lambda_c$$).

{% include audio_table.html table_data=site.data.examples_synth base_url="https://adasp.telecom-paris.fr/rc-ext/demos_companion-pages/vqvae_examples/synth" %}

#### 'Good' outputs
{: .no_toc}
Here, we show a sample of the best outputs of our system (below the 5th percentile) according to the LSD metric.

{% include audio_table.html table_data=site.data.examples_synth_good base_url="https://adasp.telecom-paris.fr/rc-ext/demos_companion-pages/vqvae_examples/synth" %}

#### 'Bad' outputs
{: .no_toc}
Similarly, here is a sample of the worst outputs (above the 95th percentile) according to the LSD metric.

{% include audio_table.html table_data=site.data.examples_synth_bad base_url="https://adasp.telecom-paris.fr/rc-ext/demos_companion-pages/vqvae_examples/synth" %}

### Real inputs
The following are outputs on the 'Mixing Secrets' test set, first some cherry-picked ones and then a random sample.

#### Selection
{: .no_toc}
{% include audio_table.html table_data=site.data.examples_real_selected base_url="https://adasp.telecom-paris.fr/rc-ext/demos_companion-pages/vqvae_examples/real" %}

#### Random sample
{: .no_toc}
{% include audio_table.html table_data=site.data.examples_real base_url="https://adasp.telecom-paris.fr/rc-ext/demos_companion-pages/vqvae_examples/real" %}

## Additional information
This section contains details omitted from the paper for brevity.

### Artificial test set
The artificial test set was created from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) using a set of files held out from the training set.
The audio was synthesized using the [Timbres Of Heaven](http://midkar.com/soundfonts/) SoundFont, which was not used for the training set.

We randomly drew 721 content-style input pairs and generated a corresponding ground-truth target for each pair by synthesizing the content input using the instrument (MIDI program) of the style input.
To avoid pairs of extremely different inputs (e.g. bass line + piccolo duet) for which the task would make little sense, we sorted all instrument parts into 4 bins using two median splits: on the average pitch and on the average number of voices (simultaneous notes); we then formed each pair by drawing two examples from the same bin.
To obtain a balanced distribution of instruments, we limited the total number of examples per MIDI program to 4.

### 'Real' test set
The 'real data' test set was created from the [Mixing Secrets](https://www.cambridge-mt.com/ms/mtk/) collection.
We used filename matching to exclude most drum, vocal and multi-instrument tracks and to balance the distribution of the remaining instruments
(dominated by electric guitar and bass).
To form the input pairs, we performed the same binning procedure as for the artificial test set, using the [multi-pitch MELODIA](https://essentia.upf.edu/reference/std_MultiPitchMelodia.html)
algorithm to estimate the average pitch and number of voices.


### Timbre dissimilarity metric
The metric uses a sequence of MFCC vectors (only coefficients 2–13) as input and is trained using the triplet loss
(using the code from the [ISMIR 2020 metric learning tutorial](https://github.com/bmcfee/ismir2020-metric-learning)).
The training dataset consists of 7381 triplets (anchor, positive, negative) extracted from the [Mixing Secrets](https://www.cambridge-mt.com/ms/mtk/)
data so that the anchor and the positive example are from the same file and the negative example is from a different file.
The aim is to make the metric good at discriminating between different instruments, but largely pitch-independent.

<script type='text/javascript' src='https://d1bxh8uas1mnw7.cloudfront.net/assets/embed.js'></script>
