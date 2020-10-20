---
title: Hello world
description: Site under construction
---

<script>
var playingAudio = null;

window.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('audio').forEach(function (element) {
    element.addEventListener('play', function() {
      if (playingAudio != null && playingAudio !== element && !playingAudio.paused) {
        playingAudio.pause();
      }
      playingAudio = element;
    });
  });
});
</script>


## Contents
* TOC
{:toc}

## Examples

### Artificial inputs

<table class="audio-table">
<thead>
  <tr>
    {% for name in site.data.examples_synth.header %}
    <th>{{ name }}</th>
    {% endfor %}
  </tr>
</thead>
<tbody>
  {% for row in site.data.examples_synth.body %}
  <tr>
    {% for item in row %}
    <td>
			<audio src="https://perso.telecom-paris.fr/ocifka/vqvae_examples/synth/{{ item.audio_url }}" controls></audio>
			{% if item.label %}
				<span class="audio-label">{{ item.label }}</span>
			{% endif %}
		</td>
    {% endfor %}
  </tr>
  {% endfor %}
</tbody>
</table>

### Real inputs

<table class="audio-table">
<thead>
  <tr>
    {% for name in site.data.examples_real.header %}
    <th>{{ name }}</th>
    {% endfor %}
  </tr>
</thead>
<tbody>
  {% for row in site.data.examples_real.body %}
  <tr>
    {% for item in row %}
    <td>
			<audio src="https://perso.telecom-paris.fr/ocifka/vqvae_examples/real/{{ item.audio_url }}" controls></audio>
			{% if item.label %}
				<span class="audio-label">{{ item.label }}</span>
			{% endif %}
		</td>
    {% endfor %}
  </tr>
  {% endfor %}
</tbody>
</table>

## Additional information
This section contains details omitted from the paper for brevity.

### Artificial test set
The artificial test set was created from the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/){:target="_blank"} using a set of files held out from the training set.
The audio was synthesized using the [Timbres Of Heaven](http://midkar.com/soundfonts/){:target="_blank"} SoundFont, which was not used for the training set.

We randomly draw 721 content-style input pairs and generate a corresponding ground-truth target for each pair by synthesizing the content input using the instrument of the style input.
To avoid pairs of extremely different inputs (e.g. bass line + piccolo duet) for which the task would make little sense, we sort all instrument parts into 4 bins using two median splits: on the average pitch and on the average number of voices (simultaneous notes); we then form each pair by drawing two examples from the same bin.
To obtain a balanced distribution of instruments, we limit the total number of examples per MIDI program to 4.

### Timbre dissimilarity metric
The metric uses a sequence of MFCC vectors (using only coefficients 2–13) as input and is trained using the triplet loss.
The training dataset consists of 7381 triplets (anchor, positive, negative) are extracted from the [Mixing Secrets](https://www.cambridge-mt.com/ms/mtk/)
data so that the anchor and the positive example are from the same file and the negative example is from a different file.
