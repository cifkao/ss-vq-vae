---
title: Hello world
description: Site under construction
---
## Artificial inputs

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

## Real inputs

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


<script>
var playingAudio = null;

window.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('audio').forEach(function (element) {
    element.addEventListener('play', function() {
      if (playingAudio != null && !playingAudio.paused) {
        playingAudio.pause();
      }
      playingAudio = element;
    });
  });
});
</script>
