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
