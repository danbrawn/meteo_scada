$(document).ready(function() {
  function renderStats() {
    $('#stats-today').html('<p>Температура: 10°C - 20°C</p>');
    $('#stats-month').html('<p>Макс температура: 25°C</p>');
    $('#stats-year').html('<p>Годишен дъжд: 100 mm</p>');
  }
  renderStats();
});
