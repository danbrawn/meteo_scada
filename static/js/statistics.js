$(document).ready(function () {
  function listToHtml(items) {
    return '<ul>' + items.map(item => `<li>${item}</li>`).join('') + '</ul>';

  }

  fetch('/statistics_data')
    .then(response => response.json())
    .then(data => {
      $('#stats-today').html(listToHtml(data.today || []));
      $('#stats-month').html(listToHtml(data.month || []));
      $('#stats-year').html(listToHtml(data.year || []));
      $('#stats-alltime').html(listToHtml(data.all || []));
    })
    .catch(err => {
      console.error('Error loading statistics', err);
    });

});
