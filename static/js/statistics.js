$(document).ready(function () {
  function listToHtml(items) {
    return (
      '<ul class="stats-list">' +
      items
        .map(
          item =>
            `<li class="stats-item"><span class="stats-label">${item.label}</span>` +
            `<span class="stats-value">${item.value}</span></li>`
        )
        .join('') +
      '</ul>'
    );
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
