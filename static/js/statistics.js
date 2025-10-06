$(document).ready(function () {
  function renderList(items) {
    if (!items || !items.length) {
      return '';
    }
    return (
      '<ul class="stats-list">' +
      items
        .map(item => {
          const valueHtml = Array.isArray(item.value)
            ? item.value
                .map(v => `<span class="stats-subvalue">${v}</span>`)
                .join('')
            : item.value;
          return (
            `<li class="stats-item"><span class="stats-label">${item.label}</span>` +
            `<span class="stats-value">${valueHtml}</span></li>`
          );
        })
        .join('') +
      '</ul>'
    );
  }

  function groupToHtml(grouped) {
    if (!grouped) {
      return '';
    }
    const leftHtml = renderList(grouped.left || []);
    const rightHtml = renderList(grouped.right || []);
    return (
      '<div class="stats-columns">' +
      `<div class="stats-column stats-column-left">${leftHtml}</div>` +
      `<div class="stats-column stats-column-right">${rightHtml}</div>` +
      '</div>'
    );
  }

  function fetchStatistics() {
    fetch('/statistics_data')
      .then(response => response.json())
      .then(data => {
        $('#stats-today').html(groupToHtml(data.today));
        $('#stats-month').html(groupToHtml(data.month));
        $('#stats-year').html(groupToHtml(data.year));
        $('#stats-alltime').html(groupToHtml(data.all));
      })
      .catch(err => {
        console.error('Error loading statistics', err);
      });
  }

  fetchStatistics();
  setInterval(fetchStatistics, 60000);
});
