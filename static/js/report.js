$(document).ready(function() {
  const params = [
    { key: 'T_AIR', name: 'Температура °C' },
    { key: 'REL_HUM', name: 'Относителна влажност %' },
    { key: 'P_REL', name: 'Относително налягане hPa' },
    { key: 'WIND_SPEED_1', name: 'Скорост на вятъра km/h' },
    { key: 'WIND_SPEED_2', name: 'Скорост на вятъра m/s' },
    { key: 'WIND_DIR', name: 'Посока на вятъра (DEG)' },
    { key: 'RAIN', name: 'Валежи (l/m2)' },
    { key: 'T_AIR_14', name: 'Температура 14:00 °C' },
    { key: 'REL_HUM_14', name: 'Отн. влажност 14:00 %' },
    { key: 'P_REL_14', name: 'Отн. налягане 14:00 hPa' },
    { key: 'EVAPOR_DAY', name: 'Изпарение mm/d' }
  ];
  const days = Array.from({length: 31}, (_, i) => i + 1);

  function buildTable(data) {
    let thead = '<tr><th class="sticky-col">Параметър</th>' +
      days.map(d => `<th>${d}</th>`).join('') + '</tr>';
    $('#report-table thead').html(thead);
    let rows = params.map(p => {
      const values = data[p.key] || [];
      const cells = days.map(d => {
        const v = values[d-1];
        return `<td>${v !== undefined && v !== null ? v : ''}</td>`;
      }).join('');
      return `<tr><td class="sticky-col">${p.name}</td>${cells}</tr>`;
    }).join('');
    $('#report-table tbody').html(rows);
  }

  function loadData(year, month) {
    $.getJSON(`/report_data?year=${year}&month=${month}`)
      .done(function(data) {
        buildTable(data);
      })
      .fail(function() {
        console.error('Failed to load report data');
        buildTable({});
      });
  }

  const monthSelect = $('#month-select');
  for (let m = 1; m <= 12; m++) {
    monthSelect.append(`<option value="${m}">${m}</option>`);
  }
  const yearSelect = $('#year-select');
  const currentYear = new Date().getFullYear();
  for (let y = currentYear - 5; y <= currentYear; y++) {
    yearSelect.append(`<option value="${y}">${y}</option>`);
  }
  monthSelect.val(new Date().getMonth() + 1);
  yearSelect.val(currentYear);

  $('#load-report').on('click', function() {
    loadData(yearSelect.val(), monthSelect.val());
  });

  loadData(currentYear, new Date().getMonth() + 1);
});
