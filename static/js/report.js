$(document).ready(function() {
  const params = [
    { key: 'T_AIR', name: 'Температура', unit: '°C' },
    { key: 'T_WATER', name: 'Температура на водата', unit: '°C' },
    { key: 'REL_HUM', name: 'Относителна влажност', unit: '%' },
    { key: 'P_REL', name: 'Относително налягане', unit: 'hPa' },
    { key: 'P_ABS', name: 'Абсолютно налягане', unit: 'hPa' },
    { key: 'WIND_SPEED_1', name: 'Скорост на вятъра', unit: 'km/h' },
    { key: 'WIND_SPEED_2', name: 'Скорост на вятъра', unit: 'm/s' },
    { key: 'WIND_DIR', name: 'Посока на вятъра', unit: 'DEG' },
    { key: 'RAIN', name: 'Валежи', unit: 'l/m²' },
    { key: 'T_AIR_14', name: 'Температура 14:00', unit: '°C' },
    { key: 'REL_HUM_14', name: 'Отн. влажност 14:00', unit: '%' },
    { key: 'P_REL_14', name: 'Отн. налягане 14:00', unit: 'hPa' },
    { key: 'EVAPOR_DAY', name: 'Изпарение', unit: 'mm/d' }
  ];
  let days = [];
  function updateDays(year, month) {
    const numDays = new Date(Number(year), Number(month), 0).getDate();
    days = Array.from({ length: numDays }, (_, i) => i + 1);
  }
  let currentData = {};

  function buildTable(data) {
    let thead = '<tr><th class="sticky-col">Параметър</th>' +
      days.map(d => `<th>${d}</th>`).join('') + '</tr>';
    $('#report-table thead').html(thead);
    let rows = params.map(p => {
      const values = data[p.key] || [];
      const cells = days.map(d => {
        const v = values[d-1];
        const num = typeof v === 'string' ? parseFloat(v.replace(',', '.')) : v;
        return `<td>${v !== undefined && v !== null && !isNaN(num) ? num.toLocaleString('bg-BG', { minimumFractionDigits: 1, maximumFractionDigits: 1, useGrouping: false }) : ''}</td>`;
      }).join('');
      return `<tr><td class="sticky-col">${p.name}, ${p.unit}</td>${cells}</tr>`;
    }).join('');
    $('#report-table tbody').html(rows);
  }

  function loadData(year, month) {
    updateDays(year, month);
    $.getJSON(`/report_data?year=${year}&month=${month}`)
      .done(function(data) {
        currentData = data;
        buildTable(data);
      })
      .fail(function(jqXHR) {
        console.error('Failed to load report data', jqXHR.responseText);
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

  $('#export-csv').on('click', function() {
    const year = yearSelect.val();
    const month = String(monthSelect.val()).padStart(2, '0');
    let csv = ['Параметър;' + days.join(';')];
    params.forEach(p => {
      const values = currentData[p.key] || [];
      const row = [`${p.name}, ${p.unit}`];
      for (let i = 0; i < days.length; i++) {
        const v = values[i];
        const num = typeof v === 'string' ? parseFloat(v.replace(',', '.')) : v;
        row.push(v !== undefined && v !== null && !isNaN(num) ? num.toLocaleString('bg-BG', { minimumFractionDigits: 1, maximumFractionDigits: 1, useGrouping: false }) : '');
      }
        csv.push(row.join(';'));
    });
    const csvContent = csv.join('\n');
    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `Meteo_Dushanci_${month}_${year}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  });
  loadData(currentYear, new Date().getMonth() + 1);
});
