// Render Plotly graphs with localized labels, including solar radiation axes
$(document).ready(function() {
  function renderGraphs() {
    const period = $('#period-select').val();
    $.getJSON(`/graph_data?period=${period}`, function(data) {
      if (!data.DateRef || data.DateRef.length === 0) {
        [
          'graph-temp-hum',
          'graph-pressure',
          'graph-wind',
          'graph-rain',
          'graph-evaporation',
          'graph-solar-radiation'
        ].forEach(id => {
          document.getElementById(id).innerHTML = '<p>Няма данни за избрания период</p>';
        });
        return;
      }
      const x = data.DateRef.map(d => new Date(d));
      const tickSettings = {
        '24h': { dtick: 3600000, tickformat: '%H:%M' },
        '30d': { dtick: 86400000, tickformat: '%d.%m' },
        '365d': { dtick: 'M1', tickformat: '%b' }
      };
      const baseLayout = {
        xaxis: { ...tickSettings[period], type: 'date', title: 'Дата/час', automargin: true },
        margin: { l: 80, r: 80, t: 40, b: 80 },
        legend: { orientation: 'h', y: -0.3 }
      };
      const config = { responsive: true, locale: 'bg' };

      const plots = [
        {
          id: 'graph-temp-hum',
          data: [
            { x, y: data.T_AIR, name: 'Температура', type: 'scatter', yaxis: 'y1', line: { shape: 'spline' } },
            { x, y: data.REL_HUM, name: 'Относителна влажност', type: 'scatter', yaxis: 'y2', line: { shape: 'spline' } }
          ],
          layout: {
            title: 'Температура и Влажност',
            yaxis: { title: 'Температура (°C)' },
            yaxis2: { title: 'Влажност (%)', overlaying: 'y', side: 'right' }
          }
        },
        {
          id: 'graph-pressure',
          data: [
            { x, y: data.P_ABS, name: 'Налягане - абсолютно', type: 'scatter', line: { shape: 'spline' } },
            { x, y: data.P_REL, name: 'Налягане - относително', type: 'scatter', line: { shape: 'spline' } }
          ],
          layout: {
            title: 'Налягане',
            yaxis: { title: 'Налягане (hPa)' }
          }
        },
        {
          id: 'graph-wind',
          data: [
            { x, y: data.WIND_SPEED_1, name: 'Скорост на вятъра 1', type: 'scatter', yaxis: 'y1', line: { shape: 'spline' } },
            { x, y: data.WIND_SPEED_2, name: 'Скорост на вятъра 2', type: 'scatter', yaxis: 'y2', line: { shape: 'spline' } }
          ],
          layout: {
            title: 'Вятър',
            yaxis: { title: 'Скорост 1 (km/h)' },
            yaxis2: { title: 'Скорост 2 (m/s)', overlaying: 'y', side: 'right' }
          }
        },
        {
          id: 'graph-rain',
          data: [
            { x, y: data.RAIN_MINUTE, name: 'Дъжд', type: 'bar', marker: { color: 'blue' } }
          ],
          layout: {
            title: 'Дъжд',
            yaxis: { title: 'Дъжд (mm)' }
          }
        },
        {
          id: 'graph-evaporation',
          data: [
            { x, y: data.EVAPOR_MINUTE, name: 'Изпарение', type: 'bar', marker: { color: 'green' } }
          ],
          layout: {
            title: 'Изпарение',
            yaxis: { title: 'Изпарение (mm)' }
          }
        },
        {
          id: 'graph-solar-radiation',
          data: [
            { x, y: data.RADIATION, name: 'Слънчева радиация', type: 'bar', marker: { color: 'orange' } }
          ],
          layout: {
            title: 'Слънчева радиация',
            yaxis: { title: 'Слънчева радиация (W/m²)' }
          }
        }
      ];

      plots.forEach(plot => {
        Plotly.purge(plot.id);
        const layout = { ...baseLayout, ...plot.layout };
        Plotly.newPlot(plot.id, plot.data, layout, config);
      });
    });
  }
  $('#period-select').change(renderGraphs);
  renderGraphs();
});
