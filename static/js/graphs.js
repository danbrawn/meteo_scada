$(document).ready(function() {
  function renderGraphs() {
    const period = $('#period-select').val();
    $.getJSON(`/graph_data?period=${period}`, function(data) {
      const x = data.DateRef.map(d => new Date(d));
      const tickSettings = {
        '24h': { dtick: 3600000, tickformat: '%H:%M' },
        '30d': { dtick: 86400000, tickformat: '%d.%m' },
        '365d': { dtick: 'M1', tickformat: '%b' }
      };
      const baseLayout = { xaxis: tickSettings[period] };
      const config = { responsive: true };

      const plots = [
        {
          id: 'graph-temp-hum',
          data: [
            { x, y: data.T_AIR, name: 'T_AIR', type: 'scatter' },
            { x, y: data.REL_HUM, name: 'REL_HUM', type: 'scatter' }
          ],
          title: 'Температура и Влажност'
        },
        {
          id: 'graph-pressure',
          data: [
            { x, y: data.P_ABS, name: 'P_ABS', type: 'scatter' },
            { x, y: data.P_REL, name: 'P_REL', type: 'scatter' }
          ],
          title: 'Налягане'
        },
        {
          id: 'graph-wind',
          data: [
            { x, y: data.WIND_SPEED_1, name: 'WIND_SPEED_1', type: 'scatter' },
            { x, y: data.WIND_SPEED_2, name: 'WIND_SPEED_2', type: 'scatter' }
          ],
          title: 'Вятър'
        },
        {
          id: 'graph-rain',
          data: [
            { x, y: data.RAIN_MINUTE, name: 'RAIN_MINUTE', type: 'bar' }
          ],
          title: 'Дъжд'
        },
        {
          id: 'graph-radiation',
          data: [
            { x, y: data.RADIATION, name: 'RADIATION', type: 'scatter' }
          ],
          title: 'Радиация'
        }
      ];

      plots.forEach(plot => {
        Plotly.purge(plot.id);
        Plotly.newPlot(plot.id, plot.data, { ...baseLayout, title: plot.title }, config);
        Plotly.relayout(plot.id, { margin: { l: 80, r: 80, t: 40, b: 40 } });
      });
    });
  }
  $('#period-select').change(renderGraphs);
  renderGraphs();
});
