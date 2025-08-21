$(document).ready(function() {
  function renderGraphs() {
    const period = $('#period-select').val();
    $.getJSON(`/graph_data?period=${period}`, function(data) {
      const x = data.DateRef.map(d => new Date(d));
      const tickSettings = {
        '24h': {dtick: 3600000, tickformat: '%H:%M'},
        '30d': {dtick: 86400000, tickformat: '%d.%m'},
        '365d': {dtick: 'M1', tickformat: '%b'}
      };
      const baseLayout = {
        margin: { t: 30 },
        height: 400,
        xaxis: tickSettings[period]
      };
      const config = { responsive: true };

      Plotly.newPlot('graph-temp-hum', [
        {x: x, y: data.T_AIR, name: 'T_AIR', type: 'scatter'},
        {x: x, y: data.REL_HUM, name: 'REL_HUM', type: 'scatter'}
      ], {...baseLayout, title: 'Температура и Влажност'}, config);

      Plotly.newPlot('graph-pressure', [
        {x: x, y: data.P_ABS, name: 'P_ABS', type: 'scatter'},
        {x: x, y: data.P_REL, name: 'P_REL', type: 'scatter'}
      ], {...baseLayout, title: 'Налягане'}, config);

      Plotly.newPlot('graph-wind', [
        {x: x, y: data.WIND_SPEED_1, name: 'WIND_SPEED_1', type: 'scatter'},
        {x: x, y: data.WIND_SPEED_2, name: 'WIND_SPEED_2', type: 'scatter'}
      ], {...baseLayout, title: 'Вятър'}, config);

      Plotly.newPlot('graph-rain', [
        {x: x, y: data.RAIN_MINUTE, name: 'RAIN_MINUTE', type: 'bar'}
      ], {...baseLayout, title: 'Дъжд'}, config);

      Plotly.newPlot('graph-radiation', [
        {x: x, y: data.RADIATION, name: 'RADIATION', type: 'scatter'}
      ], {...baseLayout, title: 'Радиация'}, config);
    });
  }
  $('#period-select').change(renderGraphs);
  renderGraphs();
});
