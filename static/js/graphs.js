$(document).ready(function() {
  function renderGraphs() {
    const x = [1,2,3,4,5];
    Plotly.newPlot('graph-temp-hum', [
      {x: x, y: [10,12,14,13,11], name: 'T_AIR', type: 'scatter'},
      {x: x, y: [60,58,65,62,61], name: 'REL_HUM', type: 'scatter'}
    ]);
    Plotly.newPlot('graph-pressure', [
      {x: x, y: [950,952,951,953,954], name: 'P_ABS', type: 'scatter'},
      {x: x, y: [930,932,931,933,934], name: 'P_REL', type: 'scatter'}
    ]);
    Plotly.newPlot('graph-wind', [
      {x: x, y: [5,7,6,8,5], name: 'WIND_SPEED_1', type: 'scatter'},
      {x: x, y: [3,4,5,4,3], name: 'WIND_SPEED_2', type: 'scatter'}
    ]);
    Plotly.newPlot('graph-rain', [
      {x: x, y: [0,1,0,2,0], name: 'RAIN_MINUTE', type: 'bar'}
    ]);
    Plotly.newPlot('graph-radiation', [
      {x: x, y: [100,200,150,250,300], name: 'RADIATION', type: 'scatter'}
    ]);
  }
  $('#period-select').change(renderGraphs);
  renderGraphs();
});
