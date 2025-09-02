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
      const now = new Date();
      const rangeMs = { '24h': 24 * 3600000, '30d': 30 * 86400000, '365d': 365 * 86400000 }[period];
      const xRange = [new Date(now.getTime() - rangeMs), now];
      const baseLayout = {
        xaxis: { ...tickSettings[period], type: 'date', title: 'Дата/час', automargin: true, range: xRange },
        margin: { l: 80, r: 80, t: 40, b: 80 },
        legend: { orientation: 'h', y: -0.3 },
        hoverlabel: { namelength: -1 },
        showlegend: true
      };
      const config = { responsive: true, locale: 'bg' };

      const plots = [
        {
          id: 'graph-temp-hum',
          data: [
            {
              x,
              y: data.T_AIR,
              name: 'Температура [°C]',
              type: 'scatter',
              yaxis: 'y1',
              line: { shape: 'spline', color: 'red' },
              hovertemplate: '%{fullData.name}: %{y:.1f} °C<extra></extra>'
            },
            {
              x,
              y: data.REL_HUM,
              name: 'Относителна влажност [%]',
              type: 'scatter',
              yaxis: 'y2',
              line: { shape: 'spline', color: 'blue' },
              hovertemplate: '%{fullData.name}: %{y:.1f} %<extra></extra>'
            }
          ],
          layout: {
            title: 'Температура и Влажност',
            yaxis: { tickformat: '.1f', hoverformat: '.1f', automargin: true, color: 'red', linecolor: 'red' },
            yaxis2: {
              overlaying: 'y',
              side: 'right',
              tickformat: '.1f',
              hoverformat: '.1f',
              showline: true,
              automargin: true,
              color: 'blue',
              linecolor: 'blue'
            }
          }
        },
        {
          id: 'graph-pressure',
          data: [
            {
              x,
              y: data.P_ABS,
              name: 'Налягане - абсолютно [hPa]',
              type: 'scatter',
              yaxis: 'y1',
              line: { shape: 'spline', color: 'green' },
              hovertemplate: '%{fullData.name}: %{y:.1f} hPa<extra></extra>'
            },
            {
              x,
              y: data.P_REL,
              name: 'Налягане - относително [hPa]',
              type: 'scatter',
              yaxis: 'y2',
              line: { shape: 'spline', color: 'purple' },
              hovertemplate: '%{fullData.name}: %{y:.1f} hPa<extra></extra>'
            }
          ],
          layout: {
            title: 'Налягане',
            yaxis: { tickformat: '.1f', hoverformat: '.1f', automargin: true, color: 'green', linecolor: 'green' },
            yaxis2: {
              overlaying: 'y',
              side: 'right',
              tickformat: '.1f',
              hoverformat: '.1f',
              showline: true,
              automargin: true,
              color: 'purple',
              linecolor: 'purple'
            }
          }
        },
        {
          id: 'graph-wind',
          data: [
            {
              x,
              y: data.WIND_SPEED_1,
              name: 'Скорост на вятъра 1 [km/h]',
              type: 'scatter',
              yaxis: 'y1',
              line: { shape: 'spline', color: 'orange' },
              hovertemplate: '%{fullData.name}: %{y:.1f} km/h<extra></extra>'
            },
            {
              x,
              y: data.WIND_SPEED_2,
              name: 'Скорост на вятъра 2 [m/s]',
              type: 'scatter',
              yaxis: 'y2',
              line: { shape: 'spline', color: 'teal' },
              hovertemplate: '%{fullData.name}: %{y:.1f} m/s<extra></extra>'
            }
          ],
          layout: {
            title: 'Вятър',
            yaxis: { tickformat: '.1f', hoverformat: '.1f', automargin: true, color: 'orange', linecolor: 'orange' },
            yaxis2: {
              overlaying: 'y',
              side: 'right',
              tickformat: '.1f',
              hoverformat: '.1f',
              showline: true,
              automargin: true,
              color: 'teal',
              linecolor: 'teal'
            }
          }
        },
        {
          id: 'graph-rain',
          data: [
            {
              x,
              y: data.RAIN_MINUTE,
              name: 'Дъжд [mm]',
              type: 'bar',
              marker: { color: 'blue' },
              hovertemplate: '%{fullData.name}: %{y:.1f} mm<extra></extra>'
            }
          ],
          layout: {
            title: 'Дъжд',
            yaxis: { tickformat: '.1f', hoverformat: '.1f', automargin: true, color: 'blue', linecolor: 'blue' }
          }
        },
        {
          id: 'graph-evaporation',
          data: [
            {
              x,
              y: data.EVAPOR_MINUTE,
              name: 'Изпарение [mm]',
              type: 'bar',
              marker: { color: 'green' },
              hovertemplate: '%{fullData.name}: %{y:.1f} mm<extra></extra>'
            }
          ],
          layout: {
            title: 'Изпарение',
            yaxis: { tickformat: '.1f', hoverformat: '.1f', automargin: true, color: 'green', linecolor: 'green' }
          }
        },
        {
          id: 'graph-solar-radiation',
          data: [
            {
              x,
              y: data.RADIATION,
              name: 'Слънчева радиация [W/m²]',
              type: 'bar',
              marker: { color: 'orange' },
              hovertemplate: '%{fullData.name}: %{y:.1f} W/m²<extra></extra>'
            }
          ],
          layout: {
            title: 'Слънчева радиация',
            yaxis: { tickformat: '.1f', hoverformat: '.1f', automargin: true, color: 'orange', linecolor: 'orange' }
          }
        }
      ];

      plots.forEach(plot => {
        const plotDiv = document.getElementById(plot.id);
        const layout = { ...baseLayout, ...plot.layout };
        try {
          Plotly.react(plotDiv, plot.data, layout, config);
        } catch (err) {
          console.error(`Грешка при начертаване на графика ${plot.id}:`, err);
          plotDiv.innerHTML = '<p>Графиката не може да бъде заредена.</p>';
        }
      });
    }).fail(function(jqxhr, textStatus, error) {
      console.error('Грешка при зареждане на данни за графиките:', error);
      [
        'graph-temp-hum',
        'graph-pressure',
        'graph-wind',
        'graph-rain',
        'graph-evaporation',
        'graph-solar-radiation'
      ].forEach(id => {
        document.getElementById(id).innerHTML = '<p>Грешка при зареждане на данни</p>';
      });
    });
  }
  $('#period-select').change(renderGraphs);
  renderGraphs();
});
