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
      const isEnergyUnit = period !== '24h';
      const radiationUnit = isEnergyUnit ? 'kWh/mm²' : 'W/m²';
      const radiationHoverFormat = isEnergyUnit ? '.2f' : '.1f';
      const evaporationUnit = period === '24h' ? 'mm/day' : 'mm';
      const baseLayout = {
        xaxis: { ...tickSettings[period], type: 'date', title: 'Дата/час', automargin: true },
        margin: { l: 80, r: 80, t: 40, b: 80 },
        legend: { orientation: 'h', y: -0.3 },
        hoverlabel: { namelength: -1 },
        showlegend: true
      };
      if (x.length) {
        baseLayout.xaxis.range = [x[0], x[x.length - 1]];
      }
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
              line: { shape: 'spline', color: 'green' },
              hovertemplate: '%{fullData.name}: %{y:.1f} hPa<extra></extra>'
            },
            {
              x,
              y: data.P_REL,
              name: 'Налягане - относително [hPa]',
              type: 'scatter',
              line: { shape: 'spline', color: 'purple' },
              hovertemplate: '%{fullData.name}: %{y:.1f} hPa<extra></extra>'
            }
          ],
          layout: {
            title: 'Налягане',
            yaxis: {
              tickformat: '.1f',
              hoverformat: '.1f',
              automargin: true,
              color: '#333',
              linecolor: '#333'
            }
          }
        },
        {
          id: 'graph-wind',
          data: [
            {
              x,
              y: data.WIND_SPEED_1,
              name: 'Скорост на вятъра [km/h]',
              type: 'scatter',
              yaxis: 'y1',
              line: { shape: 'spline', color: 'orange' },
              hovertemplate: '%{fullData.name}: %{y:.1f} km/h<extra></extra>'
            },
            {
              x,
              y: data.WIND_DIR,
              name: 'Посока на вятъра [°]',
              type: 'scatter',
              yaxis: 'y2',
              line: { shape: 'spline', color: 'teal' },
              hovertemplate: '%{fullData.name}: %{y:.0f}°<extra></extra>'
            }
          ],
          layout: {
            title: 'Вятър',
            yaxis: { tickformat: '.1f', hoverformat: '.1f', automargin: true, color: 'orange', linecolor: 'orange' },
            yaxis2: {
              overlaying: 'y',
              side: 'right',
              tickformat: '.0f',
              hoverformat: '.0f',
              showline: true,
              automargin: true,
              color: 'teal',
              linecolor: 'teal',
              range: [0, 360]
            }
          }
        },
        {
          id: 'graph-rain',
          data: [
            {
              x,
              y: data.RAIN,
              name: 'Валежи [mm]',
              type: 'bar',
              marker: { color: 'blue' },
              hovertemplate: '%{fullData.name}: %{y:.2f} mm<extra></extra>'
            }
          ],
          layout: {
            title: 'Валежи',
            yaxis: { tickformat: '.2f', hoverformat: '.2f', automargin: true, color: 'blue', linecolor: 'blue' }
          }
        },
        {
          id: 'graph-evaporation',
          data: [
            {
              x,
              y: data.EVAPOR_MINUTE,
              name: `Изпарение [${evaporationUnit}]`,
              type: 'bar',
              marker: { color: 'green' },
              hovertemplate: `%{fullData.name}: %{y:.1f} ${evaporationUnit}<extra></extra>`
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
              name: `Слънчева радиация [${radiationUnit}]`,
              type: 'bar',
              marker: { color: 'orange' },
              hovertemplate: `%{fullData.name}: %{y:${radiationHoverFormat}} ${radiationUnit}<extra></extra>`
            }
          ],
          layout: {
            title: 'Слънчева радиация',
            yaxis: { tickformat: radiationHoverFormat, hoverformat: radiationHoverFormat, automargin: true, color: 'orange', linecolor: 'orange' }
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
  setInterval(renderGraphs, 60000);
});
