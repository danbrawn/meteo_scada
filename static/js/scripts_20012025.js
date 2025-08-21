$(document).ready(function () {
    // Toggle containers when the "Моментни данни" / "Архивни данни" button is clicked
    $('#moment-data-btn').click(function () {
        $('#moment-data-container').toggleClass('hidden');
        $('#archive-data-container').toggleClass('hidden');

        // Toggle button text based on the active container
        const isMomentDataVisible = !$('#moment-data-container').hasClass('hidden');
        $('#moment-data-btn').text(isMomentDataVisible ? 'Архивни данни' : 'Моментни данни');
    });

    // Initialize datepickers for the start and end date fields
    $('#from, #to').datepicker({
        dateFormat: 'yy-mm-dd',
        changeMonth: true,
        changeYear: true
    });

    // Handle the plot button click to update all plots
     $('#plot-btn').click(function () {
        const startDate = $('#from').val();
        const endDate = $('#to').val();
        const category = $('#category').val();

        console.log("Start Date:", startDate);
        console.log("End Date:", endDate);
        console.log("Category:", category);

        $.ajax({
            type: 'POST',
            url: '/plot',
            data: { start_date: startDate, end_date: endDate, category: category },
            success: function (response) {
                // Parse the JSON response for each plot
                var plot1Data = JSON.parse(response.plot1);
                var windRoseData = JSON.parse(response.wind_rose);
                var plot3Data = JSON.parse(response.plot3);

                // Array of container IDs and corresponding data
                var plotContainers = [
                    { id: 'plot-container-1', data: plot1Data },
                    { id: 'plot-container-2', data: windRoseData },
                    { id: 'plot-container-3', data: plot3Data }
                ];

                plotContainers.forEach(plot => {
                    Plotly.purge(plot.id); // Clear any existing plot
                    Plotly.newPlot(plot.id, plot.data.data, plot.data.layout, { responsive: true });

                    // Adjust layout margins
                    var updatedLayout = Object.assign({}, plot.data.layout, {
                        margin: { l: 80, r: 80, t: 40, b: 40 }
                    });
                    Plotly.relayout(plot.id, updatedLayout);
                });
            },
            error: function (err) {
                console.error("Error making request:", err);
            }
        });
     });


    // Handle the export button click to export selected plot to Excel
    $('#export-btn').click(function () {
          var startDate = document.getElementById('from').value;
          var endDate = document.getElementById('to').value;

          // Ensure the date fields are filled
          if (!startDate || !endDate) {
              alert('Please select a valid date range.');
              return;
          }

          var visibleTraces = [];
          var plotData = document.getElementById('plot-container-1').data;

          plotData.forEach(function(trace) {
              // Check if the trace is visible
              if (trace.visible === true || trace.visible === undefined) {
                  visibleTraces.push(trace.name); // Ensure `trace.name` corresponds to column names
              }
          });

          // Send data to backend
          $.ajax({
              type: 'POST',
              url: '/export_to_excel',
              contentType: 'application/json',
              data: JSON.stringify({
                  start_date: startDate,
                  end_date: endDate,
                  selected_traces: visibleTraces
              }),
              xhrFields: { responseType: 'blob' },
              success: function(response) {
                  // Handle file download
                  var blob = new Blob([response], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
                  var fileName = `Bobovdol_AMS_${startDate}_${endDate}.xlsx`;
                  var link = document.createElement('a');
                  link.href = window.URL.createObjectURL(blob);
                  link.download = fileName;
                  link.click();
              },
              error: function(xhr, status, error) {
                  console.error('Export failed:', error);
                  alert('Failed to export data.');
              }
          });
    });
    $('#export2-btn').click(function () {
          var startDate = document.getElementById('from').value;
          var endDate = document.getElementById('to').value;

          // Ensure the date fields are filled
          if (!startDate || !endDate) {
              alert('Please select a valid date range.');
              return;
          }

          var visibleTraces = [];
          var plotData = document.getElementById('plot-container-3').data;

          plotData.forEach(function(trace) {
              // Check if the trace is visible
              if (trace.visible === true || trace.visible === undefined) {
                  visibleTraces.push(trace.name); // Ensure `trace.name` corresponds to column names
              }
          });

          // Send data to backend
          $.ajax({
              type: 'POST',
              url: '/export2_to_excel',
              contentType: 'application/json',
              data: JSON.stringify({
                  start_date: startDate,
                  end_date: endDate,
                  selected_traces: visibleTraces
              }),
              xhrFields: { responseType: 'blob' },
              success: function(response) {
                  // Handle file download
                  var blob = new Blob([response], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
                  var fileName = `meteo_sni_${startDate}_${endDate}.xlsx`;
                  var link = document.createElement('a');
                  link.href = window.URL.createObjectURL(blob);
                  link.download = fileName;
                  link.click();
              },
              error: function(xhr, status, error) {
                  console.error('Export failed:', error);
                  alert('Failed to export data.');
              }
          });
    });

    // Function to update the Last 24 Hours plots
    function updateLast24HoursPlots(plots) {
        const container = $('#last-24-hours-plot');
        container.empty(); // Clear existing content

        // Append each plot HTML
        plots.forEach(plotHtml => {
            const div = $('<div>').html(plotHtml);
            container.append(div);
        });
    }
    function renderDashboard(data, containerId, columnOrder, columnUnits, columnNamesBG) {
        const container = $(`#${containerId}`);
        container.empty(); // Clear existing content

        // Thresholds for individual columns
        const thresholds = {
            "SO2": [
                { max: 300, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 350, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "NO": [
                { max: 180, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 200, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "NO2": [
                { max: 180, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 200, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "NOX": [
                { max: 180, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 200, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "FP1": [
                { max: 40, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 50, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "FP2_5": [
                { max: 40, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 50, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "FP10": [
                { max: 40, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 50, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "SYSTEM_GASANALYSER_FAULT_1": [
                { max: 0.016, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 0.016, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "SYSTEM_GASANALYSER_FAULT_2": [
                { max: 0.016, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 0.016, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "SYSTEM_GASANALYSER_FAULT_3": [
                { max: 0.016, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 0.016, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "SYSTEM_GASANALYSER_FAULT_4": [
                { max: 0.016, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 0.016, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "HIGH_TEMP": [
                { max: 0.016, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 0.016, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "FIRE_ALARM": [
                { max: 0.016, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 0.016, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "DOOR1_OP": [
                { max: 0.016, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 0.016, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "DOOR2_OP": [
                { max: 0.016, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 0.016, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ],
            "POWER_BAD": [
                { max: 0.016, color: 'MediumSeaGreen', textColor: 'black' },
                { max: 0.016, color: 'yellow', textColor: 'black' },
                { max: Infinity, color: 'red', textColor: 'black' }
            ]
            // Add more columns as needed
        };
        // Define excluded columns for specific dashboards
        const excludedColumns = {
            'last-min-values-dashboard': ['RAIN_HOUR', 'RAIN_DAY', 'RAIN_TOTAL'],
            'last-hour-values-dashboard': ['RAIN_MINUTE', 'RAIN_DAY', 'RAIN_TOTAL']
        };
        // Ensure 'DateRef' is always first, followed by the provided column order
        const orderedKeys = ['DateRef', ...columnOrder.filter(col => col !== 'DateRef')];

        orderedKeys.forEach((key) => {
            // Skip excluded columns for the current dashboard
            if (excludedColumns[containerId] && excludedColumns[containerId].includes(key)) {
                return;
            }
            if (key in data) { // Ensure the key exists in the data
                // Select appropriate Bulgarian column names based on containerId
                const displayName =
                    key === 'DateRef'
                        ? 'Дата и час на записа' // Keep DateRef unchanged
                        : columnNamesBG[columnOrder.indexOf(key)] || key; // Use Bulgarian name if available

                // Units logic (skip for specific dashboards or DateRef)
                const unit =
                    containerId === 'last-min-status-dashboard' || key === 'DateRef'
                        ? ''
                        : columnUnits[columnOrder.indexOf(key)] || '';

                // Determine background color based on value and thresholds
                let dashboardItemStyle = 'background-color: Gainsboro; color: black;'; // Default to green if no threshold exists
                const value = data[key];
                let maxThreshold = ''; // To store the max threshold for display

                if (!isNaN(value) && key in thresholds) { // Only apply color logic if the value is numeric and thresholds exist
                    const columnThresholds = thresholds[key];
                    const threshold = columnThresholds.find(t => value <= t.max);
                    if (threshold) {
                        dashboardItemStyle = `background-color: ${threshold.color}; color: ${threshold.textColor};`;

                        // Only set maxThreshold if it's for values dashboard
                        if (containerId === 'last-min-values-dashboard' || containerId === 'last-hour-values-dashboard') {
                            maxThreshold = columnThresholds[1].max; // Get the second threshold for НДЕ
                        }
                    }
                }

                // Dashboard item HTML
                const dashboardItem = `
                    <div class="dashboard-item" style="${dashboardItemStyle}">
                        <div class="variable-name">${displayName} ${unit ? `[${unit}]` : ''}</div> <!-- Line 1 -->
                        <div class="variable-value">${value}</div> <!-- Line 2 -->
                        ${maxThreshold && (containerId === 'last-min-values-dashboard' || containerId === 'last-hour-values-dashboard')
                            ? `<div class="variable-threshold">ПДК: ${maxThreshold}</div>`
                            : ''} <!-- Line 3 (only for last-min-values-dashboard and last-hour-values-dashboard) -->
                    </div>
                `;
                container.append(dashboardItem);
            }
        });
    }


    let isFirstLoad = true; // Flag to track if it's the first data load
    let lastHourUpdated = null; // Tracks the last hour that was updated
    let isFetching = false; // Prevent overlapping fetches
    let fetchInterval;

    function startFetchInterval() {
        if (fetchInterval) {
            clearInterval(fetchInterval);
        }
        fetchInterval = setInterval(() => fetchMomentData('last_minute'), 60000); // Fetch every minute for last_minute data
    }

    function getCurrentHour() {
        const now = new Date();
        return now.getHours();
    }

    function fetchMomentData(updateType) {
        if (!['last_minute', 'hourly'].includes(updateType)) {
            console.error(`Invalid updateType passed: ${updateType}`);
            return;
        }

        if (isFirstLoad && updateType === 'last_minute') {
            $('#loading-message').show();
        }

        $.ajax({
            type: 'GET',
            url: `/moment_data?update_type=${updateType}`, // Pass updateType as query parameter
            cache: false,
            success: function (response) {
                if (isFirstLoad && updateType === 'hourly') {
                    $('#loading-message').hide(); // Hide the loading message
                }

                if (response.error) {
                    console.error("Error in API response:", response.error);
                    return;
                }

                if (updateType === 'last_minute') {
                    // Render Last Minute Status Dashboard
                    if (response.status_data && response.status_data.length > 0) {
                        renderDashboard(
                            response.status_data[0],
                            'last-min-status-dashboard',
                            response.columns_status,
                            [],
                            response.columns_status_bg
                        );
                    }

                    // Render Last Minute Values Dashboard
                    if (response.min_values_data && response.min_values_data.length > 0) {
                        renderDashboard(
                            response.min_values_data[0],
                            'last-min-values-dashboard',
                            response.columns_values,
                            response.columns_units,
                            response.columns_bg
                        );
                    }
                } else if (updateType === 'hourly') {
                    const currentHour = getCurrentHour();

                    if (isFirstLoad || (currentHour !== lastHourUpdated && new Date().getMinutes() >= 3)) {
                        lastHourUpdated = currentHour;

                        if (response.hour_values_data && response.hour_values_data.length > 0) {
                            renderDashboard(
                                //response.hour_values_data[response.hour_values_data.length - 1],
                                response.hour_values_data[0],
                                'last-hour-values-dashboard',
                                response.columns_values,
                                response.columns_units,
                                response.columns_bg
                            );
                        }

                        if (response.plots) {
                            updateLast24HoursPlots(response.plots);
                        }
                    }
                }
            },
            error: function (err) {
                if (isFirstLoad && updateType === 'hourly') {
                    $('#loading-message').hide();
                }
                console.error("Error in fetchMomentData:", err);
            },
            complete: function () {
                if (isFirstLoad && updateType === 'hourly') {
                    isFirstLoad = false;
                    $('#loading-message').hide();
                }
            }
        });
    }

    function fetchAllDataOnLoad() {
        // Fetch last-minute data first
        fetchMomentData('last_minute');

        // Fetch hourly data after a delay to ensure sequential fetching
        setTimeout(() => {
            fetchMomentData('hourly');
        }, 2000); // Adjust delay as needed
    }

    // Schedule hourly updates
    function scheduleHourlyUpdates() {
        // Fetch hourly data on page load
        fetchMomentData('hourly');

        // Calculate time until 3 minutes after the next hour
        const now = new Date();
        const timeUntilNextHour = (60 - now.getMinutes()) * 60000 - now.getSeconds() * 1000;

        setTimeout(() => {
            fetchMomentData('hourly'); // Fetch once 3 minutes after the next hour
            setInterval(() => fetchMomentData('hourly'), 3600000); // Fetch hourly data every hour
        }, timeUntilNextHour + 3 * 60000); // Start 3 minutes after the next hour
    }

    // Initial fetch and interval start
    fetchAllDataOnLoad();
    startFetchInterval();
    scheduleHourlyUpdates();

});