$(document).ready(function () {
    
    function updateLast24HoursPlots(plotsJson) {
        const container = $('#last-24-hours-plot');
        const loadingMessage = $('#loading-message-24');
        const container1 = $('#last-hour-values-plot');
        const loadingMessage1 = $('#loading-message-hour');

        // Show the loading message
        loadingMessage.show();
        container.empty(); // Clear old plots

        // Render each plot from JSON
        setTimeout(() => {
            plotsJson.forEach(plotJson => {
                let plotData = JSON.parse(plotJson); // Parse JSON to get the Plotly figure

                let div = $('<div>'); // Create a div for each plot
                container.append(div);

                Plotly.newPlot(div[0], plotData.data, plotData.layout, {responsive:false, staticPlot: true,}); // Render the plot
            });

            // Hide the loading message once the plots are rendered
            loadingMessage.hide();
            loadingMessage1.hide();
        }, 500); // Small delay for smoother UI
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
        // Validate updateType
        if (!['last_minute', 'hourly'].includes(updateType)) {
            console.error(`Invalid updateType passed: ${updateType}`);
            return;
        }

        // Show loading message on first load for 'last_minute'
        if (isFirstLoad && updateType === 'last_minute') {
            //$('#loading-message-hour').show();
            $('#loading-message-24').show();
        }

        // Perform AJAX GET request
        $.ajax({
            type: 'GET',
            url: `/moment_data?update_type=${updateType}`, // Query parameter for update type
            cache: false,
            success: function (response) {
                // Hide loading message on first load for 'hourly'
                if (isFirstLoad && updateType === 'hourly') {
                    $('#loading-message-24').hide();
                    //$('#loading-message-hour').hide();
                }

                // Handle response errors
                if (response.error) {
                    console.error("Error in API response:", response.error);
                    return;
                }

                // Handle 'last_minute' updates
                if (updateType === 'last_minute') {
                    if (response.status_data && response.status_data.length > 0) {
                        renderDashboard(
                            response.status_data[0],
                            'last-min-status-dashboard',
                            response.columns_status,
                            [],
                            response.columns_status_bg
                        );
                    }

                    if (response.min_values_data && response.min_values_data.length > 0) {
                        renderDashboard(
                            response.min_values_data[0],
                            'last-min-values-dashboard',
                            response.columns_values,
                            response.columns_units,
                            response.columns_bg
                        );
                    }
                }

                // Handle 'hourly' updates
                else if (updateType === 'hourly') {
                    const currentHour = getCurrentHour();

                    // Only update if it's the first load or a new hour has passed
                    if (isFirstLoad || (currentHour !== lastHourUpdated && new Date().getMinutes() >= 3)) {
                        lastHourUpdated = currentHour;

                        if (response.hour_values_data && response.hour_values_data.length > 0) {
                            renderDashboard(
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
                    $('#loading-message-24').hide();
                    //$('#loading-message-hour').hide();
                }
                console.error("Error in fetchMomentData:", err);
            },
            complete: function () {
                // Mark first load as complete
                if (isFirstLoad && updateType === 'hourly') {
                    isFirstLoad = false;
                    $('#loading-message-24').hide();
                    //$('#loading-message-hour').hide();
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