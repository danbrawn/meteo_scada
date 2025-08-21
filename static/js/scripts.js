$(document).ready(function () {
    function equalizeTiles() {
        const tiles = document.querySelectorAll('.dashboard-item');
        if (!tiles.length) return;
        tiles.forEach(tile => tile.style.height = 'auto');
        let minHeight = Infinity;
        tiles.forEach(tile => {
            const h = tile.offsetHeight;
            if (h < minHeight) minHeight = h;
        });
        tiles.forEach(tile => tile.style.height = `${minHeight}px`);
    }

    function renderDashboard(data, columnOrder, columnUnits, columnNamesBG) {
        const container = $('#last-min-values-dashboard');
        container.empty();

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
        };

        const columnGroups = {
            'Общи параметри на въздуха': ['T_AIR', 'T_INSIDE', 'REL_HUM', 'T_WATER'],
            'Параметри на радиация': ['RADIATION'],
            'Изпарение': ['EVAPOR_MINUTE', 'EVAPOR_DAY'],
            'Параметри на вятъра': ['WIND_SPEED_1', 'WIND_SPEED_2', 'WIND_DIR', 'WIND_GUST'],
            'Атмосферно налягане': ['P_ABS', 'P_REL'],
            'Статистика за валежи': ['RAIN_MINUTE', 'RAIN_HOUR', 'RAIN_DAY', 'RAIN_MONTH', 'RAIN_YEAR']
        };

        const timestamp = data['DateRef'];
        if (timestamp) {
            const formatted = timestamp.slice(0, 16);
            $('#dashboard-title').text(`Данни за ${formatted}`);
        }

        Object.entries(columnGroups).forEach(([groupName, keys]) => {
            const groupDiv = $('<div>').addClass('dashboard-group');
            groupDiv.append(`<h3>${groupName}</h3>`);
            const groupGrid = $('<div>').addClass('dashboard-grid');

            keys.forEach(key => {
                if (!(key in data)) return;

                const displayName = columnNamesBG[columnOrder.indexOf(key)] || key;
                const unit = columnUnits[columnOrder.indexOf(key)] || '';

                let style = 'background-color: MediumSeaGreen; color: black;';

                const value = data[key];
                let maxThreshold = '';

                if (!isNaN(value) && thresholds[key]) {
                    const threshold = thresholds[key].find(t => value <= t.max);
                    if (threshold) {
                        style = `background-color: ${threshold.color}; color: ${threshold.textColor};`;
                        maxThreshold = thresholds[key][1]?.max;
                    }
                }

                const dashboardItem = `
                    <div class="dashboard-item" style="${style}">
                        <div class="variable-name">${displayName} ${unit ? `[${unit}]` : ''}</div>
                        <div class="variable-value">${value}</div>
                        ${maxThreshold ? `<div class="variable-threshold">ПДК: ${maxThreshold}</div>` : ''}
                    </div>
                `;
                groupGrid.append(dashboardItem);
            });

            groupDiv.append(groupGrid);
            container.append(groupDiv);
        });

        equalizeTiles();

    }

    function fetchMomentData() {
        $.ajax({
            type: 'GET',
            url: '/moment_data?update_type=last_minute',
            cache: false,
            success: function (response) {
                if (response.error) {
                    console.error("Error in API response:", response.error);
                    return;
                }
                if (response.min_values_data && response.min_values_data.length > 0) {
                    renderDashboard(
                        response.min_values_data[0],
                        response.columns_values,
                        response.columns_units,
                        response.columns_bg
                    );
                }
            },
            error: function (err) {
                console.error("Error in fetchMomentData:", err);
            }
        });
    }

    fetchMomentData();
    setInterval(fetchMomentData, 60000);
});

