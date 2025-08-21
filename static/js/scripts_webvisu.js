$(document).ready(function () {
    // Initially scale the moment-data-container
//    $('#moment-data-container').css({
//        transform: 'scale(0.8)',
//        transformOrigin: 'top left'
//    });
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
});
