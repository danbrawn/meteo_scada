$(document).ready(function() {
  function buildTable() {
    const days = Array.from({length: 31}, (_, i) => i + 1);
    const params = [
      {name:'Температура °C'},
      {name:'Относителна влажност %'},
      {name:'Относително налягане hPa'},
      {name:'Скорост на вятъра km/h'},
      {name:'Скорост на вятъра m/s'},
      {name:'Посока на вятъра (DEG)'},
      {name:'Валежи (l/m2)'},
      {name:'Температура 14:00 °C'},
      {name:'Отн. влажност 14:00 %'},
      {name:'Отн. налягане 14:00 hPa'},
      {name:'Изпарение mm/d'}
    ];
    let thead = '<tr><th>Параметър</th>' + days.map(d=>`<th>${d}</th>`).join('') + '</tr>';
    $('#report-table thead').html(thead);
    let rows = params.map(p => '<tr><td>'+p.name+'</td>' + days.map(()=>'<td></td>').join('') + '</tr>').join('');
    $('#report-table tbody').html(rows);
  }
  buildTable();
});
