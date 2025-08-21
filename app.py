import subprocess
import sys
from datetime import datetime, timedelta

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from waitress import serve

import mean_1h

from flask_bcrypt import Bcrypt
import logging
import os
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    session,
    redirect,
    url_for,
    flash,
    make_response,
    got_request_exception,
)
import pymysql
import configparser
import openpyxl
from functools import wraps

import insertMissingDataFromCSV
from logging import FileHandler, WARNING
import threading
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create file handler which logs only errors
file_handler = logging.FileHandler('errorlog.log')
file_handler.setLevel(logging.ERROR)

# Create formatter and add it to the file handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)
logger.addHandler(file_handler)

# Load configuration from config.ini
with open('config.ini', 'r', encoding='utf-8') as config_file:
    config = configparser.ConfigParser(interpolation=None)
    config.read_file(config_file)

# MySQL Database Configuration
DB_HOST = config.get('SQL', 'host')
DB_PORT = config.getint('SQL', 'port')
DB_USER = config.get('SQL', 'user')
DB_PASSWORD = config.get('SQL', 'password')
DB_NAME = config.get('SQL', 'database')
DB_TABLE = config.get('SQL', 'mean_1hour_table')
DB_TABLE_MIN = config.get('SQL', 'DB_TABLE_MIN')

# Columns
DATE_COLUMN = 'DateRef'
RAW_DATA_COLUMNS = [col.strip() for col in config.get('SQL', 'columns_list').split(',')]
CALCULATED_COLUMNS = [col.strip() for col in config.get('SQL', 'calc_columns').split(',')]
DATA_COLUMNS = RAW_DATA_COLUMNS + CALCULATED_COLUMNS
RAW_UNITS = [col.strip() for col in config.get('SQL', 'columns_units_list').split(',')]
CALCULATED_UNITS = [col.strip() for col in config.get('SQL', 'calc_units_list').split(',')]
DATA_COLUMNS_UNITS = RAW_UNITS + CALCULATED_UNITS
# Pollutant-specific configuration removed; initialize empty lists for compatibility
POLLUTANT_COLUMNS = []
nde_all = []
pollutants_units = []
simple_parameters = []
simple_parameters_units = []
simple_parameters_BG = []


# Names
DATA_COLUMNS_NAMES = DATA_COLUMNS
# Excel template file path
EXCEL_TEMPLATE_PATH = 'template.xlsx'
RAW_BG = [col.strip() for col in config.get('SQL', 'DATA_COLUMNS_BG').split(',')]
CALCULATED_BG = [col.strip() for col in config.get('SQL', 'calc_columns_bg').split(',')]
DATA_COLUMNS_BG = RAW_BG + CALCULATED_BG
# Variable to store the path of the saved plot image
plot_image_path = 'plot.png'

global my_df
my_df = pd.DataFrame(columns=[DATE_COLUMN] + DATA_COLUMNS)
# Global dataframe to store the required data
df_last_min_values = pd.DataFrame(columns=[DATE_COLUMN] + DATA_COLUMNS)
def add_calculated_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    if 'RAIN_MINUTE' in df.columns:
        df['RAIN_MINUTE'] = pd.to_numeric(df['RAIN_MINUTE'], errors='coerce')
        df['RAIN_DAY'] = df.groupby(df.index.date)['RAIN_MINUTE'].cumsum()
        df['RAIN_MONTH'] = df.groupby(df.index.to_period('M'))['RAIN_MINUTE'].cumsum()
        df['RAIN_YEAR'] = df.groupby(df.index.year)['RAIN_MINUTE'].cumsum()
    if 'EVAPOR_MINUTE' in df.columns:
        df['EVAPOR_MINUTE'] = pd.to_numeric(df['EVAPOR_MINUTE'], errors='coerce')
        df['EVAPOR_DAY'] = df.groupby(df.index.date)['EVAPOR_MINUTE'].cumsum()
    return df
# Function to establish a connection to the MySQL database
def get_db_connection():
    return pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME, port=DB_PORT)

#INitialize plot

STATIC_FOLDER = os.path.join(current_dir, 'static')
#TEMPLATES_FOLDER = os.path.join(current_dir, 'templates')
template_dir = os.path.abspath('templates')
app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=template_dir)
#app = Flask(__name__, template_folder=template_dir)
#app = Flask(__name__)
bcrypt = Bcrypt(app)
app.secret_key = 'джасфжфдгсдфг'  # Replace with a secure random key
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_COOKIE_SECURE'] = True  # Use secure cookies (recommended for production)
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent client-side script access to the cookie
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Prevent cross-site requests

# Dummy user store for demonstration
users = {
    'admin': bcrypt.generate_password_hash('password123').decode('utf-8')
}

# Timezone setup (optional, adjust if needed)
TZ = pytz.timezone("Europe/Sofia")  # Adjust timezone as needed

init = 0

def update_dataframes():
    global df_last_min_values

    while True:
        try:
            db_connection = get_db_connection()
            try:
                cursor = db_connection.cursor()
                query_last_minute = f"""
                    SELECT {DATE_COLUMN}, {', '.join(RAW_DATA_COLUMNS)}
                    FROM {DB_TABLE_MIN}
                    ORDER BY {DATE_COLUMN} DESC LIMIT 1
                """
                cursor.execute(query_last_minute)
                last_minute_data = cursor.fetchall()

                df_last_min_data = pd.DataFrame(
                    last_minute_data,
                    columns=[DATE_COLUMN] + RAW_DATA_COLUMNS
                )
                if not df_last_min_data.empty:
                    df_last_min_data[RAW_DATA_COLUMNS] = df_last_min_data[RAW_DATA_COLUMNS].apply(
                        pd.to_numeric, errors='coerce'
                    )
                    df_last_min_data[DATE_COLUMN] = pd.to_datetime(df_last_min_data[DATE_COLUMN])
                    df_last_min_data.set_index(DATE_COLUMN, inplace=True)
                    df_last_min_data = add_calculated_columns(df_last_min_data)

                if not df_last_min_data.empty:
                    df_last_min_values = df_last_min_data.reindex(columns=DATA_COLUMNS)
                    df_last_min_values.reset_index(inplace=True)
                    df_last_min_values = df_last_min_values.round(1)
            finally:
                cursor.close()
                db_connection.close()
        except Exception as e:
            logger.error(f"Error updating dataframes: {e}", exc_info=True)

        time.sleep(30)


# Start the update thread
#update_dataframes()
threading.Thread(target=update_dataframes, daemon=True).start()


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and bcrypt.check_password_hash(users[username], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return render_template('index.html')
        else:
            error_message = 'Невалидно потребителско име или парола.'
            return render_template('login.html', error_message=error_message)

    return render_template('login.html')


@app.route('/logout', methods=['GET'])
def logout():
    session.clear()  # Clear session data
    flash('You have been logged out.', 'info')  # Optional: Show a message
    return redirect(url_for('login'))  # Redirect to the login page


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('You need to log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# Route to serve the HTML webpage
@app.route('/')
@login_required
def index():
    if 'username' not in session:  # Check if user is logged in
        return redirect(url_for('login'))  # Redirect to login if not authenticated
    return render_template('index.html')  # Serve the main page


@app.route('/graphs')
@login_required
def graphs_page():
    return render_template('graphs.html')


@app.route('/graph_data', methods=['GET'])
@login_required
def graph_data():
    period = request.args.get('period', '24h')
    try:
        db_connection = get_db_connection()
        cursor = db_connection.cursor()
        end_time = datetime.now()

        if period == '24h':
            start_time = end_time - timedelta(hours=24)
            table = DB_TABLE_MIN
        elif period == '30d':
            start_time = end_time - timedelta(days=30)
            table = DB_TABLE
        else:
            start_time = end_time - timedelta(days=365)
            table = DB_TABLE

        query = (
            f"SELECT {DATE_COLUMN}, {', '.join(RAW_DATA_COLUMNS)} FROM {table} "
            f"WHERE {DATE_COLUMN} >= %s AND {DATE_COLUMN} <= %s ORDER BY {DATE_COLUMN} ASC"
        )
        cursor.execute(
            query,
            (
                start_time.strftime('%Y-%m-%d %H:%M:%S'),
                end_time.strftime('%Y-%m-%d %H:%M:%S'),
            ),
        )
        data = cursor.fetchall()
        cursor.close()
        db_connection.close()

        df = pd.DataFrame(data, columns=[DATE_COLUMN] + RAW_DATA_COLUMNS)
        if df.empty:
            return jsonify({})

        df[RAW_DATA_COLUMNS] = df[RAW_DATA_COLUMNS].apply(pd.to_numeric, errors='coerce')
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df.set_index(DATE_COLUMN, inplace=True)

        if period == '24h':
            df_res = df.resample('h').mean()
        else:
            df_others = df.drop(columns=['RADIATION']).resample('d').mean()
            rad = df['RADIATION'].resample('d').sum() / 1000
            df_res = df_others.join(rad.rename('RADIATION'))

        df_res = df_res.dropna(how='all')
        df_res.reset_index(inplace=True)
        # Replace NaN values with None to ensure valid JSON serialization
        df_res = df_res.where(pd.notnull(df_res), None)

        result = {col: df_res[col].tolist() for col in df_res.columns}
        result[DATE_COLUMN] = [ts.isoformat() for ts in result[DATE_COLUMN]]
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /graph_data endpoint: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/statistics')
@login_required
def statistics_page():
    return render_template('statistics.html')


def _format_dt(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%d %H:%M')


def _min_max_with_time(df: pd.DataFrame, column: str):
    series = df[column].dropna()
    if series.empty:
        return None
    min_time = series.idxmin()
    max_time = series.idxmax()
    return {
        'min': float(series.loc[min_time]),
        'min_time': _format_dt(min_time),
        'max': float(series.loc[max_time]),
        'max_time': _format_dt(max_time),
    }


def _build_stats(period: str):
    now = datetime.now()
    if period == 'today':
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif period == 'month':
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = (start + timedelta(days=32)).replace(day=1)
    elif period == 'year':
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(year=start.year + 1)
    else:
        start = None
        end = None

    try:
        db_connection = get_db_connection()
        cursor = db_connection.cursor()
        query = (
            f"SELECT {DATE_COLUMN}, T_AIR, REL_HUM, P_REL, WIND_GUST, WIND_DIR, RAIN_MINUTE, EVAPOR_MINUTE, RADIATION "
            f"FROM {DB_TABLE_MIN}"
        )
        params = None
        if start and end:
            query += f" WHERE {DATE_COLUMN} >= %s AND {DATE_COLUMN} < %s"
            params = (
                start.strftime('%Y-%m-%d %H:%M:%S'),
                end.strftime('%Y-%m-%d %H:%M:%S'),
            )
        cursor.execute(query, params) if params else cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        db_connection.close()
        df = pd.DataFrame(
            data,
            columns=[
                DATE_COLUMN,
                "T_AIR",
                "REL_HUM",
                "P_REL",
                "WIND_GUST",
                "WIND_DIR",
                "RAIN_MINUTE",
                "EVAPOR_MINUTE",
                "RADIATION",
            ],
        )
    except Exception as e:
        logger.error(f"Database error while building stats: {e}", exc_info=True)
        return []

    if df.empty:
        return []

    for col in ['T_AIR', 'REL_HUM', 'P_REL', 'WIND_GUST', 'WIND_DIR', 'RAIN_MINUTE', 'EVAPOR_MINUTE', 'RADIATION']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df.set_index(DATE_COLUMN, inplace=True)

    result = []
    temp = _min_max_with_time(df, 'T_AIR')
    if temp:
        result.append(
            f"Температура: мин {temp['min']:.1f}°C ({temp['min_time']}), "
            f"макс {temp['max']:.1f}°C ({temp['max_time']})"
        )

    hum = _min_max_with_time(df, 'REL_HUM')
    if hum:
        result.append(
            f"Относителна влажност: мин {hum['min']:.1f}% ({hum['min_time']}), "
            f"макс {hum['max']:.1f}% ({hum['max_time']})"
        )

    press = _min_max_with_time(df, 'P_REL')
    if press:
        result.append(
            f"Атмосферно налягане: мин {press['min']:.1f} hPa ({press['min_time']}), "
            f"макс {press['max']:.1f} hPa ({press['max_time']})"
        )

    gust_series = df['WIND_GUST'].dropna()
    if not gust_series.empty:
        gust_time = gust_series.idxmax()
        gust_value = float(gust_series.loc[gust_time])
        direction = df.loc[gust_time, 'WIND_DIR'] if 'WIND_DIR' in df.columns else None
        dir_text = f", посока {direction}" if pd.notnull(direction) else ''
        result.append(
            f"Порив на вятъра: макс {gust_value:.1f} km/h{dir_text} ({_format_dt(gust_time)})"
        )

    rain_total = df['RAIN_MINUTE'].dropna().sum()
    if rain_total and period == 'today':
        result.append(f"Сума дъжд за деня: {rain_total:.1f} mm")
    elif rain_total:
        result.append(f"Сума дъжд: {rain_total:.1f} mm")

    if period == 'today':
        evap_total = df['EVAPOR_MINUTE'].dropna().sum()
        if evap_total:
            result.append(f"Изпарение за деня: {evap_total:.1f} mm")

    if period != 'today' and not df['RAIN_MINUTE'].dropna().empty:
        daily_rain = df['RAIN_MINUTE'].resample('D').sum()
        max_day = daily_rain.max()
        max_day_time = _format_dt(daily_rain.idxmax())
        result.append(f"Макс за ден: {max_day:.1f} mm ({max_day_time})")
        intensity_series = df['RAIN_MINUTE'].dropna()
        intensity_time = intensity_series.idxmax()
        result.append(
            f"Макс интензитет: {float(intensity_series.loc[intensity_time]):.1f} mm/min "
            f"({_format_dt(intensity_time)})"
        )

    rad_series = df['RADIATION'].dropna()
    if not rad_series.empty:
        rad_time = rad_series.idxmax()
        rad_max = float(rad_series.loc[rad_time])
        rad_sum = float(rad_series.sum())
        result.append(
            f"Глобална радиация: макс {rad_max:.1f} W/m² ({_format_dt(rad_time)})"
        )
        result.append(f"Сума глобална радиация: {rad_sum:.1f} W/m²")

    return result


@app.route('/statistics_data')
@login_required
def statistics_data():
    try:
        data = {
            'today': _build_stats('today'),
            'month': _build_stats('month'),
            'year': _build_stats('year'),
            'all': _build_stats('all'),
        }
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in /statistics_data endpoint: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/report')
@login_required
def report_page():
    return render_template('report.html')


@app.route('/plot', methods=['POST'])
def plot():
    global my_df
    try:
        # Get start and end dates from the frontend
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        category = request.form['category']

        # Add 1 hour to end_date
        end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

        # Connect to the database
        db_connection = get_db_connection()

        # Fetch data from the database based on the date range
        cursor = db_connection.cursor()
        query = f"SELECT {DATE_COLUMN}, {', '.join(RAW_DATA_COLUMNS)} FROM {DB_TABLE} " \
                f"WHERE {DATE_COLUMN} >= %s AND {DATE_COLUMN} <= %s ORDER BY {DATE_COLUMN} ASC "
        cursor.execute(query, (start_date+' 00:00:00', end_date.strftime('%Y-%m-%d %H:%M:%S')))
        data = cursor.fetchall()

        # Close database connection
        cursor.close()
        db_connection.close()

        # Create a DataFrame from the fetched data with only DATE_COLUMN and consumption columns
        df = pd.DataFrame(data, columns=[DATE_COLUMN] + RAW_DATA_COLUMNS)

        # Convert DATE_COLUMN to datetime
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

        # Set DATE_COLUMN as index for proper time-based grouping
        df.set_index(DATE_COLUMN, inplace=True)
        df = add_calculated_columns(df)

        # Group data by category (hourly, daily, or monthly)
        if category == "hourly":
            df_grouped = df.groupby(pd.Grouper(freq='h')).mean()
        elif category == "daily":
            df_grouped = df.groupby(pd.Grouper(freq='D')).mean()
        elif category == "monthly":
            df_grouped = df.groupby(pd.Grouper(freq='M')).mean()
        df_grouped = add_calculated_columns(df_grouped)
        df_grouped.reset_index(inplace=True)
        df_grouped.rename(columns={'index': DATE_COLUMN}, inplace=True)

        my_df = df_grouped
        columns_to_plot = [col for col in DATA_COLUMNS if col in my_df.columns]

        # Create Plotly Express figure for consumption
        fig1 = px.line(my_df, x=DATE_COLUMN, y=columns_to_plot, title="Всички данни")

        # Update axis titles
        fig1.update_xaxes(title_text="Дата")

        # Use the same title layout as fig3
        fig1.update_layout(
            hovermode="x unified",
            height=600,  # Adjust the height
            legend_title_text="Данни",
            title={
                'text': "Всички данни",
                'x': 0.5,  # Center the title
                'xanchor': 'center',  # Anchor the title at the center
                'yanchor': 'top'  # Anchor the title at the top
            },
            autosize=True  # Ensure the figure resizes dynamically

        )

        # Update legend names using Bulgarian column names
        bg_names_for_plot = [DATA_COLUMNS_BG[DATA_COLUMNS.index(c)] for c in columns_to_plot]
        for trace, data_col, data_col_name in zip(fig1.data, columns_to_plot, bg_names_for_plot):
            trace.name = data_col_name  # Set the trace name
            trace.hovertemplate = None  # Remove the default hover template

        # Update trace mode to markers+lines
        fig1.update_traces(mode="markers+lines", hovertemplate=None)

        # Wind Rose Plot for wind direction and speed
        wind_angle = my_df['WIND_DIR']
        wind_speed = my_df['WIND_SPEED_1']


        # Example wind speed and direction data
        wind_rose_fig = go.Figure(go.Scatterpolar(
            r=wind_speed,
            theta=wind_angle,
            mode='markers',
            marker=dict(size=8, color=wind_speed, colorscale='Viridis', showscale=True),
            name="Wind Rose"
        ))

        wind_rose_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True),
                angularaxis=dict(
                                rotation=90,  # Start from 0° at the top (North)
                                direction="clockwise",
                                tickmode='array',
                                 tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                 ticktext=['North', 'N-E', 'East', 'S-E', 'South', 'S-W', 'West', 'N-W'])
            ),
            title={
                'text': "Роза на ветровете",
                'x': 0.5,  # Center the title
                'xanchor': 'center',  # Anchor the title at the center
            },
            # height=600,  # Adjust the height
            # width=800,# Adjust the width
            autosize=True
        )
        # Convert plots to JSON format
        plot_json1 = fig1.to_json()
        wind_rose_json = wind_rose_fig.to_json()

        # Return plots in a JSON response
        return jsonify({
            'plot1': plot_json1,
            'wind_rose': wind_rose_json
        })
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return str(e), 500


@app.route('/export_to_excel', methods=['POST'])
def export_to_excel():
    global my_df
    try:
        # Parse JSON data
        data = request.get_json()
        start_date = pd.to_datetime(data['start_date'])
        end_date = pd.to_datetime(data['end_date'])
        selected_traces = data.get('selected_traces', [])

        # Create translation dictionaries from lists
        column_map_latin_to_bg = dict(zip(DATA_COLUMNS, DATA_COLUMNS_BG))
        column_map_bg_to_latin = dict(zip(DATA_COLUMNS_BG, DATA_COLUMNS))

        # Translate selected traces from Bulgarian to Latin
        selected_traces_latin = [column_map_bg_to_latin.get(trace, trace) for trace in selected_traces]

        # Filter DataFrame
        if selected_traces_latin:
            filtered_df = my_df[['DateRef'] + selected_traces_latin]
        else:
            filtered_df = my_df

        # Translate column names to Bulgarian for export
        selected_columns_bg = ['Дата'] + [column_map_latin_to_bg.get(col, col) for col in selected_traces_latin]

        # Load Excel template
        wb = openpyxl.load_workbook(EXCEL_TEMPLATE_PATH)
        ws = wb.active

        # Write metadata
        ws['E8'] = start_date.strftime('%d-%m-%Y')
        ws['E9'] = end_date.strftime('%d-%m-%Y')

        # Write headers
        for col_idx, col_name in enumerate(selected_columns_bg, start=2):
            ws.cell(row=11, column=col_idx, value=col_name)

        # Write data
        for row_idx, (_, row) in enumerate(filtered_df.iterrows(), start=12):
            for col_idx, value in enumerate(row, start=2):
                ws.cell(row=row_idx, column=col_idx, value=value)

        # Save and send file
        file_path = os.path.abspath('output.xlsx')
        wb.save(file_path)
        return send_file(
            file_path,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='output.xlsx'
        )
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "Failed to export data"}), 500


@app.route('/export2_to_excel', methods=['POST'])
def export2_to_excel():
    global my_df
    try:
        # Parse JSON data
        data = request.get_json()
        start_date = pd.to_datetime(data['start_date'])
        end_date = pd.to_datetime(data['end_date'])
        selected_traces = data.get('selected_traces', [])

        # Create translation dictionaries from lists
        column_map_latin_to_bg = dict(zip(DATA_COLUMNS, DATA_COLUMNS_BG))
        column_map_bg_to_latin = dict(zip(DATA_COLUMNS_BG, DATA_COLUMNS))

        # Translate selected traces from Bulgarian to Latin
        selected_traces_latin = [column_map_bg_to_latin.get(trace, trace) for trace in selected_traces]

        # Filter DataFrame
        if selected_traces_latin:
            filtered_df = my_df[['DateRef'] + selected_traces_latin]
        else:
            filtered_df = my_df

        # Translate column names to Bulgarian for export
        selected_columns_bg = ['Дата'] + [column_map_latin_to_bg.get(col, col) for col in selected_traces_latin]

        # Load Excel template
        wb = openpyxl.load_workbook(EXCEL_TEMPLATE_PATH)
        ws = wb.active

        # Write metadata
        ws['E8'] = start_date.strftime('%Y-%m-%d')
        ws['E9'] = end_date.strftime('%Y-%m-%d')

        # Write headers
        for col_idx, col_name in enumerate(selected_columns_bg, start=2):
            ws.cell(row=11, column=col_idx, value=col_name)

        # Write data
        for row_idx, (_, row) in enumerate(filtered_df.iterrows(), start=12):
            for col_idx, value in enumerate(row, start=2):
                ws.cell(row=row_idx, column=col_idx, value=value)

        # Save and send file
        file_path = os.path.abspath('output.xlsx')
        wb.save(file_path)
        return send_file(
            file_path,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='output.xlsx'
        )
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "Failed to export data"}), 500


def run_insert_missing_data():
    try:
        insertMissingDataFromCSV.main()
    except Exception as e:
        logger.error(f"Error updating data from CSV: {e}")


@app.route('/insert_missing_data', methods=['POST'])
def insert_missing_data():
    try:
        run_insert_missing_data()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/moment_data', methods=['GET'])
def moment_data():
    global df_last_min_values

    def format_date(record):
        try:
            if 'DateRef' in record and record['DateRef']:
                if isinstance(record['DateRef'], pd.Timestamp):
                    record['DateRef'] = record['DateRef'].strftime('%d.%m.%Y %H:%M')
                else:
                    record['DateRef'] = datetime.strptime(record['DateRef'], '%a, %d %b %Y %H:%M:%S %Z').strftime('%d.%m.%Y %H:%M')
        except Exception as e:
            print(f"Error formatting date: {e}")

    def prepare_data(df):
        return df.reset_index(drop=True).to_dict('records') if not df.empty else []

    try:
        update_type = request.args.get("update_type", "last_minute")
        if update_type != "last_minute":
            return jsonify({"error": "Invalid update_type"}), 400

        df_last_min = df_last_min_values.copy()
        df_last_min.fillna(0, inplace=True)
        df_last_min.round(1)

        min_values_data = prepare_data(df_last_min)
        for record in min_values_data:
            format_date(record)

        return jsonify({
            "min_values_data": min_values_data,
            "columns_values": DATA_COLUMNS,
            "columns_bg": DATA_COLUMNS_BG,
            "columns_units": DATA_COLUMNS_UNITS,
        })

    except Exception as e:
        print(f"Error in /moment_data endpoint: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Initialize BackgroundScheduler
scheduler = BackgroundScheduler()

# Job to compute hourly averages
def run_hourly_mean():
    now = datetime.now()
    mean_1h.mean_1h(now, now)

# Function to start the scheduler
def start_scheduler():
    scheduler.add_job(run_insert_missing_data, 'cron', hour='*', minute='*', second='1')
    scheduler.add_job(run_hourly_mean, 'cron', minute='0', second='30')
    scheduler.start()
    print("Scheduler started in a separate thread...")

# Start scheduler in a separate thread
threading.Thread(target=start_scheduler, daemon=True).start()

# Stop the scheduler gracefully when the app stops
def shutdown_scheduler(sender, **extra):
    if scheduler.running:
        scheduler.shutdown()
        print("Scheduler has been shut down gracefully.")

# Attach the shutdown function to the 'got_request_exception' signal
got_request_exception.connect(shutdown_scheduler, app)

if init == 0:
    run_insert_missing_data()

if __name__ == '__main__':
    #insertMissingDataFromCSV.main()
    app.run(host='0.0.0.0', port=5010, debug=False)
    # uncomment this to start in non production
    #serve(app, host='0.0.0.0', port=50023)

