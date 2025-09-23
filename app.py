import subprocess
import sys
from datetime import datetime, timedelta
from decimal import Decimal

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from waitress import serve

import mean_1h

from flask_bcrypt import Bcrypt
import logging
import os
import pandas as pd
import numpy as np
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
from pymysql.cursors import DictCursor
import configparser
import openpyxl
from functools import wraps
from typing import Dict, List

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
# Constants for radiation unit conversion
KWH_PER_M2_FROM_MINUTE = 1 / 60000.0  # Sum of 1-minute W/m² values -> kWh/m²
KWH_PER_M2_FROM_HOUR = 1 / 1000.0  # Sum of 1-hour W/m² values -> kWh/m²
# Variable to store the path of the saved plot image
plot_image_path = 'plot.png'


def _get_config_list(section: str, option: str) -> List[str]:
    if not config.has_section(section) or not config.has_option(section, option):
        return []
    return [item.strip() for item in config.get(section, option).split(',') if item.strip()]


ALARM_TABLE = None
ALARM_DATE_COLUMN = DATE_COLUMN
ALARM_COLUMNS: List[str] = []
ALARM_COLUMN_LABELS: Dict[str, str] = {}
ALARM_COLUMN_DESCRIPTIONS: Dict[str, str] = {}

if config.has_section('ALARMS'):
    ALARM_TABLE = config.get('ALARMS', 'table', fallback=None)
    ALARM_DATE_COLUMN = config.get('ALARMS', 'date_column', fallback=DATE_COLUMN)
    ALARM_COLUMNS = _get_config_list('ALARMS', 'columns')
    alarm_column_names = _get_config_list('ALARMS', 'columns_names')
    alarm_column_descriptions = _get_config_list('ALARMS', 'columns_descriptions')

    for idx, column in enumerate(ALARM_COLUMNS):
        label = alarm_column_names[idx] if idx < len(alarm_column_names) else column
        description = (
            alarm_column_descriptions[idx]
            if idx < len(alarm_column_descriptions)
            else label
        )
        ALARM_COLUMN_LABELS[column] = label
        ALARM_COLUMN_DESCRIPTIONS[column] = description

global my_df
my_df = pd.DataFrame(columns=[DATE_COLUMN] + DATA_COLUMNS)
# Global dataframe to store the required data
df_last_min_values = pd.DataFrame(columns=[DATE_COLUMN] + DATA_COLUMNS)
def add_calculated_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    if 'RAIN_MINUTE' in df.columns:
        df['RAIN_MINUTE'] = pd.to_numeric(df['RAIN_MINUTE'], errors='coerce')
        rain_minute = df['RAIN_MINUTE']
        if rain_minute.notna().any():
            hour_periods = df.index.to_period('h')
            rain_hour_running = (
                rain_minute.groupby(hour_periods)
                .expanding()
                .mean()
                .reset_index(level=0, drop=True)
            )
            df['RAIN_HOUR'] = rain_hour_running

            hourly_mean = rain_minute.groupby(hour_periods).mean()
            hourly_mean.index = hourly_mean.index.to_timestamp()
            hour_start_times = df.index.floor('h')

            def _completed_totals(freq: str) -> np.ndarray:
                cumsum = hourly_mean.groupby(hourly_mean.index.to_period(freq)).cumsum()
                completed = (cumsum - hourly_mean).reindex(hour_start_times, fill_value=0.0)
                return completed.to_numpy()

            completed_day = _completed_totals('D')
            completed_month = _completed_totals('M')
            completed_year = _completed_totals('Y')

            df['RAIN_DAY'] = completed_day + rain_hour_running.to_numpy()
            df['RAIN_MONTH'] = completed_month + rain_hour_running.to_numpy()
            df['RAIN_YEAR'] = completed_year + rain_hour_running.to_numpy()
        else:
            df['RAIN_HOUR'] = np.nan
            df['RAIN_DAY'] = np.nan
            df['RAIN_MONTH'] = np.nan
            df['RAIN_YEAR'] = np.nan
    if 'RAIN_HOUR' in df.columns:
        df['RAIN_HOUR'] = pd.to_numeric(df['RAIN_HOUR'], errors='coerce')
    if 'EVAPOR_MINUTE' in df.columns:
        df['EVAPOR_MINUTE'] = pd.to_numeric(df['EVAPOR_MINUTE'], errors='coerce')
        df['EVAPOR_DAY'] = df['EVAPOR_MINUTE']
    return df
# Function to establish a connection to the MySQL database
def get_db_connection():
    return pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME, port=DB_PORT)


def _is_alarm_active(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'', '0', 'false', 'inactive', 'не'}:
            return False
        return True
    return bool(value)


def get_active_alarms():
    if not ALARM_TABLE or not ALARM_COLUMNS:
        return {'timestamp': None, 'items': []}

    db_connection = None
    cursor = None
    try:
        db_connection = get_db_connection()
        cursor = db_connection.cursor(DictCursor)
        selected_columns = list(dict.fromkeys([ALARM_DATE_COLUMN] + ALARM_COLUMNS))
        columns_clause = ', '.join(selected_columns)
        query = (
            f"SELECT {columns_clause} FROM {ALARM_TABLE} "
            f"ORDER BY {ALARM_DATE_COLUMN} DESC LIMIT 1"
        )
        cursor.execute(query)
        result = cursor.fetchone()
    except Exception as exc:
        logger.error(f"Error fetching alarms: {exc}", exc_info=True)
        return {'timestamp': None, 'items': []}
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if db_connection is not None:
            try:
                db_connection.close()
            except Exception:
                pass

    if not result:
        return {'timestamp': None, 'items': []}

    timestamp = result.get(ALARM_DATE_COLUMN)
    items = []
    for column in ALARM_COLUMNS:
        value = result.get(column)
        if _is_alarm_active(value):
            processed_value = value
            if isinstance(value, Decimal):
                processed_value = float(value)
            elif isinstance(value, np.generic):
                processed_value = value.item()
            elif isinstance(value, datetime):
                processed_value = value.strftime('%Y-%m-%dT%H:%M:%S')
            items.append(
                {
                    'column': column,
                    'name': ALARM_COLUMN_LABELS.get(column, column),
                    'description': ALARM_COLUMN_DESCRIPTIONS.get(column, column),
                    'value': processed_value,
                }
            )

    if isinstance(timestamp, datetime):
        timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S')
    else:
        timestamp_str = str(timestamp) if timestamp else None

    return {'timestamp': timestamp_str, 'items': items}

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
app.config['SESSION_COOKIE_SECURE'] = False  # Allow cookies over HTTP; set to True when using HTTPS
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
            cursor = None
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

                    if 'RAIN_MINUTE' in df_last_min_data.columns and not df_last_min_data.empty:
                        last_timestamp = df_last_min_data.index[-1]
                        if pd.notna(last_timestamp):
                            end_time = last_timestamp.to_pydatetime()
                            hour_start = end_time.replace(minute=0, second=0, microsecond=0)
                            day_start = hour_start.replace(hour=0)
                            month_start = hour_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                            year_start = hour_start.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

                            rain_hour_value = np.nan
                            avg_query = (
                                f"SELECT AVG(RAIN_MINUTE) FROM {DB_TABLE_MIN} "
                                f"WHERE {DATE_COLUMN} >= %s AND {DATE_COLUMN} <= %s"
                            )
                            try:
                                cursor.execute(avg_query, (hour_start, end_time))
                                avg_result = cursor.fetchone()
                                if avg_result and avg_result[0] is not None:
                                    rain_hour_value = float(avg_result[0])
                                else:
                                    rain_hour_value = 0.0
                            except Exception as exc:
                                logger.error(
                                    f"Error calculating RAIN_HOUR average: {exc}",
                                    exc_info=True,
                                )
                                rain_hour_value = 0.0
                            df_last_min_data.loc[last_timestamp, 'RAIN_HOUR'] = rain_hour_value

                            sum_query = (
                                f"SELECT COALESCE(SUM(RAIN_MINUTE), 0) FROM {DB_TABLE} "
                                f"WHERE {DATE_COLUMN} > %s AND {DATE_COLUMN} <= %s"
                            )

                            def _completed_sum(start_boundary: datetime) -> float:
                                try:
                                    cursor.execute(sum_query, (start_boundary, hour_start))
                                    sum_result = cursor.fetchone()
                                    if sum_result and sum_result[0] is not None:
                                        value = sum_result[0]
                                        if isinstance(value, Decimal):
                                            return float(value)
                                        return float(value)
                                except Exception as exc:
                                    logger.error(
                                        f"Error calculating rainfall total from {start_boundary}: {exc}",
                                        exc_info=True,
                                    )
                                return 0.0

                            rain_hour_contribution = 0.0 if pd.isna(rain_hour_value) else rain_hour_value
                            day_total = _completed_sum(day_start) + rain_hour_contribution
                            month_total = _completed_sum(month_start) + rain_hour_contribution
                            year_total = _completed_sum(year_start) + rain_hour_contribution

                            df_last_min_data.loc[last_timestamp, 'RAIN_DAY'] = day_total
                            df_last_min_data.loc[last_timestamp, 'RAIN_MONTH'] = month_total
                            df_last_min_data.loc[last_timestamp, 'RAIN_YEAR'] = year_total

                if not df_last_min_data.empty:
                    df_last_min_values = df_last_min_data.reindex(columns=DATA_COLUMNS)
                    df_last_min_values.reset_index(inplace=True)
                    numeric_columns = df_last_min_values.select_dtypes(include=[np.number]).columns
                    df_last_min_values[numeric_columns] = df_last_min_values[numeric_columns].round(1)
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass
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
        return f(*args, **kwargs)
    return decorated_function


# Route to serve the HTML webpage
@app.route('/')
@login_required
def index():
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
            if 'WIND_DIR' in df.columns:
                df_res['WIND_DIR'] = df['WIND_DIR'].resample('h').apply(_vector_average)
            if 'EVAPOR_MINUTE' in df.columns:
                df_res['EVAPOR_MINUTE'] = df['EVAPOR_MINUTE'].resample('h').apply(_last_valid_value)
        elif period == '30d':
            df_res = df.drop(columns=['RADIATION'], errors='ignore').resample('d').mean()
            if 'WIND_DIR' in df.columns:
                df_res['WIND_DIR'] = df['WIND_DIR'].resample('d').apply(_vector_average)
            if 'RADIATION' in df.columns:
                rad = df['RADIATION'].resample('d').sum(min_count=1) * KWH_PER_M2_FROM_HOUR
                df_res = df_res.join(rad.rename('RADIATION'))
            if 'EVAPOR_MINUTE' in df.columns:
                df_res['EVAPOR_MINUTE'] = df['EVAPOR_MINUTE'].resample('d').apply(_last_valid_value)
        else:
            df_res = df.drop(columns=['RADIATION'], errors='ignore').resample('M').mean()
            if 'WIND_DIR' in df.columns:
                df_res['WIND_DIR'] = df['WIND_DIR'].resample('M').apply(_vector_average)
            if 'RADIATION' in df.columns:
                daily_rad = df['RADIATION'].resample('d').sum(min_count=1) * KWH_PER_M2_FROM_HOUR
                rad = daily_rad.resample('M').sum(min_count=1)
                df_res = df_res.join(rad.rename('RADIATION'))
            if 'EVAPOR_MINUTE' in df.columns:
                df_res['EVAPOR_MINUTE'] = df['EVAPOR_MINUTE'].resample('M').apply(_last_valid_value)
            if not df_res.empty:
                df_res.index = df_res.index.to_period('M').to_timestamp()

        df_res = df_res.dropna(how='all')
        df_res.reset_index(inplace=True)
        # Replace NaN values with None to ensure valid JSON serialization
        df_res = df_res.astype(object).where(pd.notnull(df_res), None)

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
    min_val = series.min()
    min_time = series.idxmin()
    max_val = series.max()
    max_time = series.idxmax()
    return {
        'min': float(min_val),
        'min_time': _format_dt(min_time),
        'max': float(max_val),
        'max_time': _format_dt(max_time),
    }


def format_number(val: float) -> str:
    sign = '-' if val < 0 else ''
    val = abs(val)
    whole, frac = f"{val:.1f}".split('.')
    groups = []
    while whole:
        groups.append(whole[-3:])
        whole = whole[:-3]
    whole_with_space = ' '.join(reversed(groups))
    return f"{sign}{whole_with_space},{frac}"


def _vector_average(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors='coerce').dropna()
    if values.empty:
        return np.nan
    radians = np.deg2rad(values)
    sin_sum = np.sin(radians).sum()
    cos_sum = np.cos(radians).sum()
    if sin_sum == 0 and cos_sum == 0:
        return np.nan
    angle = np.degrees(np.arctan2(sin_sum, cos_sum))
    if angle < 0:
        angle += 360
    return angle


def _last_valid_value(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[-1])


def _dew_point(temp_c: pd.Series, rel_hum: pd.Series) -> pd.Series:
    """Calculate dew point temperature given air temperature and relative humidity."""
    temp_c = pd.to_numeric(temp_c, errors="coerce")
    rel_hum = pd.to_numeric(rel_hum, errors="coerce")
    a = 17.27
    b = 237.7
    alpha = (a * temp_c / (b + temp_c)) + np.log(rel_hum / 100.0)
    return (b * alpha) / (a - alpha)


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
            f"SELECT {DATE_COLUMN}, T_AIR, T_WATER, REL_HUM, P_REL, P_ABS, WIND_GUST, WIND_DIR, RAIN_MINUTE, EVAPOR_MINUTE, RADIATION "
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
                "T_WATER",
                "REL_HUM",
                "P_REL",
                "P_ABS",
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

    for col in ['T_AIR', 'T_WATER', 'REL_HUM', 'P_REL', 'P_ABS', 'WIND_GUST', 'WIND_DIR', 'RAIN_MINUTE', 'EVAPOR_MINUTE', 'RADIATION']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df.set_index(DATE_COLUMN, inplace=True)

    result = []

    temp = _min_max_with_time(df, 'T_AIR')
    if temp:
        result.append({
            "label": "Температура",
            "value": [
                f"мин {format_number(temp['min'])}°C ({temp['min_time']})",
                f"макс {format_number(temp['max'])}°C ({temp['max_time']})",
            ],
        })

    water = _min_max_with_time(df, 'T_WATER')
    if water:
        result.append({
            "label": "Температура на водата",
            "value": [
                f"мин {format_number(water['min'])}°C ({water['min_time']})",
                f"макс {format_number(water['max'])}°C ({water['max_time']})",
            ],
        })

    hum = _min_max_with_time(df, 'REL_HUM')
    if hum:
        result.append({
            "label": "Относителна влажност",
            "value": [
                f"мин {format_number(hum['min'])}% ({hum['min_time']})",
                f"макс {format_number(hum['max'])}% ({hum['max_time']})",
            ],
        })

    if 'T_AIR' in df.columns and 'REL_HUM' in df.columns:
        df['DEW_POINT'] = _dew_point(df['T_AIR'], df['REL_HUM'])
        dew = _min_max_with_time(df, 'DEW_POINT')
        if dew:
            result.append({
                "label": "Точка на роса",
                "value": [
                    f"мин {format_number(dew['min'])}°C ({dew['min_time']})",
                    f"макс {format_number(dew['max'])}°C ({dew['max_time']})",
                ],
            })

    press_rel = _min_max_with_time(df, 'P_REL')
    if press_rel:
        result.append({
            "label": "Относително налягане",
            "value": [
                f"мин {format_number(press_rel['min'])} hPa ({press_rel['min_time']})",
                f"макс {format_number(press_rel['max'])} hPa ({press_rel['max_time']})",
            ],
        })

    press_abs = _min_max_with_time(df, 'P_ABS')
    if press_abs:
        result.append({
            "label": "Абсолютно налягане",
            "value": [
                f"мин {format_number(press_abs['min'])} hPa ({press_abs['min_time']})",
                f"макс {format_number(press_abs['max'])} hPa ({press_abs['max_time']})",
            ],
        })

    gust_series = df['WIND_GUST'].dropna()
    if not gust_series.empty:
        gust_value = float(gust_series.max())
        gust_time = gust_series.idxmax()
        direction = None
        if 'WIND_DIR' in df.columns:
            dir_val = df.loc[gust_time, 'WIND_DIR']
            if isinstance(dir_val, pd.Series):
                dir_val = dir_val.iloc[0]
            direction = dir_val
        dir_text = f", посока {direction}" if pd.notnull(direction) else ''
        result.append({
            "label": "Порив на вятъра",
            "value": f"макс {format_number(gust_value)} km/h{dir_text} ({_format_dt(gust_time)})",
        })

    rain_series = df['RAIN_MINUTE'].dropna()
    if not rain_series.empty:
        rain_total = float(rain_series.sum())
        result.append({
            "label": "Сума валежи",
            "value": f"{format_number(rain_total)} mm",
        })

    evap_series = df['EVAPOR_MINUTE'].dropna()
    if not evap_series.empty:
        evap_value = float(evap_series.iloc[-1])
        label = "Изпарение за деня" if period == 'today' else "Изпарение"
        unit = "mm" if period == 'today' else "mm/day"
        result.append({
            "label": label,
            "value": f"{format_number(evap_value)} {unit}",
        })

    if period != 'today' and not df['RAIN_MINUTE'].dropna().empty:
        daily_rain = df['RAIN_MINUTE'].resample('D').sum()
        max_day = daily_rain.max()
        max_day_time = _format_dt(daily_rain.idxmax())
        result.append({
            "label": "Макс за ден",
            "value": f"{format_number(max_day)} mm ({max_day_time})",
        })
        intensity_series = df['RAIN_MINUTE'].dropna()
        if not intensity_series.empty:
            intensity_value = float(intensity_series.max())
            intensity_time = intensity_series.idxmax()
            result.append({
                "label": "Макс интензитет",
                "value": f"{format_number(intensity_value)} mm/min ({_format_dt(intensity_time)})",
            })

    rad_series = df['RADIATION'].dropna()
    if not rad_series.empty:
        rad_max = float(rad_series.max())
        rad_time = rad_series.idxmax()
        result.append({
            "label": "Слънчева радиация",
            "value": f"макс {format_number(rad_max)} W/m² ({_format_dt(rad_time)})",
        })
        daily_energy = rad_series.resample('D').sum() * KWH_PER_M2_FROM_MINUTE
        if daily_energy.empty:
            total_energy = 0.0
        elif period in {'today', 'month'}:
            total_energy = float(daily_energy.sum())
        else:
            monthly_energy = daily_energy.resample('M').sum()
            total_energy = float(monthly_energy.sum()) if not monthly_energy.empty else 0.0
        result.append({
            "label": "Сума слънчева радиация",
            "value": f"{format_number(total_energy)} kWh/mm²",
        })

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


@app.route('/report_data', methods=['GET'])
@login_required
def report_data_endpoint():
    """Return daily statistics for a given month and year."""
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)

    today_day = datetime.today().day
    if not year or not month:
        return jsonify({})
    try:
        db_connection = get_db_connection()
        cursor = db_connection.cursor()

        # Discover available columns to avoid SQL errors if some are missing
        cursor.execute(f"SHOW COLUMNS FROM {DB_TABLE}")
        available = {row[0] for row in cursor.fetchall()}

        requested = [
            "T_AIR",
            "T_WATER",
            "REL_HUM",
            "P_REL",
            "P_ABS",
            "WIND_SPEED_1",
            "WIND_SPEED_2",
            "WIND_DIR",
            "RADIATION",
            "RAIN_MINUTE",
            "EVAPOR_MINUTE",
        ]
        cols = [c for c in requested if c in available]

        query = (
            f"SELECT {DATE_COLUMN}, {', '.join(cols)} FROM {DB_TABLE} "
            f"WHERE YEAR({DATE_COLUMN}) = %s AND MONTH({DATE_COLUMN}) = %s "
            f"AND DAY({DATE_COLUMN}) < %s "
            f"ORDER BY {DATE_COLUMN} ASC"
        )
        cursor.execute(query, (year, month, today_day))
        data = cursor.fetchall()
        cursor.close()
        db_connection.close()

        df = pd.DataFrame(data, columns=[DATE_COLUMN] + cols)
        if df.empty:
            logger.error(f"No data returned for {year}-{month} from {DB_TABLE}")
            return jsonify({})

        # Convert numeric values and track columns that fail to parse
        for c in cols:
            raw = df[c].astype(str)
            df[c] = pd.to_numeric(raw.str.replace(",", "."), errors="coerce")
            if df[c].isna().all():
                logger.error(
                    "Column %s has no numeric data for %04d-%02d; sample raw values: %s",
                    c,
                    year,
                    month,
                    raw.head().tolist(),
                )

        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df.set_index(DATE_COLUMN, inplace=True)
        df.sort_index(inplace=True)

        import calendar

        days_in_month = calendar.monthrange(year, month)[1]
        idx = pd.date_range(start=datetime(year, month, 1), periods=days_in_month, freq="D")

        combined = pd.DataFrame(index=idx)

        mean_cols = [
            c
            for c in [
                "T_AIR",
                "T_WATER",
                "REL_HUM",
                "P_REL",
                "P_ABS",
                "WIND_SPEED_1",
                "WIND_SPEED_2",
            ]
            if c in df.columns
        ]
        if mean_cols:
            combined = combined.join(df[mean_cols].resample("D").mean())
        if "WIND_DIR" in df.columns:
            wind_daily = df["WIND_DIR"].resample("D").apply(_vector_average)
            combined = combined.join(wind_daily.rename("WIND_DIR"))

        if "RAIN_MINUTE" in df.columns:
            combined = combined.join(
                df["RAIN_MINUTE"].resample("D").sum(min_count=1).rename("RAIN")
            )

        if "EVAPOR_MINUTE" in df.columns:
            combined = combined.join(
                df["EVAPOR_MINUTE"].resample("D").apply(_last_valid_value).rename("EVAPOR_DAY")
            )

        if "RADIATION" in df.columns:
            rad_daily = df["RADIATION"].resample("D").sum(min_count=1) * KWH_PER_M2_FROM_HOUR
            combined = combined.join(rad_daily.rename("RADIATION"))

        at_14_cols = [c for c in ["T_AIR", "REL_HUM", "P_REL"] if c in df.columns]
        if at_14_cols:
            combined = combined.join(
                df.between_time("14:00", "14:00")[at_14_cols]
                .resample("D")
                .mean()
                .add_suffix("_14")
            )

        combined = combined.round(1)

        now = datetime.now()
        if year == now.year and month == now.month:
            today = now.date()
            if today in combined.index:
                combined.loc[today] = np.nan

        for col in combined.columns:
            if combined[col].isna().all():
                logger.error(
                    "No computed data for column %s in %04d-%02d after aggregation",
                    col,
                    year,
                    month,
                )

        result = {
            col: [None if pd.isna(v) else float(v) for v in combined[col]]
            for col in combined.columns
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /report_data endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


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
                if isinstance(value, str):
                    try:
                        value = float(value.replace(',', '.'))
                    except ValueError:
                        pass
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
                if isinstance(value, str):
                    try:
                        value = float(value.replace(',', '.'))
                    except ValueError:
                        pass
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

        alarms_info = get_active_alarms()
        ftp_status = getattr(insertMissingDataFromCSV, 'last_ftp_status', None)
        if isinstance(ftp_status, np.bool_):
            ftp_status = bool(ftp_status)

        return jsonify({
            "min_values_data": min_values_data,
            "columns_values": DATA_COLUMNS,
            "columns_bg": DATA_COLUMNS_BG,
            "columns_units": DATA_COLUMNS_UNITS,
            "alarms": alarms_info.get('items', []),
            "alarms_timestamp": alarms_info.get('timestamp'),
            "ftp_connection_ok": ftp_status,
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
    #app.run(host='0.0.0.0', port=5010, debug=False)
    # uncomment this to start in non production
    serve(app, host='0.0.0.0', port=5010)

