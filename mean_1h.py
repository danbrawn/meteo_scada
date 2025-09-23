
from datetime import datetime, timedelta
import pandas as pd
import sqlalchemy
import sqlalchemy as sa
import sys
import numpy as np
import configparser

# Load configuration from config.ini
with open('config.ini', 'r', encoding='utf-8') as config_file:
    config = configparser.ConfigParser(interpolation=None)
    config.read_file(config_file)

# Get the column names as a comma-separated string from the INI file
columns_string = config.get('SQL', 'columns_list')

# Split the comma-separated string into a list of column names
columns_list = [column.strip() for column in columns_string.split(',')]

# Create the DataFrames using only the measurement columns
output_data = pd.DataFrame(columns=['DateRef'] + columns_list)
raw_data = pd.DataFrame(columns=['DateRef'] + columns_list)
temp_data = pd.DataFrame(columns=['DateRef'] + columns_list)

engine = None
output_columns = len(columns_list)
# Total number of data columns (excluding DateRef)
data_columns_raw = len(columns_list)

now = datetime.now()
minutesFromHourStarted = now.minute
secondsFromHourStarted = now.second
# Check if the mean_1h.py script is run with input parameters
if len(sys.argv) > 2:
    start_time = datetime.strptime(sys.argv[1], "%Y-%m-%d %H:%M:%S")
    end_time_manual = datetime.strptime(sys.argv[2], "%Y-%m-%d %H:%M:%S")
else:
    start_time = datetime.now()-timedelta(hours=1)
    start_time = start_time.replace(minute=0, second=0, microsecond=0)
    end_time_manual = start_time

# end_time = sys.argv[2] if len(sys.argv) > 1 else datetime.now()-timedelta(minutes=minutesFromHourStarted)

def openSQLconnection(start_date):
    global output_data
    global start_time
    global end_time
    global raw_data
    global engine
    global config

    # Get the parameter values from the 'SQL' section
    user = config.get('SQL', 'user')
    password = config.get('SQL', 'password')
    host = config.get('SQL', 'host')
    port = config.get('SQL', 'port')
    database = config.get('SQL', 'database')
    raw_table = config.get('SQL', 'DB_TABLE_MIN')

    # Connect to the MySQL database using SQLAlchemy and PyMySQL
    conn_str = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
    engine = sa.create_engine(conn_str, echo=True)

    # Construct the list of column names for the query
    query_columns = ", ".join(raw_data.columns)

    try:
        # Query the MySQL table and retrieve the data
        query = f"SELECT {query_columns} FROM {raw_table} WHERE DateRef BETWEEN %s AND %s ORDER BY DateIn"
        start_time = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end_time = start_time + timedelta(minutes=59, seconds=59)
        raw_data = pd.read_sql(query, engine, params=(start_time, end_time))
        if raw_data.empty:
            output_data.at[0, 'DateRef'] = start_time
            output_data.loc[0, columns_list] = np.nan
            return 'No_data_for_that_hour'
        else:
            raw_data['DateRef'] = pd.to_datetime(raw_data['DateRef'])  # Convert DateRef column to datetime type
            return 'Exists_data_for_that_hour'
    except Exception as e:
        print(f"openSQLconnection {e}")
    st=1


def makeHourData():
    global output_data, start_time, raw_data, temp_data, config

    try:
        # Initialize temp_data to hold all minute-level data
        temp_data = []

        # Loop through each minute of the hour and get data
        for minute in range(60):
            minute_start = start_time + timedelta(minutes=minute)
            minute_end = minute_start + timedelta(minutes=1)

            # Filter rows within the minute
            minute_data = raw_data[(raw_data['DateRef'] >= minute_start) & (raw_data['DateRef'] < minute_end)]

            # Use the last row if data exists, else append 'No Data'
            if not minute_data.empty:
                latest_row = minute_data.iloc[-1]
                temp_data.append([minute_start] + list(latest_row[1:]))
            else:
                temp_data.append([minute_start] + ['No Data'] * (len(raw_data.columns) - 1))

        # Convert temp_data to a DataFrame
        temp_df = pd.DataFrame(temp_data, columns=raw_data.columns)

        # Compute the mean for numeric columns (excluding 'DateRef')
        numeric = temp_df.iloc[:, 1:].apply(
            lambda s: pd.to_numeric(s.astype(str).str.replace(',', '.'), errors='coerce')
        )
        mean_values = numeric.mean().round(4)
        if 'RAIN' in numeric.columns:
            rain_series = numeric['RAIN'].dropna()
            rain_total = float(rain_series.sum()) if not rain_series.empty else 0.0
            mean_values['RAIN'] = round(rain_total, 4)

        # Update output_data with the mean values
        for col_name, value in mean_values.items():
            output_data.at[0, col_name] = value

        # Set 'DateRef' for the output_data
        # output_data.at[0, 'DateRef'] = start_time
        output_data.at[0, 'DateRef'] = start_time + pd.Timedelta(hours=1)

    except Exception as e:
        print(f"makeHourData {e}")


def populateMean1hour():
    global engine
    global output_data
    global config
    global start_time
    mean_1hour_table = config.get('SQL', 'mean_1hour_table')
    # Insert output_data into the "mean1hour" table using the opened sqlalchemy engine
    try:
        hour_end = start_time + timedelta(hours=1)
        if datetime.now() < hour_end:
            print(f"Hour starting at {start_time} not finished; skipping insert")
            return
        # Ensure numeric columns use a dot decimal separator before writing
        numeric_cols = output_data.columns.drop('DateRef')
        output_data[numeric_cols] = output_data[numeric_cols].apply(
            lambda s: pd.to_numeric(s.astype(str).str.replace(',', '.'), errors='coerce')
        )
        target_time = output_data.at[0, 'DateRef']
        with engine.connect() as conn:
            exists = conn.execute(
                sa.text(f"SELECT 1 FROM {mean_1hour_table} WHERE DateRef = :dt LIMIT 1"),
                {"dt": target_time}
            ).scalar()
        if exists:
            print(f"Record for {target_time} already exists; skipping insert")
            return

        # Now use to_sql() with your DataFrame
        output_data.to_sql(mean_1hour_table, con=engine, if_exists='append', index=False)
        # output_data.to_sql(mean_1hour_table, con=engine, if_exists='append', index=False,
        #                    dtype={'Note': sqlalchemy.NVARCHAR(length=50)})
        print("populateMean1hour ok")
    except Exception as e:
        print(f"populateMean1hour {e}")


def closeSQLconnection():
    global engine
    # Close the MySQL connection
    engine.dispose()


def mean_1h(start_datetime, end_datetime):
    # Initialize necessary global variables and load configuration as required
    start_time = start_datetime - timedelta(hours=1)
    end_time_manual = end_datetime

    # Make sure openSQLconnection, makeHourData, populateMean1hour, and closeSQLconnection
    # functions are defined as in your existing script

    # Execute the data extraction and processing loop
    if end_time_manual > start_time:
        time_difference = end_time_manual - start_time
        time_difference_in_s = time_difference.total_seconds()
        time_difference_in_h = int(divmod(time_difference_in_s, 3600)[0])
        for day in range(1, time_difference_in_h + 1):  # Loop through each hour in the range
            start_time = start_time + timedelta(hours=1)
            hour_end = start_time + timedelta(hours=1)
            if datetime.now() < hour_end:
                print(f"Hour starting at {start_time} not finished; skipping")
                continue
            result = openSQLconnection(start_time.strftime('%Y-%m-%d %H:%M:%S'))
            if result == 'Exists_data_for_that_hour':
                makeHourData()
                populateMean1hour()
            else:
                print(f"No data for hour starting at {start_time}; skipping")
            closeSQLconnection()
    else:
        hour_end = start_time + timedelta(hours=1)
        if datetime.now() > hour_end:
            result = openSQLconnection(start_time.strftime('%Y-%m-%d %H:%M:%S'))
            if result == 'Exists_data_for_that_hour':
                makeHourData()
                populateMean1hour()
            else:
                print(f"No data for hour starting at {start_time}; skipping")
            closeSQLconnection()
        else:
            print(f"Hour starting at {start_time} not finished; skipping")


#temp_date_str = '2024-07-01 09:00:00'
#temp_date = datetime.strptime(temp_date_str, '%Y-%m-%d %H:%M:%S')
#start_time = temp_date
#end_time_manual=temp_date

#for i in range(24):
#result = openSQLconnection(temp_date_str) #'2023-05-17 13:00:00') #start_time)
#
# if end_time_manual > start_time:
#     time_difference=end_time_manual-start_time
#     time_difference_in_s=time_difference.total_seconds()
#     time_difference_in_h=int(divmod(time_difference_in_s, 3600)[0])
#     for day in range(1, time_difference_in_h+1):  # Loop through each day of June
#         start_time = start_time + timedelta(hours=1)
#         result = openSQLconnection(start_time.strftime('%Y-%m-%d %H:%M:%S'))
#         if result == 'No_data_for_that_hour':
#             populateMean1hour()
#         elif result == 'Exists_data_for_that_hour':
#             makeHourData()
#             populateMean1hour()
#         closeSQLconnection()
# else:
#     result = openSQLconnection(start_time.strftime('%Y-%m-%d %H:%M:%S'))
#     if result == 'No_data_for_that_hour':
#         populateMean1hour()
#     elif result == 'Exists_data_for_that_hour':
#         makeHourData()
#         populateMean1hour()
#     closeSQLconnection()
#temp_date_str = temp_date.strftime('%Y-%m-%d %H:%M:%S')





