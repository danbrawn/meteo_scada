import os
import pandas as pd
import configparser
from datetime import datetime,timedelta
from ftplib import FTP
import sqlalchemy as sa
from sqlalchemy.sql import text
import mean_1h
import logging
from logging import FileHandler,WARNING


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
def call_mean_hourly(start_datetime, end_datetime):
    print(f"Calling mean_1h.py with range {start_datetime} to {end_datetime}")
    mean_1h.mean_1h(start_datetime, end_datetime)

# Step 1: Read from config.ini file
def read_config():
    # Load configuration from config.ini
    with open('config.ini', 'r', encoding='utf-8') as config_file:
        config = configparser.ConfigParser(interpolation=None)
        config.read_file(config_file)

    # Read database config
    db_config = {
        'username': config.get('SQL', 'user'),
        'password': config.get('SQL', 'password'),
        'host': config.get('SQL', 'host'),
        'port': config.get('SQL', 'port'),
        'database': config.get('SQL', 'database'),
        'table_name': config.get('SQL', 'DB_TABLE_MIN')
    }

    # Read FTP config
    csv_cols = [col.strip() for col in config.get('CSV', 'csv_col_names').split(',')]
    db_cols = [col.strip() for col in config.get('CSV', 'db_col_names').split(',')]
    ftp_config = {
        'ftp_host': config.get('CSV', 'ftp_host'),
        'ftp_port': config.get('CSV', 'ftp_port'),
        'ftp_username': config.get('CSV', 'ftp_username'),
        'ftp_password': config.get('CSV', 'ftp_password'),
        'remote_csv_path': config.get('CSV', 'remote_csv_path'),
        'local_csv_folder': config.get('CSV', 'local_csv_folder'),  # Retrieve local folder path
        'csv_template_name': config.get('CSV', 'csv_template_name').strip(),
        'csv_col_names': csv_cols,  # Comma-separated columns for CSV
        'db_col_names': db_cols,  # Comma-separated columns for DB
        'column_mapping': dict(zip(csv_cols, db_cols))
    }

    return db_config, ftp_config


# Step 2: Connect to the MySQL database using SQLAlchemy and PyMySQL
def create_db_connection(db_config):
    conn_str = (
        f"mysql+pymysql://{db_config['username']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = sa.create_engine(conn_str, echo=True)
    return engine


# Step 3: Find the last record in the DB
def get_last_record_datetime(engine, table_name):
    query = text(f"SELECT MAX(`DateRef`) FROM {table_name}")
    with engine.connect() as connection:
        result = connection.execute(query).scalar()
    return result


# Step 4: Connect to the FTP server and fetch the list of files
def connect_ftp(ftp_config):
    ftp = FTP()
    ftp.connect(ftp_config['ftp_host'], int(ftp_config['ftp_port']))
    ftp.login(ftp_config['ftp_username'], ftp_config['ftp_password'])
    ftp.set_pasv(True)  # Force Passive Mode
    print("FTP connection successful")
    return ftp


# Step 5: Download CSV files from FTP server
def download_csv_files(ftp, last_date_str, remote_csv_path, local_folder, ftp_config):

    current_date = datetime.now()

    # Convert last_date_str to datetime object
    #last_date = datetime.strptime(last_date_str, '%Y-%m-%d %H:%M:%S')
    try:
        if isinstance(last_date_str, str):
            last_date = datetime.strptime(last_date_str.strip(), '%Y-%m-%d %H:%M:%S')
        else:
            last_date = last_date_str
        #last_date = datetime.strptime(last_date_str.strip(), '%Y-%m-%d %H:%M:%S')
    except ValueError as e:
        print(f"Error parsing last_record_datetime: {e}")
        return
    # Extract year from last_date
    last_year = last_date.year

    # Construct remote path for the year of the last record
    year_folder = f"{last_year}"
    remote_year_path = f"{remote_csv_path}/{year_folder}"

    try:
        ftp.cwd(remote_year_path)
        months = ftp.nlst()

        # Iterate through months and download files
        for month_folder in months:
            try:
                ftp.cwd(month_folder)
                files = ftp.nlst()

                for file_name in files:
                    if file_name.startswith(ftp_config['csv_template_name']) and file_name.endswith('.csv'):
                        try:
                            date_str = file_name.replace(ftp_config['csv_template_name'], '').replace('.csv', '')
                            file_date = datetime.strptime(date_str, '%Y_%m_%d')
                            file_date_end = datetime.combine(file_date, datetime.max.time())
                            # Include records from last_date (inclusive) up to current_date
                            if last_date <= file_date_end or last_date <= current_date :
                                local_file = os.path.join(local_folder, file_name)
                                with open(local_file, 'wb') as f:
                                    ftp.retrbinary(f"RETR {file_name}", f.write)
                                print(f"Downloaded: {file_name}")
                        except ValueError as ve:
                            print(f"Date parsing error for file {file_name}: {ve}")

            except Exception as e:
                print(f"Failed to process month folder {month_folder}: {e}")
            finally:
                ftp.cwd("..")

    except Exception as e:
        print(f"Failed to process year folder {year_folder}: {e}")

    ftp.quit()


def zero_seconds(dt):
    """ Helper function to zero out the seconds and microseconds in a datetime object """
    return dt.replace(second=0, microsecond=0)


def insert_data_into_db(engine, table_name, csv_data, column_mapping, db_col_names):
    # Rename columns based on the mapping
    csv_data = csv_data.rename(columns=column_mapping)

    # Drop duplicate columns if any
    csv_data = csv_data.loc[:, ~csv_data.columns.duplicated()]

    # Ensure 'DateRef' is populated correctly from 'Time'
    if 'DateRef' not in csv_data.columns and 'Time' in csv_data.columns:
        csv_data['DateRef'] = pd.to_datetime(csv_data['Time'], errors='coerce')
    elif 'DateRef' in csv_data.columns:
        csv_data['DateRef'] = pd.to_datetime(csv_data['DateRef'], errors='coerce')

    # Zero out seconds in CSV data
    csv_data.loc[:, 'DateRef'] = csv_data['DateRef'].apply(zero_seconds)

    # Convert measurement columns to numeric
    for col in db_col_names:
        if col != 'DateRef' and col in csv_data.columns:
            csv_data[col] = pd.to_numeric(csv_data[col], errors='coerce')

    # Ensure that all required columns are present in the DataFrame
    missing_cols = set(db_col_names) - set(csv_data.columns)
    if missing_cols:
        print(f"Error: Missing columns in CSV data: {missing_cols}")
        return

    # Reorder columns to match the database table columns
    csv_data = csv_data[db_col_names]

    # Read existing data from the database
    try:
        existing_data = pd.read_sql_table(table_name, engine, columns=db_col_names)
        existing_data['DateRef'] = pd.to_datetime(existing_data['DateRef'])
    except Exception as e:
        print(f"Error reading data from the database: {e}")
        return

    # Zero out seconds in existing data
    existing_data.loc[:, 'DateRef'] = existing_data['DateRef'].apply(zero_seconds)

    # Filter out rows that already exist in the database
    new_data = csv_data[~csv_data['DateRef'].isin(existing_data['DateRef'])]

    if new_data.empty:
        print("No new records to insert.")
    else:
        # Insert the new data into the database
        try:
            new_data.to_sql(table_name, engine, if_exists='append', index=False)
            print(f"Inserted {len(new_data)} records into {table_name}.")
        except Exception as e:
            print(f"Error inserting data into the database: {e}")

def main():
    try:
        # Read config
        db_config, ftp_config = read_config()
    except Exception as e:
        logging.error(f"Error reading configuration: {e}")
        return

    try:
        # Create a database connection
        engine = create_db_connection(db_config)
    except Exception as e:
        logging.error(f"Error creating database connection: {e}")
        return

    try:
        # Get the last record date from the DB
        last_record_date = get_last_record_datetime(engine, db_config['table_name'])
    except Exception as e:
        logging.error(f"Error fetching last record datetime: {e}")
        return

    if last_record_date is None:
        print("No records in the database. Processing all CSV files.")
        last_record_date = datetime.min
    else:
        print(f"Last record date in the DB: {last_record_date}")

    # Define a local folder to temporarily store downloaded CSV files
    local_csv_folder = ftp_config['local_csv_folder']
    if not os.path.exists(local_csv_folder):
        os.makedirs(local_csv_folder)

    try:
        # Connect to the FTP server and download CSV files
        ftp = connect_ftp(ftp_config)
        download_csv_files(ftp, last_record_date, ftp_config['remote_csv_path'], local_csv_folder, ftp_config)
    except Exception as e:
        logging.error(f"Error connecting to FTP server or downloading files: {e}")
        return

    # Initialize tracking for newly inserted minute data
    earliest_new_datetime = None
    latest_new_datetime = None
    new_hours = set()

    # Process and insert each downloaded CSV file
    for csv_file in os.listdir(local_csv_folder):
        csv_path = os.path.join(local_csv_folder, csv_file)
        try:
            # Read entire CSV to gracefully handle missing columns
            csv_data = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                csv_data = pd.read_csv(csv_path, encoding='latin1')
            except Exception as e:
                logging.error(f"Error reading CSV file {csv_file}: {e}")
                continue
        except Exception as e:
            logging.error(f"General error reading CSV file {csv_file}: {e}")
            continue

        # Ensure all expected columns exist
        for col in ftp_config['csv_col_names']:
            if col not in csv_data.columns:
                csv_data[col] = pd.NA
        csv_data = csv_data[ftp_config['csv_col_names']]

        try:
            # Ensure 'Time' column in CSV is treated as datetime
            if 'Time' in csv_data.columns:
                csv_data['DateRef'] = pd.to_datetime(csv_data['Time'], errors='coerce')
            else:
                print(f"Error: 'Time' column is missing from the CSV file {csv_file}.")
                continue

            # Filter rows where 'DateRef' is greater than the last record date
            csv_data_filtered = csv_data[csv_data['DateRef'] > last_record_date]

            if csv_data_filtered.empty:
                print(f"No new data in file {csv_file}. Skipping.")
                continue

            # Track the time span and affected hours for the new data
            current_min = csv_data_filtered['DateRef'].min()
            current_max = csv_data_filtered['DateRef'].max()
            earliest_new_datetime = current_min if earliest_new_datetime is None else min(earliest_new_datetime, current_min)
            latest_new_datetime = current_max if latest_new_datetime is None else max(latest_new_datetime, current_max)

            new_hours.update(csv_data_filtered['DateRef'].dt.floor('H'))

            # Insert the filtered data into the database
            insert_data_into_db(
                engine,
                db_config['table_name'],
                csv_data_filtered,
                ftp_config['column_mapping'],
                ftp_config['db_col_names'],
            )
        except Exception as e:
            logging.error(f"Error processing file {csv_file}: {e}")
            continue

    # # Step 3: Find the last record in the DB
    # def get_last_record_datetime(engine, table_name):
    #     query = text(f"SELECT MAX(`DateRef`) FROM {table_name}")
    #     with engine.connect() as connection:
    #         result = connection.execute(query).scalar()
    #     return result
    try:
        if new_hours:
            # Process only hours that have fully passed. For example, if data spans
            # 11:00–12:10 at 12:10, compute only for 11:00; the 12:00 hour will
            # be handled after it completes (at 13:00 or later).
            boundary_hour = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            for hour in sorted(new_hours):
                hour_dt = hour.to_pydatetime() if hasattr(hour, 'to_pydatetime') else hour
                if hour_dt <= boundary_hour:
                    call_mean_hourly(hour_dt, hour_dt)
        else:
            return "No new data inserted; skipping hourly mean calculation."
    except Exception as e:
        logging.error(f"Error calculating hourly mean: {e}")
        print(e)
        return

    return "Data update complete."


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An unexpected error occurred in main execution: {e}")

