import os
import pandas as pd
import configparser
from datetime import datetime,timedelta
from ftplib import FTP, error_perm
import sqlalchemy as sa
from sqlalchemy.sql import text
import mean_1h
import logging


current_dir = os.path.dirname(os.path.abspath(__file__))
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatter and file handlers while avoiding duplicates
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

error_log_path = os.path.join(current_dir, 'errorlog.log')
warning_log_path = os.path.join(current_dir, 'errorlog.txt')

def _add_file_handler(path, level):
    absolute_path = os.path.abspath(path)
    for existing_handler in logger.handlers:
        if isinstance(existing_handler, logging.FileHandler) and os.path.abspath(existing_handler.baseFilename) == absolute_path:
            existing_handler.setLevel(level)
            if existing_handler.formatter is None:
                existing_handler.setFormatter(formatter)
            return

    handler = logging.FileHandler(absolute_path)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


_add_file_handler(error_log_path, logging.ERROR)
_add_file_handler(warning_log_path, logging.WARNING)

last_ftp_status = None
last_ftp_error = None
last_ftp_check = None
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
        'table_name': config.get('SQL', 'DB_TABLE_MIN'),
        'hourly_table': config.get('SQL', 'mean_1hour_table')
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
    global last_ftp_status, last_ftp_error, last_ftp_check
    ftp = FTP()
    ftp.connect(ftp_config['ftp_host'], int(ftp_config['ftp_port']))
    ftp.login(ftp_config['ftp_username'], ftp_config['ftp_password'])
    ftp.set_pasv(True)  # Force Passive Mode
    print("FTP connection successful")
    last_ftp_status = True
    last_ftp_error = None
    last_ftp_check = datetime.now()
    return ftp


# Step 5: Download CSV files from FTP server
def download_csv_files(ftp, last_date_str, remote_csv_path, local_folder, ftp_config):

    current_datetime = datetime.now()

    try:
        if isinstance(last_date_str, str):
            last_date = datetime.strptime(last_date_str.strip(), '%Y-%m-%d %H:%M:%S')
        else:
            last_date = last_date_str
    except (TypeError, ValueError) as e:
        logger.error(
            "Error parsing last_record_datetime '%s' while downloading CSV files: %s",
            last_date_str,
            e,
            exc_info=True,
        )
        ftp.quit()
        return

    if last_date is None:
        start_date = current_datetime.date()
    else:
        start_date = last_date.date()

    end_date = current_datetime.date()

    if start_date > end_date:
        start_date = end_date

    remote_base_path = remote_csv_path.rstrip('/')

    target_date = start_date
    while target_date <= end_date:
        year_folder = f"{target_date.year}"
        month_folder = target_date.strftime('%m')
        remote_dir = f"{remote_base_path}/{year_folder}/{month_folder}"
        file_name = (
            f"{ftp_config['csv_template_name']}{target_date.strftime('%Y_%m_%d')}.csv"
        )

        try:
            ftp.cwd(remote_dir)
        except error_perm as e:
            logger.error(
                "Remote directory missing for %s at %s: %s",
                target_date,
                remote_dir,
                e,
                exc_info=True,
            )
            target_date += timedelta(days=1)
            continue
        except Exception as e:
            logger.error(
                "Failed to change directory to %s while downloading CSV files: %s",
                remote_dir,
                e,
                exc_info=True,
            )
            target_date += timedelta(days=1)
            continue

        local_file = os.path.join(local_folder, file_name)

        try:
            with open(local_file, 'wb') as f:
                ftp.retrbinary(f"RETR {file_name}", f.write)
            print(f"Downloaded: {file_name}")
        except error_perm as e:
            logger.error(
                "File not found for %s: %s in %s: %s",
                target_date,
                file_name,
                remote_dir,
                e,
                exc_info=True,
            )
            if os.path.exists(local_file):
                os.remove(local_file)
        except Exception as e:
            logger.error(
                "Failed to download %s from %s: %s",
                file_name,
                remote_dir,
                e,
                exc_info=True,
            )
            if os.path.exists(local_file):
                os.remove(local_file)

        target_date += timedelta(days=1)

    ftp.quit()


def zero_seconds(dt):
    """ Helper function to zero out the seconds and microseconds in a datetime object """
    return dt.replace(second=0, microsecond=0)


def insert_data_into_db(engine, table_name, csv_data, column_mapping, db_col_names):
    # Rename columns based on the mapping
    csv_data = csv_data.rename(columns=column_mapping)

    if 'RAIN' not in csv_data.columns and 'RAIN_MINUTE' in csv_data.columns:
        csv_data = csv_data.rename(columns={'RAIN_MINUTE': 'RAIN'})

    # Drop duplicate columns if any
    csv_data = csv_data.loc[:, ~csv_data.columns.duplicated()]

    # Ensure 'DateRef' is populated correctly from 'Time'
    if 'DateRef' not in csv_data.columns and 'Time' in csv_data.columns:
        csv_data['DateRef'] = pd.to_datetime(csv_data['Time'], errors='coerce')
    elif 'DateRef' in csv_data.columns:
        csv_data['DateRef'] = pd.to_datetime(csv_data['DateRef'], errors='coerce')

    # Zero out seconds in CSV data
    csv_data.loc[:, 'DateRef'] = csv_data['DateRef'].apply(zero_seconds)

    # Remove rows without timestamps and make sure the last measurement for
    # each minute is retained after sorting chronologically
    csv_data = csv_data.dropna(subset=['DateRef'])
    csv_data = csv_data.sort_values('DateRef')
    csv_data = csv_data.drop_duplicates(subset=['DateRef'], keep='last')

    # Convert measurement columns to numeric
    for col in db_col_names:
        if col != 'DateRef' and col in csv_data.columns:
            csv_data[col] = pd.to_numeric(csv_data[col], errors='coerce')

    # Keep rainfall values as mm/h intensity so raw_1min stores the direct readings

    # Ensure that all required columns are present in the DataFrame
    missing_cols = set(db_col_names) - set(csv_data.columns)
    if missing_cols:
        logger.error(
            "Missing required columns for table %s while inserting data: %s",
            table_name,
            ', '.join(sorted(missing_cols)),
        )
        return

    # Reorder columns to match the database table columns
    csv_data = csv_data[db_col_names]

    # Read existing data from the database
    try:
        existing_data = pd.read_sql_table(table_name, engine, columns=['DateRef'])
        existing_data['DateRef'] = pd.to_datetime(existing_data['DateRef'])
    except Exception as e:
        logger.error(
            "Error reading existing data from table %s: %s",
            table_name,
            e,
            exc_info=True,
        )
        return

    # Zero out seconds in existing data
    existing_data.loc[:, 'DateRef'] = existing_data['DateRef'].apply(zero_seconds)

    existing_dates = set(existing_data['DateRef'])

    # Split CSV data into new and existing timestamps
    existing_updates = csv_data[csv_data['DateRef'].isin(existing_dates)]
    new_data = csv_data[~csv_data['DateRef'].isin(existing_dates)]

    # Update existing records in a transaction
    if not existing_updates.empty:
        update_columns = [col for col in db_col_names if col != 'DateRef']
        set_clause = ', '.join([f"`{col}` = :{col}" for col in update_columns])
        update_stmt = text(
            f"UPDATE {table_name} SET {set_clause} WHERE `DateRef` = :DateRef"
        )
        try:
            with engine.begin() as connection:
                for _, row in existing_updates.iterrows():
                    params = {col: row[col] for col in db_col_names if col in existing_updates.columns}
                    connection.execute(update_stmt, params)
            print(f"Updated {len(existing_updates)} records in {table_name}.")
        except Exception as e:
            logger.error(
                "Error updating existing records in table %s from file data: %s",
                table_name,
                e,
                exc_info=True,
            )

    if new_data.empty:
        print("No new records to insert.")
    else:
        # Insert the new data into the database
        try:
            new_data.to_sql(table_name, engine, if_exists='append', index=False)
            print(f"Inserted {len(new_data)} records into {table_name}.")
        except Exception as e:
            logger.error(
                "Error inserting new data into table %s: %s",
                table_name,
                e,
                exc_info=True,
            )

def main():
    try:
        # Read config
        db_config, ftp_config = read_config()
    except Exception as e:
        logger.error("Error reading configuration: %s", e, exc_info=True)
        return

    try:
        # Create a database connection
        engine = create_db_connection(db_config)
    except Exception as e:
        logger.error("Error creating database connection: %s", e, exc_info=True)
        return

    try:
        # Get the last record date from the DB
        last_record_date = get_last_record_datetime(engine, db_config['table_name'])
    except Exception as e:
        logger.error("Error fetching last record datetime: %s", e, exc_info=True)
        return

    if last_record_date is None:
        print("No records in the database. Downloading recent CSV files only.")
        last_record_date = datetime.now() - timedelta(days=1)
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
        global last_ftp_status, last_ftp_error, last_ftp_check
        last_ftp_status = False
        last_ftp_error = str(e)
        last_ftp_check = datetime.now()
        logger.error("Error connecting to FTP server or downloading files: %s", e, exc_info=True)
        return

    # Track the latest timestamp from newly inserted minute data
    earliest_new_datetime = None
    latest_new_datetime = None

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
                logger.error("Error reading CSV file %s with latin1 fallback: %s", csv_file, e, exc_info=True)
                continue
        except Exception as e:
            logger.error("General error reading CSV file %s: %s", csv_file, e, exc_info=True)
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
                logger.error(
                    "Missing 'Time' column in CSV file %s while preparing data for insertion",
                    csv_file,
                )
                continue

            # Filter rows where 'DateRef' is greater than the last record date
            csv_data_filtered = csv_data[csv_data['DateRef'] > last_record_date]

            if csv_data_filtered.empty:
                print(f"No new data in file {csv_file}. Skipping.")
                continue

            # Track the timestamp range for the new data
            current_min = csv_data_filtered['DateRef'].min()
            if earliest_new_datetime is None or current_min < earliest_new_datetime:
                earliest_new_datetime = current_min
            current_max = csv_data_filtered['DateRef'].max()
            if latest_new_datetime is None or current_max > latest_new_datetime:
                latest_new_datetime = current_max

            # Insert the filtered data into the database
            insert_data_into_db(
                engine,
                db_config['table_name'],
                csv_data_filtered,
                ftp_config['column_mapping'],
                ftp_config['db_col_names'],
            )
        except Exception as e:
            logger.error("Error processing file %s: %s", csv_file, e, exc_info=True)
            continue

    try:
        hourly_table = db_config['hourly_table']
        last_hour_record = get_last_record_datetime(engine, hourly_table)
        if latest_new_datetime is not None:
            latest_completed_hour = (
                latest_new_datetime.replace(minute=0, second=0, microsecond=0)
                - timedelta(hours=1)
            )

            if earliest_new_datetime is None:
                first_unprocessed_hour = latest_completed_hour
            else:
                first_unprocessed_hour = earliest_new_datetime.replace(
                    minute=0, second=0, microsecond=0
                )

            if last_hour_record is not None:
                current_hour = (
                    last_hour_record.replace(minute=0, second=0, microsecond=0)
                    + timedelta(hours=1)
                )
                if current_hour < first_unprocessed_hour:
                    current_hour = first_unprocessed_hour
            else:
                current_hour = first_unprocessed_hour

            while current_hour is not None and current_hour <= latest_completed_hour:
                call_mean_hourly(current_hour, current_hour + timedelta(hours=1))
                current_hour += timedelta(hours=1)
    except Exception as e:
        logger.error("Error calculating hourly mean: %s", e, exc_info=True)
        return

    return "Data update complete."


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("An unexpected error occurred in main execution: %s", e, exc_info=True)

