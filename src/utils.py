import pandas as pd
import numpy as np
import os 
from datetime import timedelta
import copy
import matplotlib.pyplot as plt


class DataSetCreator():
    def __init__(self, data_path):
        self.data_path = data_path

    def load_file_into_df(self, file_name):
        return pd.read_csv(os.path.join(self.data_path, file_name))
    
    def get_data(self, path='halfhourly_dataset/halfhourly_dataset/'):
        data_files = os.listdir(os.path.join(self.data_path, path))
        df_total = None

        for i, file in enumerate(data_files):
            print(f"Loading file {i+1} / {len(data_files)}: {file} into dataframe")
            if file.endswith('.csv'):
               df = self.load_file_into_df(os.path.join(path,file))
               df_total = self.combine_energy_household_count(df, df_total=df_total)

        df_total = df_total.reset_index()
        df_total = df_total.drop(np.where(df_total['energy(kWh/hh)'] <= 0.0)[0])

        return df_total.reset_index(drop=True)

    def combine_energy_household_count(self, df: pd.DataFrame, df_total: pd.DataFrame=None ) -> pd.DataFrame: 
        df = df.rename(columns={"tstp": "timestamp"})
        df.sort_values(by='timestamp', inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        household_count = df.groupby('timestamp')[['LCLid']].nunique()
        household_count = household_count.rename(columns={"LCLid": "houses"})

        df['energy(kWh/hh)'] = pd.to_numeric(df['energy(kWh/hh)'], errors='coerce')

        total_energy = df.groupby('timestamp')[['energy(kWh/hh)']].sum()

        time_energy_count = total_energy.join(household_count, on='timestamp')

        if df_total is not None:
            df_total = pd.concat([df_total, time_energy_count]).groupby('timestamp').sum()
        else:
            df_total = time_energy_count

        del df
        del time_energy_count
        del household_count

        return df_total
    
    def add_weather_data(self, df, weather_path='weather_hourly_darksky.csv'):
        weather_hourly_df = pd.read_csv(os.path.join(self.data_path, weather_path))
        weather_hourly_df = weather_hourly_df.rename(columns={"time": "timestamp"})
        weather_hourly_df = weather_hourly_df.drop(['icon', 'windBearing', 'apparentTemperature', 'summary'], axis=1)
        weather_hourly_df['timestamp'] = pd.to_datetime(weather_hourly_df['timestamp'], utc=True)

        # Get time vec
        weather_hourly_df = weather_hourly_df.sort_values(by='timestamp')
        start_time = weather_hourly_df['timestamp'].iloc[0]
        end_time = weather_hourly_df['timestamp'].iloc[-1]
        iterated_time = start_time + timedelta(minutes=30)
        all_needed_times = [copy.deepcopy(iterated_time)]
        while iterated_time < end_time:
            iterated_time = iterated_time + timedelta(minutes=30)
            all_needed_times.append(copy.deepcopy(iterated_time))
        time_df = pd.DataFrame({'timestamp': all_needed_times})

        # Interpolate weather data
        weather_half_hour_df = pd.merge(time_df, weather_hourly_df, on='timestamp', how='left')
        weather_half_hour_df.sort_values(by='timestamp', inplace=True)
        weather_half_hour_df['precipType'].fillna(method='ffill', inplace=True)
        weather_half_hour_df['precipType'].fillna(method='bfill', inplace=True)
        for col_it in ['temperature', 'dewPoint', 'pressure', 'windSpeed', 'humidity', 'visibility']:
            weather_half_hour_df[col_it].interpolate(method='quadratic', inplace=True)
            weather_half_hour_df[col_it].fillna(method='ffill', inplace=True)
            weather_half_hour_df[col_it].fillna(method='bfill', inplace=True)

        # Replace precipType with index values
        weather_half_hour_df['precipType'] = weather_half_hour_df['precipType'].apply(lambda x: self.PrecipTypeToVal(x))

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Add weather data to df
        df = df.join(weather_half_hour_df.set_index('timestamp'), on='timestamp')

        return df
    
    def add_holidays(self, df, holiday_path='uk_bank_holidays.csv'):
        holidays_df = pd.read_csv(os.path.join(self.data_path, holiday_path))
        holidays_df = holidays_df.drop('Type', axis=1)
        holidays_df['Bank holidays'] = pd.to_datetime(holidays_df['Bank holidays'], format='%Y-%m-%d', utc=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['holiday'] = df['timestamp'].isin(holidays_df['Bank holidays'])
        df['holiday'] = df['holiday'].astype(int)

        return df
    
    def PrecipTypeToVal(self, precip_type):
        if precip_type == 'rain':
            return 0
        elif precip_type == 'snow':
            return 1
        else:
            raise RuntimeError('that is not a good precip type')
        
    def ValToPrecipType(precip_type):
        if precip_type == 0:
            return 'rain'
        elif precip_type == 1:
            return 'snow'
        else:
            raise RuntimeError('that is not a good precip type value')
    
    def seperate_timestamp(self, df: pd.DataFrame, column_order=['timestamp', 'year', 'month', 'day', 'hour', 'minute', 'day_of_week', 'weekend', 'holiday']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x > 4 else 0)

        columns = df.columns.tolist()
        for col in column_order:
            columns.remove(col)
        
        df = df[column_order + columns]

        return df

if __name__ == '__main__':
    
    print(os.getcwd())   

    creator = DataSetCreator('data')

    df = creator.get_data()

    print(df.head())

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    df = creator.add_holidays(df)
    df = creator.add_weather_data(df)
    df = creator.seperate_timestamp(df)

    print(df.head())

    df.info()

    df.to_csv('processed_data.csv', index=False)


