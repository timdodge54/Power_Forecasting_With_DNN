import pandas as pd
import numpy as np
import os 


class DataSetCreator():
    def __init__(self, data_path):
        self.data_path = data_path

    def load_file_into_df(self, file_name):
        return pd.read_csv(os.path.join(self.data_path, file_name))
    
    def get_data(self):
        data_files = os.listdir(self.data_path)
        df_total = pd.DataFrame()

        for i, file in enumerate(data_files):
           print(f"Loading file {i+1} / {len(data_files)}: {file} into dataframe")
           if file.endswith('.csv'):
               df = self.load_file_into_df(file)
               df_total = self.combine_energy_household_count(df_total, df)
        
        return df_total

    def combine_energy_household_count(self, df_total: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame: 
        df = df.rename(columns={"tstp": "timestamp"})
        df.sort_values(by='timestamp', inplace=True)

        household_count = df.groupby('timestamp')[['LCLid']].nunique()
        household_count = household_count.rename(columns={"LCLid": "Household Count"})

        df['energy(kWh/hh)'] = pd.to_numeric(df['energy(kWh/hh)'], errors='coerce')

        total_energy = df.groupby('timestamp')[['energy(kWh/hh)']].sum()

        time_energy_count = total_energy.join(household_count, on='timestamp')

        df_total = pd.concat([df_total, time_energy_count]).groupby('timestamp').sum()

        del df
        del time_energy_count
        del household_count

        return df_total


if __name__ == '__main__':
    
    
    print(os.getcwd())   

    creator = DataSetCreator('data/halfhourly_dataset/halfhourly_dataset/')

    df = creator.get_data()

    print(df.head())

    df.to_csv('processed_data.csv')


