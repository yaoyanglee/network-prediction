import json

import numpy as np
import pandas as pd
from scapy.all import rdpcap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PLDataParser:
    def __init__(self, data_path):
        self.scaler = MinMaxScaler()

        # Open the JSON file and load the JSON data into a Python dictionary
        with open(data_path, 'r') as file:
            self.biflow_data = json.load(file)

    def generate_debug_set(self, min_PL_count=1000):
        pl_data = []
        for currentBiflow, biflow_data in self.biflow_data.items():
            biflowPacketData = biflow_data['packet_data']
            payload_length = biflowPacketData['L4_payload_bytes']
            # print(payload_length)
            if len(payload_length) >= min_PL_count and len(payload_length) < 10000:
                pl_data.extend(payload_length)
                return pd.DataFrame({'PL': payload_length})

        print("No biflows with at least 1000 datapoints")
        return None

    def aggregate_PL(self):  # Aggregate all biflows in this JSON file
        all_pl_data = []

        for currentBiflow, biflow_data in self.biflow_data.items():
            biflowPacketData = biflow_data['packet_data']

            # Extract L4 payload length
            payload_length = biflowPacketData['L4_payload_bytes']

            # Append interarrival times to the list
            all_pl_data.extend(payload_length)

        # Create a DataFrame from aggregated interarrival times
        pl_df = pd.DataFrame({'PL': all_pl_data})

        return pl_df

    def plotPL(self, df):  # Plotting the interarrival time of the packets
        plt.plot(df.index, df['PL'], marker='o')
        plt.xlabel('Packet Number')
        plt.ylabel('Payload Length')
        plt.title('L4 payload length of Packets')
        plt.grid(True)
        plt.show()

    def plotPL_minmax(self, df):  # Plotting the interarrival time after min-max scaling
        plt.plot(df.index, df['PL_scaled'], marker='o')
        plt.xlabel('Packet Number')
        plt.ylabel('Payload Length')
        plt.title('L4 payload length of Packets')
        plt.grid(True)
        plt.show()

    def saturate_99(self, df):  # Saturating to the 99th percentile
        percentile_99 = df["PL"].quantile(0.99)
        df['PL'] = df['PL'].apply(
            lambda x: min(x, percentile_99))

        return df

    def minmax_scaler(self, df):  # Apply min-max scaler to the dataset
        df['PL_scaled'] = self.scaler.fit_transform(
            df[['PL']])

        return df

    def get_minmax_scaler(self):
        return self.scaler

    def create_sequences(self, data, input_window, output_window):
        xs, ys = [], []

        for i in range(len(data) - input_window - output_window + 1):
            x = data[i:i+input_window]
            y = data[i+input_window:i+input_window+output_window]
            xs.append(x)
            ys.append(y)

        if len(xs) != len(ys):
            raise Exception(
                "Number of input data is not equal to the number of ground truth data")

        return np.array(xs), np.array(ys)
