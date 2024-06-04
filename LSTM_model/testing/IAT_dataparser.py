import json

import numpy as np
import pandas as pd
from scapy.all import rdpcap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

'''
Main Class for managing the dataset
'''


class IATDataParser:
    def __init__(self, data_path):
        self.scaler = MinMaxScaler()

        # Open the JSON file and load the JSON data into a Python dictionary
        with open(data_path, 'r') as file:
            self.biflow_data = json.load(file)

    def generate_debug_set(self, min_iat_count=1000):
        '''
        Returns the first biflow dictionary with at least `min_iat_count` interarrival times (iat).

        Parameters:
        min_iat_count (int): Minimum number of interarrival times required.

        Returns:
        dict: The first biflow dictionary containing at least `min_iat_count` interarrival times.
        '''

        iat_data = []
        for currentBiflow, biflow_data in self.biflow_data.items():
            biflowPacketData = biflow_data['packet_data']
            biflowIAT = biflowPacketData['iat']  # Extract interarrival times

            if len(biflowIAT) >= min_iat_count and len(biflowIAT) <= 10000:
                iat_data.extend(biflowIAT)
                return pd.DataFrame({'Interarrival': iat_data})

        print("No biflows with at least 1000 datapoints")
        return None

    def aggregate_IAT(self):  # Aggregate all biflows in this JSON file
        all_iat_data = []

        for currentBiflow, biflow_data in self.biflow_data.items():
            biflowPacketData = biflow_data['packet_data']

            biflowIAT = biflowPacketData['iat']  # Extract interarrival times

            # Append interarrival times to the list
            all_iat_data.extend(biflowIAT)

        # Create a DataFrame from aggregated interarrival times
        iat_df = pd.DataFrame({'Interarrival': all_iat_data})

        return iat_df

    def plotIAT(self, df):  # Plotting the interarrival time of the packets
        plt.plot(df.index, df['Interarrival'], marker='o')
        plt.xlabel('Packet Number')
        plt.ylabel('Interarrival Time')
        plt.title('Interarrival Times of Packets')
        plt.grid(True)
        plt.show()

    def plotIAT_minmax(self, df):  # Plotting the interarrival time after min-max scaling
        plt.plot(df.index, df['Interarrival_scaled'], marker='o')
        plt.xlabel('Packet Number')
        plt.ylabel('Interarrival Time')
        plt.title('Interarrival Times of Packets')
        plt.grid(True)
        plt.show()

    def saturate_99(self, df):  # Saturating to the 99th percentile
        percentile_99 = df["Interarrival"].quantile(0.99)
        df['Interarrival'] = df['Interarrival'].apply(
            lambda x: min(x, percentile_99))

        return df

    def minmax_scaler(self, df):  # Apply min-max scaler to the dataset
        df['Interarrival_scaled'] = self.scaler.fit_transform(
            df[['Interarrival']])

        return df

    def get_minmax_scaler(self):
        return self.scaler

    '''
    Function to create a dataset for training. The data is segmented into the memory window (input_window) and prediction window (output_window).
    Dataset returned is: train dataset (len = input_window), train dataset ground truth predictions (len = output_window) 
    '''

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

    # Split into training and testing sets
    def data_set_generator(self, mem_window_set, gd_truth_set, batch_size):
        train_size = int(len(mem_window_set) * 0.8)
        X_train, y_train = mem_window_set[:
                                          train_size], gd_truth_set[:train_size]
        X_test, y_test = mem_window_set[train_size:], gd_truth_set[train_size:]

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # print("X_train shape: ", X_test_tensor.size())

        # Create PyTorch dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # batch_size = 32
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
