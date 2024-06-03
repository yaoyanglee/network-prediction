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

Features for RFR:
1. IAT 
2. Raw payload (Equals to number of IAT in most cases, need to somehow encode the entire packet into some value)
3. TCP Window Size (IAT and TCP window size always same) - Intuition is number of remaining frames to transmit?
4. source port (Need to encode, eg. One hot encode)
5. destination port (Need to encode, eg. One hot encode)
6. packet direction
7. IP Packet bytes
8. L4 Payload Bytes
'''


class IATDataParser:
    def __init__(self, data_path):
        self.scaler = MinMaxScaler()

        # Open the JSON file and load the JSON data into a Python dictionary
        with open(data_path, 'r') as file:
            self.biflow_data = json.load(file)

    def find_all_debug_set(self, min_iat_count=2000):
        iat_data = []

        for currentBiflow, biflow_data in self.biflow_data.items():
            biflowPacketData = biflow_data['packet_data']
            biflowFlowFeatures = biflow_data['flow_features']
            biflowFlowMetadata = biflow_data['flow_metadata']
            biflowIAT = biflowPacketData['iat']  # Extract interarrival times

            # and len(biflowIAT) <= 10000
            if len(biflowIAT) >= min_iat_count and len(biflowIAT) <= 5000:
                iat_data.append(biflowIAT)
                return pd.DataFrame({'Interarrival': iat_data})

        return iat_data

    def generate_debug_set(self, min_iat_count=2000):
        '''
        Returns the first biflow dictionary with at least `min_iat_count` interarrival times (iat).

        Parameters:
        min_iat_count (int): Minimum number of interarrival times required.

        Returns:
        dict: The first biflow dictionary containing at least `min_iat_count` interarrival times.
        '''

        iat_data = []

        # print("Biflow packet data: ",
        #       self.biflow_data['192.168.20.111,68,192.168.20.254,67,17']['packet_data'])
        # print('\n')
        # print("Biflow features: ",
        #       self.biflow_data['192.168.20.111,68,192.168.20.254,67,17']['flow_features'])
        # print('\n')
        # print("Biflow metadata: ",
        #       self.biflow_data['192.168.20.111,68,192.168.20.254,67,17']['flow_metadata'])

        for currentBiflow, biflow_data in self.biflow_data.items():
            biflowPacketData = biflow_data['packet_data']
            biflowFlowFeatures = biflow_data['flow_features']
            biflowFlowMetadata = biflow_data['flow_metadata']
            biflowIAT = biflowPacketData['iat']  # Extract interarrival times

            if len(biflowIAT) != len(biflowPacketData['L4_raw_payload']):
                print("IAT and payload length different number of elements")

            # print("TCP_win_size: ", biflowPacketData['TCP_win_size'])
            # print("len TCP_win_size: ", len(biflowPacketData['TCP_win_size']))
            if len(biflowIAT) != len(biflowPacketData['TCP_win_size']):
                print("IAT and TCP_win_size length different")

            if len(biflowPacketData['IP_packet_bytes']) != len(biflowPacketData['L4_payload_bytes']):
                print("IP_packet_bytes and L4_payload_bytes length different")

            # and len(biflowIAT) <= 10000
            if len(biflowIAT) >= min_iat_count and len(biflowIAT) <= 5000:
                iat_data.extend(biflowIAT)
                return pd.DataFrame({'Interarrival': iat_data})

        return None

    def aggregate_IAT(self):  # Aggregate all biflows in this JSON file
        all_iat_data = []

        for currentBiflow, biflow_data in self.biflow_data.items():
            biflowPacketData = biflow_data['packet_data']
            print()
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
            test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader


data_path = r"C:\Users\yylee\OneDrive\Desktop\yylee\network-pred\network-prediction\data\MIRAGE\MIRAGE-COVID-CCMA-2022\Raw_JSON\Teams\Teams\1619019338_com.microsoft.teams_mirage2020dataset_labeled_biflows_all_packets_encryption_metadata.json"

# Instantiate DataParser class
iat_data_parser = IATDataParser(data_path)
test_IAT = iat_data_parser.find_all_debug_set()
