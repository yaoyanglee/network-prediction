from IAT_dataparser import IATDataParser

data_path = "/home/ce-intern/Desktop/network_time_prediction/data/pcap/MIRAGE-COVID-CCMA-2022/Raw_JSON/Teams/1619005750_com.microsoft.teams_mirage2020dataset_labeled_biflows_all_packets_encryption_metadata.json"
parser = IATDataParser(data_path)

all_IAT = parser.generate_debug_set()
all_IAT = parser.saturate_99(all_IAT)
all_IAT = parser.minmax_scaler(all_IAT)

memory_window = 63
prediction_window = 1
iat_memory_window_set, iat_gd_truth_set = parser.create_sequences(
    all_IAT['Interarrival_scaled'], memory_window, prediction_window)  # Splitting data into memory_window and prediction_window (ground truth)


print(iat_memory_window_set[0])
print(iat_memory_window_set[1])
print(iat_memory_window_set[2])
print("length of initial dataset: ", len(all_IAT['Interarrival_scaled']))
print("length of all sequences: ", iat_memory_window_set.shape)

print(iat_gd_truth_set[0])
print(iat_gd_truth_set[1])
print(iat_gd_truth_set[2])
print("length of initial dataset: ", len(all_IAT['Interarrival_scaled']))
print("length of all sequences: ", iat_gd_truth_set.shape)
