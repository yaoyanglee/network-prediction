import pandas as pd
from scapy.all import rdpcap

# Read PCAP file using Scapy
packets = rdpcap('data/pcap/2024-dataset1/2024data_00000_20240101120000')

# Extract packet information
packet_data = []
for packet in packets:
    if 'IP' in packet:
        try:
            ip_layer = packet['IP']
            transport_layer = packet[ip_layer.payload.name]

            packet_info = {
                'Source IP': ip_layer.src,
                'Destination IP': ip_layer.dst,
                'Protocol': ip_layer.payload.name,
                'Source Port': transport_layer.sport if hasattr(transport_layer, 'sport') else None,
                'Destination Port': transport_layer.dport if hasattr(transport_layer, 'dport') else None,
                'Packet Size': len(packet)
            }
            packet_data.append(packet_info)

        except IndexError:
            packet_info = {
                'Source IP': ip_layer.src,
                'Destination IP': ip_layer.dst,
                'Protocol': ip_layer.payload.name,
                'Source Port': None,
                'Destination Port': None,
                'Packet Size': len(packet)
            }
            packet_data.append(packet_info)

# Convert to Pandas DataFrame
df = pd.DataFrame(packet_data)

# Drop rows with NaN or None values
df_cleaned = df.dropna()

# Filter rows where Protocol is TCP
tcp_df = df_cleaned[df_cleaned['Protocol'] == 'TCP']

# Display DataFrame with only TCP rows
print(tcp_df)
