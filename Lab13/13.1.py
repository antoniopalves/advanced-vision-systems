import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 4: Read and process events.txt
def read_events(file_path):
    timestamps = []
    x_coords = []
    y_coords = []
    polarities = []

    with open(file_path, 'r') as file:
        for line in file:
            timestamp, x, y, polarity = line.strip().split()
            timestamps.append(float(timestamp))
            x_coords.append(int(x))
            y_coords.append(int(y))
            polarities.append(int(polarity))

    return np.array(timestamps), np.array(x_coords), np.array(y_coords), np.array(polarities)

# Step 5: Parse events
file_path = r'c:\Users\anton\Documents\Erasmus\AVS\Lab13\events.txt'
timestamps, x_coords, y_coords, polarities = read_events(file_path)

# Step 6: Filter events with timestamps less than 1 second
filtered_indices = timestamps < 1.0
timestamps = timestamps[filtered_indices]
x_coords = x_coords[filtered_indices]
y_coords = y_coords[filtered_indices]
polarities = polarities[filtered_indices]

# Step 7: Split events into individual variables
num_events = len(timestamps)

# Step 8: Analyze and print event details
print(f'Number of events: {num_events}')
print(f'Timestamps (first 10): {timestamps[:10]}')
print(f'X coordinates (first 10): {x_coords[:10]}')
print(f'Y coordinates (first 10): {y_coords[:10]}')

# Correct polarity distribution analysis
positive_events = np.sum(polarities == 1)
negative_events = np.sum(polarities == -1)
print(f'Number of positive events: {positive_events}')
print(f'Number of negative events: {negative_events}')

# Step 9: Visualize event data with a 3D chart
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use different colors for positive and negative events
pos_indices = polarities == 1
neg_indices = polarities == -1

ax.scatter(x_coords[pos_indices], y_coords[pos_indices], timestamps[pos_indices], c='r', marker='o', label='Positive')
ax.scatter(x_coords[neg_indices], y_coords[neg_indices], timestamps[neg_indices], c='b', marker='x', label='Negative')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Timestamp')
ax.legend()

plt.show()
