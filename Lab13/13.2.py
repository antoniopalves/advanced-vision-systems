import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to read and filter events
def read_events(file_path, max_events=None, time_range=None):
    events = []
    with open(file_path, 'r') as file:
        for line in file:
            t, x, y, p = map(float, line.split())
            if max_events and len(events) >= max_events:
                break
            if time_range and (t < time_range[0] or t > time_range[1]):
                continue
            events.append((t, int(x), int(y), int(p)))
    return events

# Function to plot 3D event data
def plot_3d_events(events, title='3D Event Data', rotate_view=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    events = np.array(events)
    pos_events = events[events[:, 3] == 1]
    neg_events = events[events[:, 3] == 0]

    ax.scatter(pos_events[:, 1], pos_events[:, 2], pos_events[:, 0], c='r', label='Positive Polarity')
    ax.scatter(neg_events[:, 1], neg_events[:, 2], neg_events[:, 0], c='b', label='Negative Polarity')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Timestamp')
    ax.set_title(title)
    ax.legend()

    if rotate_view:
        ax.view_init(elev=20., azim=45)

    plt.show()

# Load events
events_file = r'c:\Users\anton\Documents\Erasmus\AVS\Lab13\events.txt'
first_8000_events = read_events(events_file, max_events=8000)
timestamp_05_to_1_events = read_events(events_file, time_range=(0.5, 1.0))

# Plot the first 8000 events
plot_3d_events(first_8000_events, title='First 8000 Events', rotate_view=True)

# Plot events with timestamp between 0.5 and 1
plot_3d_events(timestamp_05_to_1_events, title='Events with Timestamp 0.5 to 1', rotate_view=True)

# Answering questions
# How long is the sequence used during exercise 1.1 (in seconds)?
sequence_length = first_8000_events[-1][0] - first_8000_events[0][0]

# What’s the resolution of event timestamps?
timestamp_resolution = 1e-6  # Assuming microsecond resolution

# What does the time difference between consecutive events depend on?
# the time difference depends on the speed of changes in the scene being recorded

# What does positive/negative event polarity mean?
# positive polarity means an increase in brightness, negative polarity means a decrease

# What is the direction of movement of objects in exercise 1.2?
# the direction can be inferred from the slope and orientation of the event points in the 3D plot
