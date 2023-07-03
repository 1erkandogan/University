import matplotlib.pyplot as plt

# Packet arrival times and service times
arrival_times = [1, 3, 4, 7, 8, 15]
service_times = [3.5, 4, 2, 1, 1.5, 4]

# Function to calculate the number of packets in the system at time t


def num_packets(t, fcfs=True):
    packets = []
    for i in range(len(arrival_times)):
        # Check if the packet is still in the system at time t
        if t >= arrival_times[i]:
            # Calculate the time the packet spent in the system
            time_in_system = t - arrival_times[i]
            # Calculate the time remaining to serve the packet
            time_remaining = service_times[i] - time_in_system
            if fcfs:
                # For FCFS, add the packet to the list of packets in the system
                packets.append(time_remaining)
            else:
                # For LCFS, insert the packet at the front of the list
                packets.insert(0, time_remaining)
    # Return the length of the list as the number of packets in the system
    return len(packets)


# Create a time array from 0 to 20 seconds
t = list(range(0, 21, 0.5))

# Calculate the number of packets in the system for FCFS
n_fcfs = [num_packets(i, fcfs=True) for i in t]

# Calculate the number of packets in the system for LCFS
n_lcfs = [num_packets(i, fcfs=False) for i in t]

# Create subplots for FCFS and LCFS
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

# Plot the results for FCFS
ax1.scatter(t, n_fcfs, label='FCFS', color='blue')
ax1.set_ylabel('Number of packets')
ax1.set_title('FCFS vs. LCFS Queueing System')
ax1.legend()

# Plot the results for LCFS
ax2.scatter(t, n_lcfs, label='LCFS', color='orange')
ax2.set_xlabel('Time')
ax2.set_ylabel('Number of packets')
ax2.legend()

plt.show()
