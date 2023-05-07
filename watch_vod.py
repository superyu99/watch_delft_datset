# import matplotlib.pyplot as plt

# with open('/workspace/mot/whatch_delft_dataset/view_of_delft/lidar/ImageSets/full.txt', 'r') as file:
#     lines = file.readlines()
#     frame_numbers = [int(line.strip()) for line in lines]

# plt.plot(frame_numbers, marker='o', linestyle='-')
# plt.xlabel('Index')
# plt.ylabel('Frame Number')
# plt.title('Frame Numbers')
# plt.grid(True)

# # Save the figure to a file instead of displaying it on the screen
# plt.savefig('/workspace/mot/whatch_delft_dataset/frame_numbers_plot.png', dpi=300, bbox_inches='tight')

import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg
import matplotlib.pyplot as plt

with open('/workspace/mot/whatch_delft_dataset/view_of_delft/lidar/ImageSets/full.txt', 'r') as file:
    lines = file.readlines()
    frame_numbers = [int(line.strip()) for line in lines]

plt.plot(frame_numbers, marker='o', linestyle='-')
plt.xlabel('Index')
plt.ylabel('Frame Number')
plt.title('Frame Numbers')
plt.grid(True)

# Draw horizontal lines at discontinuities
for i in range(1, len(frame_numbers)):
    if abs(frame_numbers[i] - frame_numbers[i - 1]) > 1:
        plt.axhline(y=frame_numbers[i - 1],  color='r', linestyle='--')

plt.show()