import os

path = '/workspace/mot/whatch_delft_dataset/view_of_delft/lidar/training/label_2'
empty_files = []

for file in os.listdir(path):
    if file.endswith('.txt'):
        file_path = os.path.join(path, file)
        with open(file_path, 'r') as f:
            content = f.read()
            if not content.strip():
                empty_files.append(file)

print('Empty txt files:')
for empty_file in empty_files:
    print(empty_file)