import os 
import shutil
import random
import pickle
import numpy as np 

mode = 'Train'
if mode == 'Train': 
    N = 50000
else: 
    N = 30000
    
src_folder = '/vast/zz4330/Carla_JPG/{}'.format(mode)
files1 = os.listdir(os.path.join(src_folder, 'images'))
files2 = os.listdir(os.path.join(src_folder, 'targets'))

random.seed(2023)
selected = random.sample(files1, N)
selected.sort()

# store the images to a new folder
tar_folder = '/vast/lg154/Carla_JPG/{}'.format(mode) 
new_folder = os.path.join(tar_folder, 'sub_images')

if not os.path.exists(new_folder):
    os.makedirs(new_folder)
    print("Folder created successfully.")
else:
    print("Folder already exists.")

all_targets = []
filename = os.path.join(tar_folder, '{}_list.txt'.format(mode.lower()))
with open(filename, 'w') as f: 
    for item in selected: 
        shutil.copy(os.path.join(src_folder, 'images', item), new_folder)
        f.write(str(item) + '\n')
        print('have copied file {}'.format(item))
        
        idx = item.split('_')[1].split('.')[0]
        target_path = os.path.join(src_folder, 'targets', 'target_{}.npy'.format(idx))
        target = np.load(target_path)
        all_targets.append(target.reshape(1,-1))
        
all_targets = np.concatenate(all_targets, axis=0)
print('----target shape: {}'.format(all_targets.shape))

filename = os.path.join(tar_folder, 'sub_targets.pkl')
with open(filename, 'wb') as file:
    pickle.dump(all_targets, file)
print('---has saved target to {}'.format(filename))
        


