import os
import random
import shutil


create_file_dir = 'D:\\tcav_data_file'

target_dir = "D:\\tiny-imagenet-200\\train"
file_list = []

for dir_name in os.listdir(target_dir):
    dir_name = os.path.join(dir_name, 'images')
    curr_target = os.path.join(target_dir, dir_name)
    temp_list = os.listdir(curr_target)
    for i in range(0, len(temp_list)):
        temp_list[i] = os.path.join(curr_target, temp_list[i])
    file_list.extend(temp_list)

random.shuffle(file_list)
os.chdir(create_file_dir)

for i in range(0, 501):
    temp_name = 'random500_'+str(i)
    if not os.path.exists(temp_name):
        os.mkdir(temp_name)
    curr_dir = os.path.join(create_file_dir, temp_name)
    counter = 0

    for j in range(i*50, (i+1)*50):
        shutil.copyfile(os.path.join(target_dir, file_list[j]), os.path.join(curr_dir, str(counter)+'.JPEG'))
        counter += 1

    print("Success: ", curr_dir)
