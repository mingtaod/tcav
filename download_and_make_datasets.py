import os
import argparse
from tensorflow import gfile
import imagenet_and_broden_fetcher as fetcher


def make_concepts_targets_and_randoms(source_dir, number_of_images_per_folder):

    broden_concepts = ['striped', 'dotted', 'zigzagged']

    # Make concepts from broden
    for concept in broden_concepts:
        fetcher.download_texture_to_working_folder(broden_path=os.path.join(source_dir, 'broden1_227'),
                                                   saving_path=source_dir,
                                                   texture_name=concept,
                                                   number_of_images=number_of_images_per_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create examples and concepts folders.')
    parser.add_argument('--source_dir', type=str, default='D:\\tcav_data_file',
                        help='Name for the directory where we will create the data.')
    parser.add_argument('--number_of_images_per_folder', type=int, default=50,
                        help='Number of images to be included in each folder')

    args = parser.parse_args()

    # Create folder if it doesn't exist
    if not gfile.Exists(args.source_dir):
        gfile.MakeDirs(os.path.join(args.source_dir))
        print("Created source directory at " + args.source_dir)

    # Make data
    make_concepts_targets_and_randoms(args.source_dir, args.number_of_images_per_folder)
    print("Successfully created data at " + args.source_dir)

# 思路：下载一个dataset，然后随机shuffle，从里面提取数据
