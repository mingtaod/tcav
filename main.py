from tcav import utils
import tensorflow as tf
import torch
import model
import activation_generator
import tcav
import utils_plot

if __name__ == '__main__':
    # Name of your model wrapper
    model_to_run = 'ResNet50Class8'

    # Name of parent directory that results are stored
    project_name = 'tcav_class_test'
    working_dir = 'D:\\tcav_working_dir'

    # Where activations are stored
    activation_dir = working_dir + '/activations/'

    # Where CAVs are stored
    cav_dir = working_dir + '/cavs/'

    # Where the images live
    source_dir = 'D:\\tcav_data_file'
    # bottlenecks = ['layer1', 'layer2', 'layer3']  # @param # Reduced to save time
    bottlenecks = ['layer2']

    utils.make_dir_if_not_exists(working_dir)
    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(cav_dir)

    LABEL_PATH = './imagenet_comp_graph_label_strings.txt'

    # This is a regularizer penalty parameter for linear classifier to get CAVs
    alphas = [0.1]

    target = 'cat'
    # concepts = ["dotted", "striped", "zigzagged"] # Reduced to save time
    concepts = ['striped']
    random_counterpart = 'random500_500'
    my_model = model.SmallResNet50Wrapper(LABEL_PATH)
    act_generator = activation_generator.ImageActivationGenerator(my_model, source_dir, activation_dir, max_examples=40)
    num_random_exp = 100

    mytcav = tcav.TCAV(target,
                       concepts,
                       bottlenecks,
                       act_generator,
                       alphas,
                       random_counterpart=random_counterpart,
                       cav_dir=cav_dir,
                       num_random_exp=num_random_exp,
                       random_concepts=None)

    results = mytcav.run()

    utils_plot.plot_results(results, num_random_exp=num_random_exp)


# Number of parameters = num_bottlenecks * (num_concepts_to_test + 1) * num_random_exp
# num_concepts_to_test + 1 -- this means that we need to also consider the random counterpart
