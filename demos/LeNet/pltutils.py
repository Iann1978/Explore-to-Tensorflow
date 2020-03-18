import matplotlib.pyplot as plt
import numpy as np

def func():
    print("func()")

    
def plot_images(images, cls_true, cls_pred=None):
    #assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])

        if cls_pred is None:
            xlabel = "True:{0}".format(cls_true[i])
        else:
            xlabel = "True:{0}, Pred:{1}".format(cls_true[i],cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])  

def plot_cifar10_images(images, cls_true, cls_pred=None):
    #assert len(images) == len(cls_true) == 9
    cifar10_lables = ['airplane','automobile','bird', 'cat', 'deer', 'dog', 'frog',
                     'horse', 'ship', 'truck']

    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])

        if cls_pred is None:
            xlabel = "True:{0}".format(cifar10_lables[int(cls_true[i])])
        else:
            xlabel = "True:{0}, Pred:{1}".format(cifar10_lables[cls_true[i]],
                         cifar10_lables[cls_pred[i]])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
    
def plot_error_images(images, cls_true, cls_pred):
    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    
    counter = 0;
    for i, ax in enumerate(axes.flat):
        while cls_true[counter] == cls_pred[counter] :
            counter = counter+1
        xlabel = "True:{0}, Pred:{1}".format(cls_true[counter],cls_pred[counter])
        ax.imshow(images[counter])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
        counter = counter+1
        
def plot_conv_output(values):
    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()