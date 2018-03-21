"""
Tools for plotting / visualization
"""

import matplotlib
matplotlib.use('Agg')  # no displayed figures -- need to call before loading pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import warnings

def is_square(shp, n_colors=1):
    """
    Test whether entries in shp are square numbers, or are square numbers after divigind out the
    number of color channels.
    """
    is_sqr = (shp == np.round(np.sqrt(shp))**2)
    is_sqr_colors = (shp == n_colors*np.round(np.sqrt(np.array(shp)/float(n_colors)))**2)
    return is_sqr | is_sqr_colors

def show_receptive_fields(theta, P=None, n_colors=None, max_display=100, grid_wa=None):
    """
    Display receptive fields in a grid. Tries to intelligently guess whether to treat the rows,
    the columns, or the last two axes together as containing the receptive fields. It does this
    by checking which axes are square numbers -- so you can get some unexpected plots if the wrong
    axis is a square number, or if multiple axes are. It also tries to handle the last axis
    containing color channels correctly.
    """

    shp = np.array(theta.shape)
    if n_colors is None:
        n_colors = 1
        if shp[-1] == 3:
            n_colors = 3
    # multiply colors in as appropriate
    if shp[-1] == n_colors:
        shp[-2] *= n_colors
        theta = theta.reshape(shp[:-1])
        shp = np.array(theta.shape)
    if len(shp) > 2:
        # merge last two axes
        shp[-2] *= shp[-1]
        theta = theta.reshape(shp[:-1])
        shp = np.array(theta.shape)
    if len(shp) > 2:
        # merge leading axes
        theta = theta.reshape((-1,shp[-1]))
        shp = np.array(theta.shape)
    if len(shp) == 1:
        theta = theta.reshape((-1,1))
        shp = np.array(theta.shape)

    # figure out the right orientation, by looking for the axis with a square
    # number of entries, up to number of colors. transpose if required
    is_sqr = is_square(shp, n_colors=n_colors)
    if is_sqr[0] and is_sqr[1]:
        warnings.warn("Unsure of correct matrix orientation. "
            "Assuming receptive fields along first dimension.")
    elif is_sqr[1]:
        theta = theta.T
    elif not is_sqr[0] and not is_sqr[1]:
        # neither direction corresponds well to an image
        # NOTE if you delete this next line, the code will work. The rfs just won't look very
        # image like
        return False

    theta = theta[:,:max_display].copy()

    if P is None:
        img_w = int(np.ceil(np.sqrt(theta.shape[0]/float(n_colors))))
    else:
        img_w = int(np.ceil(np.sqrt(P.shape[0]/float(n_colors))))
    nf = theta.shape[1]
    if grid_wa is None:
        grid_wa = int(np.ceil(np.sqrt(float(nf))))
    grid_wb = int(np.ceil(nf / float(grid_wa)))

    if P is not None:
        theta = np.dot(P, theta)

    vmin = np.min(theta)
    vmax = np.max(theta)

    for jj in range(nf):
        plt.subplot(grid_wa, grid_wb, jj+1)
        ptch = np.zeros((n_colors*img_w**2,))
        ptch[:theta.shape[0]] = theta[:,jj]
        if n_colors==3:
            ptch = ptch.reshape((n_colors, img_w, img_w))
            ptch = ptch.transpose((1,2,0)) # move color channels to end
        else:
            ptch = ptch.reshape((img_w, img_w))
        ptch -= vmin
        ptch /= vmax-vmin
        plt.imshow(ptch, interpolation='nearest', cmap=cm.Greys_r )
        plt.axis('off')

    return True


def plot_parameter(theta_in, base_fname_part1, base_fname_part2="", title = '', n_colors=None):
    """
    Save both a raw and receptive field style plot of the contents of theta_in.
    base_fname_part1 provides the mandatory root of the filename.
    """

    theta = np.array(theta_in.copy()) # in case it was a scalar
    print "%s min %g median %g mean %g max %g shape"%(
        title, np.min(theta), np.median(theta), np.mean(theta), np.max(theta)), theta.shape
    theta = np.squeeze(theta)
    if len(theta.shape) == 0:
        # it's a scalar -- make it a 1d array
        theta = np.array([theta])
    shp = theta.shape
    if len(shp) > 2:
        theta = theta.reshape((theta.shape[0], -1))
        shp = theta.shape

    ## display basic figure
    plt.figure(figsize=[8,8])
    if len(shp) == 1:
        plt.plot(theta, '.', alpha=0.5)
    elif len(shp) == 2:
        plt.imshow(theta, interpolation='nearest', aspect='auto', cmap=cm.Greys_r)
        plt.colorbar()

    plt.title(title)
    plt.savefig(base_fname_part1 + '_raw_' + base_fname_part2 + '.png')
    plt.close()

    ## also display it in basis function view if it's a matrix, or
    ## if it's a bias with a square number of entries
    if len(shp) >= 2 or is_square(shp[0]):
        if len(shp) == 1:
            theta = theta.reshape((-1,1))
        plt.figure(figsize=[8,8])
        if show_receptive_fields(theta, n_colors=n_colors):
            plt.suptitle(title + "receptive fields")
            plt.savefig(base_fname_part1 + '_rf_' + base_fname_part2 + '.png')
        plt.close()

def max_value(inputlist, index):
    return max([sublist[index] for sublist in inputlist])

def plot_2D(x, num_steps, filename):
    """
    plot 2D images
    """
    if num_steps == 1:
        x_0 = [sublist[0] for sublist in x]
        x_1 = [sublist[1] for sublist in x]
        plt.scatter(x_0, x_1)
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.savefig(filename + '.png')
        plt.close()
    else:
        for time in range(num_steps):
            plt.close()
            x_0 = [sublist[0] for sublist in x[time]]
            x_1 = [sublist[1] for sublist in x[time]]
            plt.scatter(x_0, x_1)
            plt.axis([-1.5, 1.5, -1.5, 1.5])
            plt.savefig(filename + '_step_'+ str(time)+'.png')
            plt.close()

def plot_grad(grad, filename):
    plt.close()
    rng = [(-1.5,1.5),(-1.5,1.5)]
    (x_beg, x_end), (y_beg, y_end) = rng
    for step in range(len(grad)):
        #X, Y, U, V = zip(*grad[step])
        start_0 = np.asarray([sublist[0] for sublist in grad[step]])
        start_1 = np.asarray([sublist[1] for sublist in grad[step]])
        end_0 = np.asarray([sublist[2] for sublist in grad[step]])
        end_1 = np.asarray([sublist[3] for sublist in grad[step]])

        speed = np.sqrt((end_0 - start_0)** 2 + (end_1 - start_1)** 2)
        UN = (end_0 - start_0)/ speed
        VN = (end_1 - start_1)/ speed
        plt.figure()
        plt.quiver(start_0, start_1, UN, VN, cmap = cm.winter,
                   headlength = 3,
                   clim = [0.,1.])
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        #plt.colorbar()
        plt.savefig(filename + '_step_'+ str(step)+'.png')
        plt.close()


def plot_images(X, fname):
    """
    Plot images in a grid.
    X is expected to be a 4d tensor of dimensions [# images]x[# colors]x[height]x[width]
    """
    ## plot
    # move color to end
    Xcol = X.reshape((X.shape[0],-1,)).T
    plt.figure(figsize=[8,8])
    if show_receptive_fields(Xcol, n_colors=X.shape[1]):
        plt.savefig(fname + '.png')
    else:
        warnings.warn('Images unexpected shape.')
    plt.close()

    ## save as a .npz file
    ##np.savez(fname + '.npz', X=X)
