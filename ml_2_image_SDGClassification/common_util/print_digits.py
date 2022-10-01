import matplotlib.pyplot as plt

def plot_digits(X, title, outpath):
    """Small helper function to plot 100 digits."""
    fig, axs = plt.subplots(nrows=20, ncols=10, figsize=(8, 8))
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((16, 16)), cmap="Greys")
        ax.axis("off")
    fig.suptitle(title, fontsize=24)
    fig.savefig(outpath)