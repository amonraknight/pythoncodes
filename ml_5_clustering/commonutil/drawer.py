import matplotlib.pyplot as plt

def drawDigits(digit_images, k):
    plt.figure(figsize=(8, 2))
    for index, eachImage in enumerate(digit_images):
        plt.subplot(k // 10, 10, index + 1)
        plt.imshow(eachImage.reshape(8, 8), cmap="binary", interpolation="bilinear")
        plt.axis('off')

    plt.show()
