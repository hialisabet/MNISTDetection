import numpy as np
import matplotlib.pyplot as plt


# Loading MNSIT dataset
def loadMNIST(prefix, folder):
    intType = np.dtype("int32").newbyteorder(">")
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(folder + "/" + prefix + "-images.idx3-ubyte", dtype="ubyte")
    magicBytes, nImages, width, height = np.frombuffer(
        data[:nMetaDataBytes].tobytes(), intType
    )
    data = data[nMetaDataBytes:].astype(dtype="int").reshape([nImages, width, height])

    labels = np.fromfile(folder + "/" + prefix + "-labels.idx1-ubyte", dtype="ubyte")[
             2 * intType.itemsize:
             ]

    return data, labels


trainingImages, trainingLabels = loadMNIST("train", "mnist/")
testImages, testLabels = loadMNIST("t10k", "mnist/")

# Seprating digits on train data
train_data = {}
for i in range(10):
    train_data[i] = np.where(trainingLabels == i)[0]

# Creating H matrixes
H = {}
for i in range(10):
    H[i] = np.zeros(shape=(len(train_data[i]), 28 * 28))

# Adding each image as a row in H[i]
for i in range(10):
    for j, s in enumerate(train_data[i]):
        H[i][j] = trainingImages[s].flatten()

# Transposing H
for i in range(10):
    H[i] = H[i].T


# Solving least sqaure using SVD and returning predicted digit
def predict_digit(i):
    z = testImages[i].flatten()
    q = np.zeros(10)
    for i in range(10):
        U, S, VT = np.linalg.svd(H[i], full_matrices=False)
        U_ = np.linalg.pinv(U)
        alpha = U_ @ z
        q[i] = np.sqrt(np.sum(np.square(U @ alpha - z)))

    return np.uint8(np.argmin(q))


# Choosing a sample to test
sample_size = 10
random_sample = np.random.choice(testImages.shape[0], sample_size, replace=False)

# Testing the sample
sums = 0
for i in random_sample:
    predicted = predict_digit(i)
    print("Predicted: ", predicted, ", actual label is ", testLabels[i])
    plt.imshow(testImages[i])
    plt.show()
    if (predicted == testLabels[i]):
        sums += 1

print(f"{sums * 100 / sample_size}%: True guesses were {sums} out of {sample_size} images...")

