from keras.datasets import mnist
import numpy as np

def load_az_dataset(dataset_path): # завантажуємо літери (A-Z)
    # ініціалізація списку даних і міток
    data = []
    labels = []

    # проводимо цикл над рядками рукописного набору цифр від А до Z
    for row in open(dataset_path):
        # розібираємо мітку та зображення з рядка
        row = row.split(",")
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")

        # зображення представлені як одноканальні (відтінки сірого) зображення
        # їхні розміри 28x28=784 пікселів
        # 784 перетворюємо ці значення у матрицю(2гий ступінь вложеності списків) розміром 28x28 списків
        image = image.reshape((28, 28))

        # оновлюємо список даних і міток
        data.append(image)
        labels.append(label)

        # конвертувати дані і мітки в масиви NumPy
        data = np.array(data, dtype="float32")
        labels = np.array(labels, dtype="int")

        # повернути кортеж із даних і міток від А до Я
        return (data, labels)


def load_zero_nine_dataset(): # Завантажуємо цифри (0-9)
    # завантажуємо датасет MNIST і згруповуємо навчальні дані та дані тестування
    # разом (ми створимо власні навчальні та тестові розділи пізніше в проекті)
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])
    # повертаємо кортеж даних і міток MNIST
    return (data, labels)
