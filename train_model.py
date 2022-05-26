# -*- coding: utf-8 -*-

# Запуск навчання:
# python train_model.py --az dataset/a_z_handwritten_data.csv --model trained_ocr.model

# Імпорт модулів
import cv2
import argparse
from cv2 import moments
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
#from tensorflow.python.keras.optimizers import SGD
from keras.optimizer_experimental.sgd import SGD
from keras.preprocessing.image import ImageDataGenerator
from imutils import build_montages
from rich import print
from rich import pretty
pretty.install()
["Rich and pretty", True]

from models import ResNet
# from typing_extensions import Required
import matplotlib
matplotlib.use("Agg") # Налаштування бекенду в модулі matplotlib
from utils import load_az_dataset
from utils import load_zero_nine_dataset


# Створюємо введення аргументів у консолі
ap = argparse.ArgumentParser()

ap.add_argument("-a", "--az", required = True, help = "шлях до датасету")
ap.add_argument("-m", "--model", type = str, required = True, help = "шлях куди буде збережений файл із навченою мережею")
ap.add_argument("-p", "--plot", type = str, default = "plot.png", help = "шлях до графіку навчання мережі")

args = vars(ap.parse_args())

# ініціалізувати кількість епох для навчання, початкову швидкість навчання, і розмір партії
EPOCHS = 70
INIT_LR = 1e-1
BS = 128

# Завантажуэмо набори A-Z з бази данних MNIST
print("[bold blue][INFO] завантаження датасету...[/bold blue]")

(azData, azLabels) = load_az_dataset(args["az"])
(digitsData, digitsLabels) = load_zero_nine_dataset()

# набір даних MNIST займає мітки 0-9, тому додаємо 10 до кожної мітки A-Z, щоб переконатися, що символи A-Z не будуть неправильно позначені як цифри
azLabels += 10


# Складаємо дані із набору
data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])

#Змінюємо архітектуру зображення із 28x28 пікселів; на 32x32
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype = "float32")

# змінюємо інтенсивність пікселів зображень від [0, 255] до [0, 1]
data = np.expand_dims(data, axis=-1)
data /= 255.0

# перетворюємо мітки із цілих чисел у вектори
le = LabelBinarizer()

labels = le.fit_transform(labels)
ounts = labels.sum(axis = 0)

classTotals = labels.sum(axis = 0)
classWeight = {}

# Проводимо цикл по всіх класах і обчислюємо вагу класа
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# розділюємо дані на навчальні та тестові розділи, використовуючи 80% даних для навчання, а решта 20% для тестування
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=None, random_state=42)

# будуємо генератор зображень для збільшення даних
aug = ImageDataGenerator(rotation_range=10, zoom_range=0.05, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, horizontal_flip=False, fill_mode="nearest")

# ініціалізуємо та компілюємо нейронну мережу
print("[bold blue][INFO] компіляція мережі...[/bold blue]")

opt = SGD(learning_rate = INIT_LR, decay = INIT_LR / EPOCHS)
model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3), (64, 64, 128, 256), reg=0.0005)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"] )

# Починаємо навчання мережі
print("[bold blue][INFO] навчання мережі...[bold blue]")

H = model.fit(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, class_weight=classWeight, verbose=1)

# Створюємо почерговий список із іменнами навчальних символів
labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames += "0123456789"
labelNames = [l for l in labelNames]

# оцінюємо мережу
print("[bold blue][INFO] оцінка мережі...[/bold blue]")
predictions = model.predict(testX, batch_size = BS)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1)), target_names = labelNames)

# зберігаємо модель
print()
model.save(args["model"], save_format="h5")

# Будуємо та зберігаємо графік із історією навчання
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Trainning Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

images = [] 
# випадковим чином вибераємо кілька тестових символів
for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
    # класифікація цих символів
    probs = model.predict(testX[np.newaxis, i])
prediction = probs.argmax(axis=1)
label = labelNames[prediction[0]]

# витягуємо зображення із тестових даних та ініціалізуйте колір текстового символу зеленим (правильну відповідь)
image = (testX[i] * 255).astype("uint8")
color = (0, 255, 0)

# У іншому випідку передбачення мітки класу є неправильним
if prediction[0] != np.argmax(testY[i]):
    color = (0, 0, 255)

# об’єднуємо канали в одне зображення, змінюємо розмір зображення з 32x32 до 96x96 щоб ми могли краще його бачити, і малюємо мітку на зображенні
image = cv2.merge([image] * 3)
image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

# додатємо зображення до нашого списку вихідних зображень
images.append(image)

# зєднюємо усі зображення на одне 
montage = build_montages(images, (96, 96), (7, 7))[0]

# демонструємо зображення
cv2.imshow("OCR Results", montage)
cv2.imwrite("Results.png", montage)
cv2.waitKey(0)