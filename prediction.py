# USAGE
# python prediction.py --model path_to_trained_model --image path_to_input_image
# python prediction.py --model trained_ocr.model --image images/hello_world.png

from tensorflow.python.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import tensorflow as tf

# приймаємо аргументи
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input image")
ap.add_argument("-m", "--model", type = str, required = True, help = "path to trained model")
args = vars(ap.parse_args())

# завантажте нашу навчену нейрону мережу
print("[INFO] loading handwriting OCR model...")
#model = load_model(args["model"])
model = tf.keras.models.load_model(args["model"],custom_objects={'Functional':tf.keras.models.Model})

# завантажуємо вхідне зображення, перетворюємо його на відтінки сірого та розмиваємо
# це для зменшення шуму
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# виявлення країв, знаходження контурів 
# сортуємо отримані контури зліва направо
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method = "left-to-right")[0]

# ініціалізація списку обмежувальних рамок контуру
chars = []

# основний цикл
for c in cnts:
    # обчислюємо обмежувальну рамку контуру
    (x, y, w, h) = cv2.boundingRect(c)

    # Створюємо прямокутники та задаємо їм розмір
    if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
        # витягуємо символ і встановлюємо порогове значення, щоб символ виглядав як *білий* 
        # (передній план) на *чорному* тлі, а потім захопіть ширину та висоту зображення з порогом
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape

        # якщо ширина більше висоти, змінюємо розмір уздовж ширини
        if tW > tH: 
            thresh = imutils.resize(thresh, width=32)

        # інакше змінюємо розмір по висоті
        else:
            thresh = imutils.resize(thresh, height=32)

        # знову беремо розміри зображення (тепер, коли його розміри змінено), 
        # а потім визначаємо, скільки нам потрібно заповнити ширину та висоту, щоб наше зображення було 32x32
        (tH, tW) = thresh.shape
        dX = int(max(0, 32 - tW) / 2.0)
        dY = int(max(0, 32 - tH) / 2.0)

        # заповнюємо та налаштовуємо зображення розміри 32x32
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded = cv2.resize(padded, (32, 32))

        # підготуємо доповнене зображення для класифікації за допомогою нашої моделі OCR
        padded = padded.astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)

        # оновлюємо список символів, які будуть перевірятись моделю OCR
        chars.append((padded, (x, y, w, h)))

# витягуємо розташування обмежувальної рамки та доповнені символи
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype = "float32")

# Розпізнайте символи за допомогою нашої моделі розпізнавання рукописного тексту
preds = model.predict(chars)

# визначити список імених міток

labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames += "0123456789"
labelNames = [l for l in labelNames]

# об’єднайте передбачення та розташування обмежувальної рамки
for (pred, (x, y, w, h)) in zip(preds, boxes):
    # шукаємо індекс мітки з найбільшою відповідною ймовірністю, потім витягніть ймовірність і мітку
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]

    # малюємо передбачення на зображенні
    print("[INFO] {} - {:.2f}%".format(label, prob * 100))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

# Демонструємо зображення
cv2.imshow("Image", image)
cv2.waitKey(0)
