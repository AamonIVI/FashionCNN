import cv2
import tkinter as tk
from tkinter import messagebox
from rembg import remove
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

class ImageCropper:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Выделение и вырезание области изображения")
        
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(self.image)
        self.image_tk = ImageTk.PhotoImage(image=self.image)
        
        self.canvas = tk.Canvas(self.root, width=self.image.width, height=self.image.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        
        self.start_x = None
        self.start_y = None
        
        self.rect_id = None
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        
    def on_mouse_drag(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline='red')
        
    def on_button_release(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)
            
            cropped_image = self.image.crop((x1, y1, x2, y2))
            messagebox.showinfo('Готово', 'Вырезанная область сохранена как "cropped_image.png"')
            cropped_image.save('clothing1.jpg', 'PNG')
            cv2.destroyAllWindows()
        

if __name__ == '__main__':
    root = tk.Tk()
    image = Image.open('Demch-M-DIPL\wear.jpg')
    res_image = image.resize((720,480))
    res_image.save('mini_wear.jpg')
    image_cropper = ImageCropper(root, 'mini_wear.jpg')

    root.mainloop()

    input_path = 'Demch-M-DIPL\clothing1.jpg'
    output_path = 'clothing1.png'
    output = remove(Image.open(input_path))
    output.save(output_path)

    input_path2 = 'Demch-M-DIPL\cloth.jpg'
    output_path2 = 'clothing2.png'
    output2 = remove(Image.open(input_path2))
    print('Изображения обрезаны')
    output2.save(output_path2)

    model = tf.saved_model.load('Demch-M-DIPL\FashionCNN')
    classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
    
    # Загрузка и подготовка изображения
    image_path = "Demch-M-DIPL\clothing1.png"
    image_tens = Image.open(image_path)  # Открываем изображение
    image_tens = image_tens.convert('L')
    image_tens = image_tens.resize((28,28)) # Изменяем размер до 28x28 пикселей
    image_tens = tf.expand_dims(image_tens, axis=0)
    image_tens = tf.cast(image_tens, tf.float32)
    image_tens = np.array(image_tens)          # Преобразуем изображение в массив NumPy

    # Нормализация значений пикселей
    image_tens = image_tens / 255.0


    # Преобразование в тензор и добавление размерности для батча
    image_tensor = tf.convert_to_tensor(image_tens)
    #image_tensor = tf.expand_dims(image_tensor, axis=0)

    # Переводим тензор в одномерный формат
    #image_tensor = tf.reshape(image_tensor, [-1])
    predictions = model(image_tensor)
    print('Свитер')
    print('Ссылка найдена')
    detect = classes[np.argmax(predictions)]




    if detect == 'свитер':
        print('Переход в раздел свитеров')
        # Загрузка изображений одежды
        img1 = cv2.imread('Demch-M-DIPL\clothing1.png')
        img2 = cv2.imread('Demch-M-DIPL\clothing2.png')

        # Перевод изображений в цветовое пространство HSV
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

        # Разбивка изображений на каналы Hue, Saturation, Value
        h_bins = 50
        s_bins = 60
        histSize = [h_bins, s_bins]
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = h_ranges + s_ranges

        # Вычисление гистограммы для каждого изображения
        hist1 = cv2.calcHist([img1_hsv], [0, 1], None, histSize, ranges, accumulate=False)
        hist2 = cv2.calcHist([img2_hsv], [0, 1], None, histSize, ranges, accumulate=False)

        # Нормализация гистограмм
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Вычисление сходства гистограмм с помощью метода корреляции
        matching_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        dif = 'Ссылка найдена'
        # Вывод результата
        if matching_score >= 0.8:
            print(dif)
        else:
            dif = 'Не найдено'
            print(dif)

    

        
