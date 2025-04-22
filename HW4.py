import sys
import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QLineEdit, QLabel, QPushButton)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 

class HW2App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.image_data = None

        self.setWindowTitle('HW4')
        self.resize(1400, 900)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 1000, 900)
        self.view = QtWidgets.QGraphicsView(self)
        self.view.setGeometry(0, 0, 1000, 900)
        self.view.setScene(self.scene)
        self.view.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.white))

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1140, 150)
        self.btn_open_file.resize(110, 40)
        self.btn_open_file.setText('開啟圖片')
        self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.open_file)
        
        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1140, 215)
        self.btn_open_file.resize(110, 40)
        self.btn_open_file.setText('FFT')
        self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.fft)

        self.mylabel_Ideal_filter = QLabel('D0 : ', self)
        self.mylabel_Ideal_filter.setFont(QtGui.QFont("Arial", 16))
        self.mylabel_Ideal_filter.move(1030, 320)
        self.mylabel_Ideal_filter.resize(110, 40)
        self.mylineedit_Ideal_filter= QLineEdit(self)
        self.mylineedit_Ideal_filter.setStyleSheet("background-color: white; color: black;")
        self.mylineedit_Ideal_filter.setPlaceholderText(' 請輸入D0:')
        self.mylineedit_Ideal_filter.move(1070, 320)
        self.mylineedit_Ideal_filter.resize(120, 40)

        self.mybutton_Ideal_filter = QPushButton('Ideal', self)
        self.mybutton_Ideal_filter.move(1020, 380)
        self.mybutton_Ideal_filter.resize(110, 40)
        self.mybutton_Ideal_filter.clicked.connect(self.Ideal_filter)
        self.mylineedit_Ideal_filter.returnPressed.connect(self.Ideal_filter)

        self.mylabel_n = QLabel('n : ', self)
        self.mylabel_n.setFont(QtGui.QFont("Arial", 16))
        self.mylabel_n.move(1220, 320)
        self.mylabel_n.resize(110, 40)
        self.mylineedit_n= QLineEdit(self)
        self.mylineedit_n.setStyleSheet("background-color: white; color: black;")
        self.mylineedit_n.setPlaceholderText(' 請輸入n:')
        self.mylineedit_n.move(1240, 320)
        self.mylineedit_n.resize(120, 40)

        self.mybutton_n = QPushButton('Butterworth', self)
        self.mybutton_n.move(1260, 380)
        self.mybutton_n.resize(110, 40)
        self.mybutton_n.clicked.connect(self.Butterworth_filter)
        self.mylineedit_n.returnPressed.connect(self.Butterworth_filter)

        self.box = QtWidgets.QComboBox(self)   
        self.box.addItems(['low','high'])   
        self.box.setGeometry(1000,275,400,30)
        self.box.currentIndexChanged.connect(self.Ideal_filter)
        
        self.mybutton_local = QPushButton('Gaussian', self)
        self.mybutton_local.move(1135, 380)
        self.mybutton_local.resize(110, 40)
        self.mybutton_local.clicked.connect(self.Gaussian_filter)

        self.mylabel_gamma_high = QLabel('gamma_high : ', self)
        self.mylabel_gamma_high.setFont(QtGui.QFont("Arial", 16))
        self.mylabel_gamma_high.move(1010, 460)
        self.mylabel_gamma_high.resize(110, 40)
        self.mylineedit_gamma_high= QLineEdit(self)
        self.mylineedit_gamma_high.setStyleSheet("background-color: white; color: black;")
        self.mylineedit_gamma_high.setPlaceholderText(' 請輸入gamma_high :')
        self.mylineedit_gamma_high.move(1135, 460)
        self.mylineedit_gamma_high.resize(120, 40)

        self.mylabel_gamma_low = QLabel('gamma_low : ', self)
        self.mylabel_gamma_low.setFont(QtGui.QFont("Arial", 16))
        self.mylabel_gamma_low.move(1010, 520)
        self.mylabel_gamma_low.resize(110, 40)
        self.mylineedit_gamma_low= QLineEdit(self)
        self.mylineedit_gamma_low.setStyleSheet("background-color: white; color: black;")
        self.mylineedit_gamma_low.setPlaceholderText(' 請輸入gamma_low :')
        self.mylineedit_gamma_low.move(1135, 520)
        self.mylineedit_gamma_low.resize(120, 40)

        self.mylabel_D0 = QLabel('D0 : ', self)
        self.mylabel_D0.setFont(QtGui.QFont("Arial", 16))
        self.mylabel_D0.move(1080, 580)
        self.mylabel_D0.resize(110, 40)
        self.mylineedit_D0= QLineEdit(self)
        self.mylineedit_D0.setStyleSheet("background-color: white; color: black;")
        self.mylineedit_D0.setPlaceholderText(' 請輸入D0 :')
        self.mylineedit_D0.move(1135, 580)
        self.mylineedit_D0.resize(120, 40)

        self.mybutton_c = QPushButton('homomorphic', self)
        self.mybutton_c.move(1270, 515)
        self.mybutton_c.resize(120, 50)
        self.mybutton_c.clicked.connect(self.homomorphic_filter)

        self.btn_close = QtWidgets.QPushButton(self)
        self.btn_close.setText('Image_blurring')
        self.btn_close.setGeometry(1135, 650, 100, 40)
        self.btn_close.resize(120, 40)
        self.btn_close.clicked.connect(self.Image_blurring)
        
        self.btn_close = QtWidgets.QPushButton(self)
        self.btn_close.setText('關閉')
        self.btn_close.setGeometry(1135, 750, 100, 30)
        self.btn_close.resize(110, 40)
        self.btn_close.clicked.connect(self.closeFile)

        self.output_height = 0
        self.output_width = 0
        self.filter_size = None

    def closeFile(self):
        ret = QtWidgets.QMessageBox.question(self, 'question', '確定關閉視窗？')
        if ret == QtWidgets.QMessageBox.Yes:
            app.quit()
        else:
            return

    def open_file(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
        if file_name:
            self.image_data = cv.imread(file_name)
            self.image_data = cv.cvtColor(self.image_data, cv.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(8, 6))
            self.pic = ax.imshow(self.image_data)
            ax.set_xticks([])
            ax.set_yticks([])
            canvas = FigureCanvas(fig)
            self.view.setAlignment(QtCore.Qt.AlignCenter)
            self.scene.clear()
            self.scene.addWidget(canvas)

    def fft(self):
        image = cv.cvtColor(self.image_data, cv.COLOR_BGR2GRAY)

        start_time = time.time()
        f_transform_shifted = np.fft.fftshift(np.fft.fft2(image))
        end_time = time.time()

        f_transform_spectrum = np.log(np.abs(f_transform_shifted) + 1)
        
        f_transform_255 = (f_transform_spectrum - np.min(f_transform_spectrum)) / (np.max(f_transform_spectrum) - np.min(f_transform_spectrum)) * 255 
    
        original_phase_spectrum = np.angle(f_transform_shifted)

        original = np.fft.ifft2(f_transform_shifted)

        Fmin = np.log(np.abs(f_transform_spectrum).min() + 1)
        Fmax = np.log(np.abs(f_transform_spectrum).max() + 1)
        
        Ynew = 255 * (np.log1p(1 + np.abs(f_transform_spectrum)) - Fmin) / (Fmax - Fmin)
        
        Ynew = Ynew * (np.exp(1j * original_phase_spectrum))
        
        processed_phase_spectrum = np.angle(Ynew)
        
        processed = np.fft.ifft2(np.fft.ifftshift(Ynew))
   
        # processed = np.clip(processed, a_min=0, a_max=None)

        processed = (processed - np.min(processed)) / (np.max(processed) - np.min(processed)) * 255

        
        fig, ax = plt.subplots(2, 3, figsize=(10, 8))
        ax[0, 0].imshow(np.abs(f_transform_255), cmap='gray')
        ax[0, 0].set_title(f' computation : {end_time - start_time:.2f} seconds\n  FFT Magnitude')

        ax[0, 1].imshow(original_phase_spectrum, cmap='gray')
        ax[0, 1].set_title('Original Phase')

        ax[0, 2].imshow(np.abs(original), cmap='gray')
        ax[0, 2].set_title('Original iFFT')

        ax[1, 0].imshow(abs(Ynew), cmap='gray')
        ax[1, 0].set_title('Processed FFT Magnitude')

        ax[1, 1].imshow(processed_phase_spectrum, cmap='gray')
        ax[1, 1].set_title('Processed Phase')

        ax[1, 2].imshow(np.abs(processed), cmap='gray')
        ax[1, 2].set_title('Processed iFFT')

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

    def Ideal_filter(self):
        self.scene.clear()
        D = int(self.mylineedit_Ideal_filter.text())
        selected_option = self.box.currentText()
        image = cv.cvtColor(self.image_data, cv.COLOR_BGR2GRAY)

        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)

        rows, cols = image.shape
        center_row, center_col = rows // 2, cols // 2
        H_low = np.zeros_like(image, dtype=complex)
        H_low[center_row - D:center_row + D, center_col - D:center_col + D] = 1

        H_high = np.ones_like(image, dtype=complex)
        H_high[center_row - D:center_row + D, center_col - D:center_col + D] = 0

        if selected_option == 'low' :
            filtered_f_transform = f_transform_shifted * H_low
        else:
            filtered_f_transform = f_transform_shifted * H_high

        filtered_image = np.fft.ifftshift(filtered_f_transform)
        filtered_image = np.fft.ifft2(filtered_image)
        filtered_image = np.abs(filtered_image)

        fig, ax = plt.subplots(1, 2, figsize=(10, 8))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original_image')
        ax[0].axis('off')

        ax[1].imshow(filtered_image, cmap='gray')
        ax[1].set_title('Ideal_filter_image')
        ax[1].axis('off')

        plt.subplots_adjust(wspace=0.5) 

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)
    
    def Butterworth_filter(self):
        self.scene.clear()
        D0 = float(self.mylineedit_Ideal_filter.text())
        n = int(self.mylineedit_n.text())
        selected_option = self.box.currentText()
        image = cv.cvtColor(self.image_data, cv.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)

        # 定義巴特沃斯濾波器
        rows, cols = image.shape
        center_row, center_col = rows // 2, cols // 2
        H = np.zeros_like(image, dtype=complex)

        if selected_option == 'low' :
            for i in range(rows):
                for j in range(cols):
                    distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
                    H[i, j] = 1 / (1 + (distance / D0)**(2 * n))

        else:
            for i in range(rows):
                for j in range(cols):
                    distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
                    H[i, j] = 1 / (1 + (D0 / distance) ** (2 * n))

        filtered_f_transform = f_transform_shifted * H

        filtered_image = np.fft.ifftshift(filtered_f_transform)
        filtered_image = np.fft.ifft2(filtered_image)
        filtered_image = np.abs(filtered_image)

        filtered_image = (filtered_image - np.min(filtered_image)) / (np.max(filtered_image) - np.min(filtered_image) ) * 255
        filtered_image = filtered_image.astype(np.uint8)

        fig, ax = plt.subplots(1, 2, figsize=(10, 8))

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original_image')
        ax[0].axis('off')

        ax[1].imshow(np.abs(filtered_image), cmap='gray')
        ax[1].set_title('Butterworth_filter_image')
        ax[1].axis('off')

        plt.subplots_adjust(wspace=0.5) 

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

    def Gaussian_filter(self):
        self.scene.clear()
        D0 = float(self.mylineedit_Ideal_filter.text())
        selected_option = self.box.currentText()
        image = cv.cvtColor(self.image_data, cv.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)

        rows, cols = image.shape
        
        if selected_option == 'low' :
            print(selected_option)
            H = np.zeros_like(image, dtype=complex)
            for i in range(rows):
                for j in range(cols):
                    d = np.sqrt((i - rows // 2) ** 2 + (j - cols // 2) ** 2)
                    print(d)
                    H[i, j] = np.exp(-d**2 / (2 * D0**2))
                    print(H[i, j])
        else:
            H = np.zeros_like(image, dtype=complex)
            for i in range(rows):
                for j in range(cols):
                    d = np.sqrt((i - rows // 2) ** 2 + (j - cols // 2) ** 2)
                    H[i, j] = 1 - np.exp(-d**2 / (2 * D0**2))

        # 在頻率域中應用濾波器
        filtered_f_transform = f_transform_shifted * H

        filtered_image = np.fft.ifftshift(filtered_f_transform)
        filtered_image = np.fft.ifft2(filtered_image)
        filtered_image = np.abs(filtered_image)

        fig, ax = plt.subplots(1, 2, figsize=(10, 8))

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original_image')
        ax[0].axis('off')

        ax[1].imshow(np.abs(filtered_image), cmap='gray')
        ax[1].set_title('Gaussian_filter_image')
        ax[1].axis('off')

        plt.subplots_adjust(wspace=0.5) 

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

    def homomorphic_filter(self):
        self.scene.clear()
        image = cv.cvtColor(self.image_data, cv.COLOR_BGR2GRAY)
        gamma_high = float(self.mylineedit_gamma_high.text()) if self.mylineedit_gamma_high.text() != '' else 3.0
        gamma_low = float(self.mylineedit_gamma_low.text()) if self.mylineedit_gamma_low.text() != '' else 0.4
        D0 = float(self.mylineedit_D0.text()) if self.mylineedit_D0.text() != '' else 20
        print(gamma_high, gamma_low, D0)

        rows, cols = image.shape
        H = np.zeros((rows, cols), dtype=complex)
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - rows // 2) ** 2 + (j - cols // 2) ** 2)
                H[i, j] = (gamma_high - gamma_low)*(1 - np.exp(-5*d**2/(D0**2))) + gamma_low

        image_log = np.log1p(image).astype(np.float64)
        
        f_transform_shifted = np.fft.fftshift(np.fft.fft2(image_log))

        filtered_fft_image = H * f_transform_shifted

        # 執行逆傅立葉變換
        filtered_image = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fft_image)))

        enhanced_image = np.expm1(filtered_image).astype(np.uint8)

        fig, ax = plt.subplots(1, 2, figsize=(10, 8))

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original_image')
        ax[0].axis('off')

        ax[1].imshow(filtered_image, cmap='gray')
        ax[1].set_title('homomorphic_filter_image')
        ax[1].axis('off')

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

    def Image_blurring(self):
            self.scene.clear()
            image = cv.cvtColor(self.image_data, cv.COLOR_BGR2GRAY)
            f_transform = np.fft.fft2(image)
            f_transform_shifted = np.fft.fftshift(f_transform)
            r, c = image.shape
            a = 0.1
            b = 0.1
            T = 1
            H = np.zeros_like(image, dtype=complex)
            for u in range(r):
                for v in range(c):
                    o = u - r//2
                    p = v - c//2
                    move = np.pi * (o * a + p * b)
                    if (o * a + p * b) != 0:
                        H[u, v] = T * (np.sin(move)) * (np.exp(-1j * move))/ move
                    else:
                        H[u, v] = 1

            filtered_fft_image = H * f_transform_shifted 

            enhanced_image = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fft_image)))

            enhanced_image = (enhanced_image - np.min(enhanced_image)) / (np.max(enhanced_image) - np.min(enhanced_image)) * 255

            enhanced_image = np.uint8(enhanced_image)

            inverse_filter =  f_transform_shifted
            
            inverse_filter = np.fft.ifft2(np.fft.ifftshift(inverse_filter))

            inverse_filter = np.clip(inverse_filter, a_min=0, a_max=None)

            inverse_filter = (inverse_filter - np.min(inverse_filter)) / (np.max(inverse_filter) - np.min(inverse_filter)) * 255

            inverse_filter = np.uint8(inverse_filter)

            Sf = H ** 2
            K  = 0.00001                                     

            # Wiener濾波
            Wiener_Filtering = np.real(np.fft.ifft2((1/H) * Sf / ((Sf + K)) * filtered_fft_image))
            Wiener_Filtering = np.clip(Wiener_Filtering, a_min = 0, a_max = None)
            
            Wiener_Filtering = (Wiener_Filtering - np.min(Wiener_Filtering)) / (np.max(Wiener_Filtering) - np.min(Wiener_Filtering)) * 255
            
            Wiener_Filtering = np.uint8(Wiener_Filtering)

            noise = np.random.normal(0, 400, image.shape).astype(np.uint8)
            noise_image = cv.add(enhanced_image, noise)

            noise_image_fft = np.real(np.fft.ifftshift(np.fft.fft2(noise_image)))
            
            inverse_noise = np.fft.ifft2(np.fft.ifftshift(enhanced_image))

            inverse_noise = np.clip(inverse_filter, a_min=0, a_max=None)

            inverse_noise = (inverse_filter - np.min(inverse_noise)) / (np.max(inverse_filter) - np.min(inverse_filter)) * 255

            inverse_noise = np.uint8(inverse_filter)

            Sf_noise = np.fft.fftshift(np.fft.fft2((noise))) ** 2
            K_noise = 0.000001

            Wiener_noise = np.real(np.fft.ifft2((1/np.fft.fftshift(np.fft.fft2(noise))) * Sf_noise / ((Sf_noise + K_noise)) * noise_image_fft))
            Wiener_noise = np.clip(Wiener_noise, a_min=0, a_max=None)
            
            Wiener_noise = (Wiener_noise - np.min(Wiener_noise)) / (np.max(Wiener_noise) - np.min(Wiener_noise)) * 255
            print(Wiener_noise)
            Wiener_noise = np.uint8(Wiener_noise)

            fig, ax = plt.subplots(2, 4, figsize=(10, 8))

            ax[0, 0].imshow(image, cmap='gray')
            ax[0, 0].set_title('Original_image')
            ax[0, 0].axis('off')

            ax[0, 1].imshow(enhanced_image, cmap='gray')
            ax[0, 1].set_title('Image_blurring_image')
            ax[0, 1].axis('off')

            ax[0, 2].imshow(inverse_filter, cmap='gray')
            ax[0, 2].set_title('inverse_filter_image')
            ax[0, 2].axis('off')

            ax[0, 3].imshow(Wiener_Filtering, cmap='gray')
            ax[0, 3].set_title('Wiener_Filtering_image')
            ax[0, 3].axis('off')

            ax[1, 0].imshow(image, cmap='gray')
            ax[1, 0].set_title('Original_image')
            ax[1, 0].axis('off')
            
            ax[1, 1].imshow(noise_image, cmap='gray')
            ax[1, 1].set_title('noisy_image_image')
            ax[1, 1].axis('off')

            ax[1, 2].imshow(inverse_noise, cmap='gray')
            ax[1, 2].set_title('inverse_noisy_image')
            ax[1, 2].axis('off')

            ax[1, 3].imshow(Wiener_noise, cmap='gray')
            ax[1, 3].set_title('Wiener_noise_image')
            ax[1, 3].axis('off')
            
            canvas = FigureCanvas(fig)
            self.view.setAlignment(QtCore.Qt.AlignCenter)
            self.scene.addWidget(canvas)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = HW2App()
    ex.show()
    sys.exit(app.exec_())