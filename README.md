# PyQt5-Image-Processing-GUI-HW4

# Image Processing GUI with PyQt5

This project implements an image processing application with a graphical user interface (GUI) using **PyQt5**. The application allows users to:

- Open an image
- Convert the image into different color spaces (RGB, CMY, HSI, XYZ, Lab, YUV)
- Apply pseudo color maps
- Perform frequency-domain filtering (FFT, Ideal, Butterworth, Gaussian, Homomorphic)
- Visualize the results with interactive plots

### Features

- **Image Loading**: Users can load any image into the GUI for processing.
- **Color Space Conversion**: Convert the image to different color spaces, such as RGB, CMY, HSI, XYZ, Lab, and YUV.
- **Pseudo Color Maps**: Apply pseudo color maps to the image to enhance visualization.
- **Frequency Domain Filtering**:
  - **FFT (Fast Fourier Transform)**: View the image in the frequency domain.
  - **Ideal Filter**: Apply ideal low-pass or high-pass filtering.
  - **Butterworth Filter**: Apply a low-pass or high-pass Butterworth filter.
  - **Gaussian Filter**: Apply a Gaussian filter in the frequency domain.
  - **Homomorphic Filter**: Apply a homomorphic filter to enhance image details.
  
### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-processing-gui.git
   ```

2. Navigate to the project folder:
   ```bash
   cd image-processing-gui
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python main.py
   ```

### Usage

1. Open the application and load an image by clicking on the "Open Image" button.
2. Choose the desired operation from the options (e.g., color space conversion, frequency-domain filtering).
3. Adjust the filter parameters (e.g., cutoff frequency for Butterworth or Gaussian filters).
4. View the results in the display window.

### Dependencies

- **PyQt5**: For building the GUI
- **NumPy**: For numerical operations
- **OpenCV**: For image processing
- **Matplotlib**: For plotting and visualizing the results

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- PyQt5 documentation for GUI components
- OpenCV for image processing techniques
- Matplotlib for visualization and plotting

---

Let me know if you want to adjust or add anything specific!
