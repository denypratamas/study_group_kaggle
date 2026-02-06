# Study Group GDGoC Computer Vision

Proyek ini berisi eksperimen klasifikasi gambar untuk deteksi masker (Mask Detection) menggunakan Deep Learning dengan framework TensorFlow/Keras. Terdapat dua pendekatan utama yang diimplementasikan dalam notebook terpisah untuk membandingkan performa model.

## Dataset
Dataset yang digunakan adalah dataset gambar wajah bermasker dan tidak bermasker.
- **Split Dataset**: Data dibagi menjadi training set (80%) dan validation set (20%) secara stratified (seimbang antar kelas).
- **Ukuran Input**: 224x224 pixel (RGB).
- **Format**: Binary Classification (`with_mask` vs `without_mask`).

## Notebook 1: Custom CNN (`notebook.ipynb`)
Pendekatan pertama menggunakan arsitektur Convolutional Neural Network (CNN) yang dibangun dari awal (*scratch*).

### Preprocessing
- **Resizing**: Mengubah ukuran gambar menjadi 224x224.
- **Rescaling**: Normalisasi nilai pixel ke rentang [0, 1] (dibagi dengan 255.0).
- **Augmentasi Data**: Menggunakan `RandomFlip`, `RandomRotation`, `RandomZoom`, dan `RandomContrast` untuk memperkaya variasi data training.

### Arsitektur Model
Model dibangun menggunakan `Sequential` API dengan susunan layer sebagai berikut:
1. **Input Layer**: (224, 224, 3)
2. **Convolutional Block 1**: Conv2D (32 filter) + MaxPooling2D
3. **Convolutional Block 2**: Conv2D (64 filter) + MaxPooling2D
4. **Convolutional Block 3**: Conv2D (128 filter) + MaxPooling2D
5. **Convolutional Block 4**: Conv2D (128 filter) + MaxPooling2D
6. **Flatten**: Mengubah tensor menjadi vektor 1D
7. **Dense Layer**: 128 neuron (Activation: ReLU)
8. **Dropout**: 0.5 (untuk mengurangi overfitting)
9. **Output Layer**: 1 neuron (Activation: Sigmoid)

### Konfigurasi Training
- **Optimizer**: Adam (learning rate = 0.0001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

---

## Notebook 2: Transfer Learning MobileNetV2 (`notebook_mobilenetv2.ipynb`)
Pendekatan kedua menggunakan teknik **Transfer Learning** dengan base model **MobileNetV2** yang telah dilatih sebelumnya pada dataset ImageNet. Pendekatan ini umumnya menghasilkan akurasi yang lebih tinggi dan konvergensi yang lebih cepat.

### Preprocessing
- **Preprocessing Khusus**: Menggunakan fungsi `tf.keras.applications.mobilenet_v2.preprocess_input` yang menormalisasi input sesuai standar MobileNetV2 (rentang nilai pixel -1 hingga 1).
- **Augmentasi Data**: Sama seperti pendekatan pertama (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`).

### Arsitektur Model
1. **Base Model**: MobileNetV2 (pre-trained ImageNet, tanpa Top Layer, trainable=False)
2. **Global Average Pooling 2D**: Mengurangi dimensi spasial fitur.
3. **Dropout**: 0.3 (untuk regularisasi)
4. **Output Layer**: 1 neuron (Activation: Sigmoid)

### Konfigurasi Training
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

## Cara Menjalankan
1. Pastikan environment Python dan library (TensorFlow, Matplotlib, Pandas, Scikit-learn) sudah terinstall.
2. Letakkan dataset pada folder `Dataset/train` dan `Dataset/test`.
3. Jalankan notebook secara berurutan (Run All Cells).

## Sumber Dataset
[GDGoC Computer Vision](https://www.kaggle.com/competitions/gdgoc-telkom-university-2026-mask-classification/data)