# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:00:28 2023

@author: DMR
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Eğitilmiş modeli yükle
model = load_model('model.h5')

# Görüntüyü oku

image_path = input("bir fotoğraf yolu giriniz:")
image = cv2.imread(image_path)

# Giriş görüntüsünü yeniden boyutlandır
resized_image = cv2.resize(image, (224, 224))

# Giriş görüntüsünü normalize et
normalized_image = resized_image / 255.0

# Giriş görüntüsünü modele uygun hale getir
input_image = np.expand_dims(normalized_image, axis=0)

# Maske tahmini yap
mask_prediction = model.predict(input_image)

# Tahmin sonucunu sınıflara dönüştür
if mask_prediction[0][0] > mask_prediction[0][1]:
    label = 'Var'
    cv2.putText(image, label, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)# maske varsa yeşil
else:
    label = 'Yok'
    cv2.putText(image, label, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)#maske yoksa kırmızı ile fotoğraf üzerine yazıyor



# Sonucu görüntü üzerinde göster
#cv2.putText(image, label, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Maske Kontrolü', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
