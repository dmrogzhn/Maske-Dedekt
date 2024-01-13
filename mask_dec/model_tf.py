# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 12:47:47 2023

@author: DMR
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os



# veri yolu ve içindeki dosyalara göre class belirtildi
path= "dataset"
classes = ["maskeli", "maskesiz"]


INIT_LR = 1e-4  # başlangıç öğrenme hızı
EPOCHS = 20     # eğitim için kullanılacak katman sayısı
BS = 32         #eğitim setini parçalayarak öğrenmesini sağlat (daha hızlı ve sağlıklı işlemler için)

# veriler ve verilerin sonuçlarının tutlulacağı dizileri oluşturuyoruz

datas = []
label = []

# verilere yani fotoğraflara erişim
# os kütüphanesi sayesinde sistem dosyalarına erişim sağlandı ve içinde gezinildi
# classes de il maskeli olduğu için ilk maskeli fotoğrafları dönecek daha sonra label e de maskeli eklemesi yapacak 
#daha sonra maskeli bittiği an maskesize sıra gelecek eriştiği maskesiz fotoğrafları dolaştıktan sonra onları da datas a ekleyip label a maskesiz bilgisini gönderecek
for cl in classes:
    in_path = os.path.join(path, cl)
    for img in os.listdir(in_path):
    	img_path = os.path.join(in_path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	datas.append(image)
    	label.append(cl)


# one-hot encoding yapıldı ki kategoriler arasında ağırlık varmış gibi olmasın
lb = LabelBinarizer()
label = lb.fit_transform(label)
label = to_categorical(label)

datas = np.array(datas , dtype="float32")
label = np.array(label)

# %20 test oranı ayarlandı. stratify = label yapıldı. yani sınıflandırma yaparken dağılımı ayarladı ki öğrenmede daha stabil olsun
x_train, x_test, y_train, y_test = train_test_split(datas,label, test_size=0.2,stratify=label,random_state=42)


# fotoğraları daha da çeşitli hale getirerek makineye daha fazla öğrenim seçeneği sunuluyor. yakınlaştırma, aynalama, döndürme gibi işlemler uygulanarak çeşitlilik sağlanıyor
ıdg = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# MobileNetV2 kullanarak modeli daha hafifletilmiş şekilde kullanmak için yazılan kod
# fully connected seçim sayesinde modelin kendi üst katmanlarından ziyade bizim katmanlarımızı kullanmamıza imkan sağlar
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))


# bu kısım önceden hazırlanmış baseModel üzerine veri kümemize uygun sınıflandırma yapmak için katman eklenen kısımdır
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


# artık gerçekte eğiteceğimiz modelin içine önceki modelleri yerleştiriyoruz

model = Model(inputs= baseModel.input, outputs= headModel)

# bu kısımda baseModel kısmındaki tüm katmanların eğitilebilir durumunu false ederek önceden eğitilmiş modelin daha fazla eğitilmesini kapatarak bizim modelimize uygun hale getiriyor

for layer in baseModel.layers:
	layer.trainable = False
    
#modelimizi compile ediyoruz. yani yukarıda belirlediğimiz her şeyi modelimize yerleştiriyoruz
# optimizasyon alogirtmasi olarak adam seçiliyor ve gerekli parametreler ayarlanıyor

optimize = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=optimize,metrics=["accuracy"])


# modelimizin eğitildiği yerdeyiz

H = model.fit(
	ıdg.flow(x_train, y_train, batch_size=BS),
	steps_per_epoch=len(x_train) // BS,
	validation_data=(x_test, y_test),
	validation_steps=len(x_test) // BS,
	epochs=EPOCHS)

model.save("model.h5")

#burada modelimizin kayip verilerine yani loss kısmına bakıyoruz. bizim için önemli olan "loss-val_loss" değerlei istediğimiz gibi mi kontrol ediyoruz
kayipVeri = pd.DataFrame(model.history.history)
kayipVeri.head()
kayipVeri.plot()
