# ___Система распознавания лиц___
Финальный проект Дмитрия Гринева
***

# Использование
## Среда для запуска проекта
Установка поддержки GPU в docker https://www.tensorflow.org/install/docker#gpu_support
    
    cd docker
    docker build -t tf2gpu .
    
  
  - **запуск jupyter server из папки проекта**
    ```
    docker run -it --rm -v $(pwd):/app -p8888:8888 --gpus all tf2gpu
    ```
    
## Подготовка проекта
  - **Скачать предобученные модели** с https://drive.google.com/file/d/1EkF4VoA0XSLZKIDO2zW0QD6MU6sdTs1l/view?usp=sharing и положить в папки models и app_back/models.
  - **Разархивировать** app_front.zip в папку проекта.
  - **Клонировать репозиторий**
  ```sh
    git clone https://github.com/leondgarse/Keras_insightface.git recognition
    cp ArcFace.ipynb /recognition
```

  Открыть jupyter-notebook по адресу 127.0.0.1:8888 с паролем __root__. 

  __Ноутбуки__
  - **FinalProject** - основной код проекта
  - **RetinaFace2** - код обучения детектора лиц
  - **Face Detector Based On EfficientDet B0** - код обучения детектора лиц на основе EffDet B0
  - **recognition/ArcFace** - код обучения распознавания лиц

__Датасеты__
- Для обучения **ArcFace** - https://github.com/deepinsight/insightface/wiki/Dataset-Zoo
- Для обучения **RetinaFace** - WiderFace 

https://drive.google.com/file/d/14Wy-BUV7BFAe91yddLKpD-DsFpeg2IW0/view?usp=sharing<br> https://drive.google.com/file/d/1sc3lUr_7LppHPEC5FhXiTIxn1HuhZNoo/view?usp=sharing 
https://drive.google.com/file/d/1YW2_XQ1l30usbKkWCSrmf4oIC9AHRyga/view?usp=sharing
  
## Приложение
  - **Запуск готового приложения API + Front**
    ```sh
    docker-compose up -d
    ```
    **API SWAGGER** - http://127.0.0.1:8000/docs 

    **VUE FRONT** - http://127.0.0.1:8080/
    