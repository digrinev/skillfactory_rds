from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

import os
import numpy as np
from tensorflow.python.framework.tensor_conversion_registry import get
from modules.face import FaceDetector, face_align_by_landmarks, prepare_image, embedding_images
from skimage.io import imread, imsave
from skimage.color import gray2rgb, rgba2rgb
import pickle
import faiss
from collections import defaultdict
from tensorflow.keras.models import load_model
import tensorflow as tf
import uuid
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler, StandardScaler

tf.keras.mixed_precision.set_global_policy("mixed_float16")

face_detector = FaceDetector(model_path='models/Resnet50_Final.pth')
model_path = os.path.join(os.getcwd(), 'models', 'glint360k_cosface_r100_fp16_0.1.h5')
face_model = load_model(model_path, compile=False)

from sklearn.preprocessing import normalize
from abc import abstractmethod, ABC


PHOTO_UPLOAD_PATH = os.path.join(os.getcwd(), 'app_front', 'dist', 'static', 'photos')


class FaceDetectorBase(ABC):
    @abstractmethod
    def detect(self):
        raise NotImplementedError


class FaceRecognizerBase(ABC):
    @abstractmethod
    def recognize(self):
        return NotImplementedError


class Recognizer(FaceRecognizerBase):
    def __init__(self, model) -> None:
        self.fm = model

    def recognize(self, imgs, batch_size=32):
        """
            Функция получает эмбеддинги из кропов лиц
        """
        steps = int(np.ceil(len(imgs) / batch_size))
        embeddings = [self.fm(imgs[ii * batch_size : (ii + 1) * batch_size]) for ii in range(steps)]
        embeddings = normalize(np.concatenate(embeddings, axis=0))

        return embeddings


class Detector(FaceDetectorBase):

    def __init__(self, model) -> None:
        self.detm = model

    def detect(self, imgs, labels=None):
        """
            imgs - массив ссылок на картинки
            
            Return:
                self.imgs - содержит выравненные и нормированные ((img - 127.5) * 0.0078125) кропы лиц, размером 112х112 пикселей.
        """

        images = []

        ids_to_remove = []
        for idx, img in enumerate(imgs):
            img_bin = imread(img)
            img_bin = prepare_image(img_bin, self.detm)
            # Если обнаружено несколько лиц, берем 1
            if img_bin.shape[0] > 1:
                img_bin = np.expand_dims(img_bin[0], axis=0)
            # Если лиц не найдено - пропускаем файл
            if img_bin.shape[0] == 0:
                # Удалим метку данного изображения и ссылку на фотку
                if labels is not None:
                    ids_to_remove.append(idx)

                # ...Тут можно залогировать куда-нибудь...
                continue

            images.append(img_bin)

        return np.concatenate(images, axis=0), ids_to_remove 
     
    

# Создадим наш пайплайн
class FRSPipelineBuilder:
    def __init__(self, imgs: list, detector_model: FaceDetectorBase, face_model: FaceRecognizerBase, labels: list = None) -> None:
        
        # принимаем ссылки на изображения
        self.__imgs = imgs

        self.__crops = None
        self.__labels = labels

        # Загружаем модели
        self.fm = face_model
        self.detm = detector_model

        self.__embeddings = []
    
    def detect(self):
        self.__crops, ids_to_remove = self.detm.detect(self.__imgs, self.__labels)
        # Удалим изображения и метки где не было детектов
        for id in ids_to_remove:
            del self.__imgs[id]
            del self.__labels[id]
        
        return self

    def recognize(self, batch_size=32):
       self.__embeddings.append(self.fm.recognize(self.__crops, batch_size=batch_size))
       return self

    def match(self):
        raise NotImplementedError

    def enroll(self):
        raise NotImplementedError

    def get_embeddings(self):
        return np.concatenate(self.__embeddings, axis=0)
    
    def get_labels(self):
        return self.__labels

    def get_faces(self):
        return self.__crops

    def get_photo_path(self):
        return self.__imgs
    

# Создадим наш storage
class StorageBase(ABC):
    @abstractmethod
    def save(self, filename, embeddings, labels):
        raise NotImplementedError

    @abstractmethod
    def load(self, filename):
        raise NotImplementedError
    
    @abstractmethod
    def add(self, profiles, labels):
        raise NotImplementedError
    
    @abstractmethod
    def remove(self, id):
        raise NotImplementedError


class PickleStorage(StorageBase):

    def __init__(self) -> None:
        self.__storage = []
        self.__labels_classes = None


    def load(self, filename):
        if os.path.isfile(filename):
            with open(filename, 'rb') as sf:
                self.__storage = pickle.load(sf)
        if os.path.isfile(filename.split('.')[0] + '_encoder.pkl'):
            with open(filename.split('.')[0] + '_encoder.pkl', 'rb') as sf:
                self.__labels_classes = pickle.load(sf)
            
        return self


    def save(self, filename):
        with open(filename, 'wb') as sf:
            pickle.dump(self.__storage, sf)
            
        with open(filename.split('.')[0] + '_encoder.pkl', 'wb') as sf:
            pickle.dump(self.__labels_classes, sf)
            
        return self
    
    
    def add(self, profiles, labels):
        
        profiles = [os.path.basename(p) for p in profiles]

        # Дополним метки классов
        if self.__labels_classes:
            current_classes = list(set(labels))
            
            for current_class in current_classes:
                if current_class not in self.__labels_classes:
                    self.__labels_classes.append(current_class)
        else:
            self.__labels_classes = list(set(labels))
            
        # Закодируем метки
        labels = [self.__labels_classes.index(l) for l in labels]
               
        for idx, label in enumerate(labels):
            #profiles_photos[label].append(profiles[idx])
            
            do_append_at_id = None
            for storage_id, s in enumerate(self.__storage):
                for key, val in s.items():
                    if key == 'class' and val == label:
                        do_append_at_id = storage_id
                    
            if do_append_at_id is not None:
                self.__storage[do_append_at_id]['photos'].append(profiles[idx])
            else:
                self.__storage.append({'class': label, 'label': self.__labels_classes[label], 'photos': [profiles[idx]] })
      
        return labels
    
    
    def remove(self, label):
        
        try:
            id = self.__labels_classes.index(label)
                        
            for sid, storage in enumerate(self.__storage):
                if storage['class'] == id:
                    self.__storage.pop(sid)
                    
            return {'msg': f'{label} was deleted from db.'}
        except ValueError:
            return {'msg': f'{label} was not found in db.'}
            
                    
    def get_storage(self):               
        return self.__storage
    
    def get_classes(self):
        return self.__labels_classes

    @property
    def classes__(self):
        return self.__labels_classes
    
    
# Создадим IndexStorage
class FaissStorage(StorageBase):
    
    def __init__(self, index = None) -> None:
        self.__index = index
            
    def load(self, index_file):
        if os.path.isfile(index_file):
            self.__index = faiss.read_index(index_file)
        return self
    
    def save(self, filename='db.index'):
        faiss.write_index(self.__index, filename)    
        
    def add(self, embeddings, label_ids):
        if not self.__index:
            self.__index = faiss.index_factory(embeddings.shape[1], "IDMap,Flat")
        
        if isinstance(label_ids, list):
            label_ids = np.array(label_ids)
            
        self.__index.add_with_ids(embeddings, label_ids.astype(np.int64))
        
        return self
    
    def remove(self, id):
        raise NotImplementedError
    
    def get_index(self):
        return self.__index
    

def get_frs_pipeline(contents, labels=None, keep_photo=False):
    """[get_frs_pipeline]

    Args:
        contents ([bytes-array]): [Загруженная фотография]
        labels ([list], optional): [Список меток для энролла в базу]. Defaults to None.
        keep_photo (bool, optional): [Сохранять фотку на диске или нет]. Defaults to False.

    Returns:
        [FRSPipelineBuilder]: [pipeline]
    """
    file_path = os.path.join(PHOTO_UPLOAD_PATH, str(uuid.uuid4())) + '.jpeg'
    
    photo_bin = Image.open(BytesIO(contents))
    photo_bin.save(file_path)
    
    fdet = Detector(face_detector)
    fm = Recognizer(face_model)
    
    frs = FRSPipelineBuilder([file_path], fdet, fm, labels=labels)
    pipeline = frs.detect().recognize()
    
    return pipeline
    

storage = PickleStorage()
index_storage = FaissStorage()
storage.load('db.pkl')
index_storage.load('db.index')

app = FastAPI(title="DGFR Recognition API", description="Dmitriy Grinev Face Recognition API")

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def enroll_face_task(photo, label):
    pipeline = get_frs_pipeline(photo, [label])
    enc_labels = storage.add(pipeline.get_photo_path(), pipeline.get_labels())
    storage.save('db.pkl')
    index_storage.add(pipeline.get_embeddings(), enc_labels)
    index_storage.save()
        
    return {"msg": f'{label} enrolled!'}
    

@app.get("/api/v1/storage/classes", tags=["StorageService"])
async def get_storage_classes():
    return {'classes': storage.get_classes()}


@app.post("/api/v1/storage/enroll", tags=["StorageService"])
async def enroll_face(photo: UploadFile = File(...), label: str = Form(...)):
    """
        Enroll face to storage
    """
    photo_face = await photo.read()
    result = enroll_face_task(photo_face, label)
    
    return result


@app.post("/api/v1/storage/list", tags=["StorageService"])
async def get_storage():
    current_storage = storage.get_storage()
   
    return storage.get_storage()


@app.post("/api/v1/storage/remove", tags=["StorageService"])
async def remove_from_storage(label):
    msg = storage.remove(label)
    storage.save('db.pkl')
    return msg


@app.post("/api/v1/storage/match", tags=["StorageService"])
async def storage_match(photo: UploadFile = File(...), q_count = 5):
    index = index_storage.get_index()    
    pipeline = get_frs_pipeline(await photo.read())
    D, I = index.search(pipeline.get_embeddings(), int(q_count))
    
    D = 1 / (1 + np.exp(D)) + 0.5 # нормализуем l2 dist в диапазон [0,1]
    
    return {'distances': D[0].tolist(), 'indexes': I[0].tolist()}


@app.post("/api/v1/storage/decode_match", tags=["StorageService"])
async def decode_match(distances: list, class_ids: list):
    detects = []
    
    current_storage = storage.get_storage()
    
    for id, class_id in enumerate(class_ids):
        detect = list(filter(lambda person: person['class'] == int(class_id), current_storage))
        
        if len(detect) > 0:
            decoded = {'label': detect[0]['label'], 'photo': detect[0]['photos'], 'conf': int(distances[id])*100}
            detects.append(decoded)
     
    return {'decoded': detects}


@app.post("/api/v1/storage/add_task")
async def add_face_task(background_tasks: BackgroundTasks, photo: UploadFile = File(...), label: str = Form(...)):
    photo = await photo.read()
    background_tasks.add_task(enroll_face_task, photo, label)
    return {"message": "Face enroll task created!"}