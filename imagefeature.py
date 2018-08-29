from keras.engine import Model
from keras.layers import GlobalAveragePooling2D
# from keras.utils import plot_model
from numpy import linalg as LA

from keras.preprocessing import image
import numpy as np


class ImageFeature:
    def __init__(self, modelname="resnet50", input_shape=(224, 224, 3), include_top=False):

        self.input_shape = input_shape

        self.modelname = modelname

        from keras.applications import resnet50
        self.model_package = resnet50
        self.model = self.model_package.ResNet50(weights='imagenet',
                                                 input_shape=self.input_shape,
                                                 pooling='max', include_top=include_top)

    def __image_init(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img

    def get_feature(self, img_path):
        img = self.__image_init(img_path=img_path)
        if self.modelname != "myvgg16":
            img = self.model_package.preprocess_input(img)
        feature = self.model.predict(img)
        # norm_feature = feature[0] / LA.norm(feature[0])
        norm_feature = feature[0]
        return norm_feature

    def retrieval(self, img_path, features):
        query_feature = self.get_feature(img_path)
        scores = np.dot(query_feature, features.T)
        rank_ID = np.argsort(scores)[::-1]
        rank_score = scores[rank_ID]

        return rank_ID, rank_score

if __name__=="__main__":
    IF = ImageFeature(include_top=True)

