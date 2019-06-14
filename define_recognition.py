from define_common import *
from define_loader import Image_loader
import cv2
import numpy as np
from keras import backend as K

from PIL import Image


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # grayscale
        pass
    elif new_image.shape[2] == 3:  # color
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # transmit
        new_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


#_ define Face_recog
class Recognition:
    def __init__(self, arg_model, arg_labels, arg_layer_name):
        self.model = arg_model
        self.labels = arg_labels
        self.layer_name = arg_layer_name
        self.predicts = []
        self.face_num = 0
        self.faces = []
        self.cv2_faces = []
        self.jetcams = []
        self.ins_image_loader = Image_loader('', image_size)
        K.set_learning_phase(1)  # set learning phase

    def get_face(self, arg_face_rect, arg_image):
        self.faces = []
        self.cv2_faces = []
        for rect in arg_face_rect:
            self.face_num = 0

            # Get face info
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]

            # Cut out only the face and save it
            get_cv2_face = arg_image[y:y+height, x:x+width]
            get_face = cv2.resize(arg_image[y:y+height, x:x+width], (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            get_face = cv2pil(get_face)
            '''
            new_image_path = capture_image_path + '/' + str(self.frame_num) + '_' + str(face_num) + '.jpg'
            cv2.imwrite(new_image_path, get_face)
            '''
            self.faces.append(get_face)
            self.cv2_faces.append(get_cv2_face)
            self.face_num += 1
        self.ins_image_loader.set_PILs(self.faces)

    def predict(self, arg_select_label, arg_frame_count):
        self.ins_image_loader.get_grayscales()
        self.predicts = self.model.predict_classes(self.ins_image_loader.grayscales)
        for for1, (predict, face) in enumerate(zip(self.predicts, self.faces)):
            if arg_select_label == predict:
                new_image_path = save_base_path + '/' + self.labels[predict] + '/' + date_str
                if os.path.isdir(new_image_path) is False:
                    os.makedirs(new_image_path)
                new_image_path += '/' + str(arg_frame_count) + '_' + str(for1) + '.jpg'
                face.save(new_image_path)

    def grad_cam(self, arg_select_label,  arg_frame_count):
        self.ins_image_loader.get_grayscales()
        self.ins_image_loader.get_colors()
        self.jetcams = []

        for x, cam_x in zip(self.ins_image_loader.grayscales, self.ins_image_loader.colors):
            X = np.expand_dims(x, axis=0)

            X = X.astype('float32')
            preprocessed_input = X  # / 255.0

            predictions = self.model.predict(preprocessed_input)
            class_idx = np.argmax(predictions[0])
            class_output = self.model.output[:, class_idx]

            conv_output = self.model.get_layer(self.layer_name).output
            grads = K.gradients(class_output, conv_output)[0]
            gradient_function = K.function([self.model.input], [conv_output, grads])

            output, grads_val = gradient_function([preprocessed_input])
            output, grads_val = output[0], grads_val[0]

            weights = np.mean(grads_val, axis=(0, 1))
            cam = np.dot(output, weights)

            cam = cv2.resize(cam, (image_size, image_size), cv2.INTER_LINEAR)
            cam = np.maximum(cam, 0)
            cam = cam / cam.max()

            jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            # jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)
            jetcam = ((np.float32(jetcam) + cam_x) / 2)
            self.jetcams.append(jetcam)

        self.predicts = self.model.predict_classes(self.ins_image_loader.grayscales)
        #for for1, (predict, jetcam) in enumerate(zip(self.predicts, self.jetcams)):
        #    if arg_select_label == predict:
        #        new_image_path = save_base_path + '/' + self.labels[predict] + '/' + date_str
        #        if os.path.isdir(new_image_path) is False:
        #            os.makedirs(new_image_path)
        #        new_image_path += '/' + str(arg_frame_count) + '_' + str(for1) + '.jpg'
        #        jetcam.save(new_image_path)
