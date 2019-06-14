#_ import
import os
from datetime import datetime

script_path = os.path.dirname(os.path.abspath(__file__))

'''
_ model path
'''
model_base_path = script_path + '/kaggle_nisime_gap_i150_result'
label_path = model_base_path + '/label.csv'
model_path = model_base_path + '/train_model.hdf5'
cascade_path = script_path + '/haarcascades/haarcascade_frontalface_alt.xml'# Path of xml file for face recognition prepared in OpenCV

'''
_ model info
'''
layer_name = 'activation_7'

'''
_ dir path
'''
save_base_path = script_path + '/recog_face'
capture_image_path = save_base_path + '/capture_image'
date_str = datetime.now().strftime('%Y%m%d_%H:%M:%S')

'''
_ cv info
'''
image_size = 48
TARGET_SIZE = 200

'''
_ display info
'''
select_exp = 0
select_op = {'face_recognition': 0, 'expression_recognition': 0, 'percent': 0, 'mosaic': 0, 'swap': 0, 'grad_cam': 0}
expression_colors = [(255, 215, 0),
                     (255, 0, 0),
                     (24, 203, 220),
                     (50, 205, 50),
                     (138, 43, 226),
                     (128, 128, 128),
                     (255, 255, 255)]# Color of frame displayed on face
op_color = (0, 0, 0)
expression_count = [0, 0, 0, 0, 0, 0, 0]
font_size = 1

# save_exp = np.array([['happy_recog','happy'],['sadness_recog','sadness'],['surprise_recog','surprise'],['disgust_recog','disgust'],['angry_recog','angry'],['fear_recog','fear'],['neutral_recog','neutral'],[datetime.now().strftime('%Y%m%d_%H:%M:%S'),datetime.now().strftime('%Y%m%d_%H:%M:%S')]])
# frame_num = 0
# frame_count = 0 # Number of frame images read
# img_count = 0 # Number of candidate images saved
# save_facerect = []
# # Cascade
# save_cam = []
