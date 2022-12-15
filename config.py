
import os
import re

MAIN_DIR = "/data/"


DATA_DIR = f'{MAIN_DIR}'
VOCAB_DIR = f'{MAIN_DIR}/iq/vocab'
CKPT_DIR = f'{MAIN_DIR}/iq/ckpt'
RANKING_DIR = f'{MAIN_DIR}/iq/rankings'
HEATMAP_DIR = f'{MAIN_DIR}/iq/heatmaps'


TORCH_HOME = "/data/pretrain_model/"
GLOVE_DIR = "/data/pretrain_model/"






FASHIONIQ_IMAGE_DIR = f'{DATA_DIR}/IQ/images'
FASHIONIQ_ANNOTATION_DIR = f'{DATA_DIR}/IQ'


SHOES_IMAGE_DIR = f'{DATA_DIR}/shoes/images'
SHOES_ANNOTATION_DIR = f'{DATA_DIR}/shoes/annotations'


CIRR_IMAGE_DIR = f'{DATA_DIR}/img_feat_res152'
CIRR_ANNOTATION_DIR = f'{DATA_DIR}/cirr'


FASHION200K_IMAGE_DIR = f'{DATA_DIR}/women'
FASHION200K_ANNOTATION_DIR = f'{FASHION200K_IMAGE_DIR}/labels'






cleanCaption = lambda cap : " ".join(re.sub('[^(a-zA-Z)\ ]', '', re.sub('[/\-\\\\]', ' ', cap)).split(" "))
