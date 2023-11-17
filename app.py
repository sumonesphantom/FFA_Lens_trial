from typing import List
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.colors as mcolors
from PIL import Image
from config import CLASSES1
from config import CLASSES2
from ultralytics import YOLO
from pathlib import Path
from io import BytesIO
import zipfile

html = """
<div style = "background-color:black;padding:18px">
<h1 style = "color:green; text-align:center"> Detect Lesion</h1>
</div>
"""
st.set_page_config(
    page_title="FFA_LENS",
)

st.markdown(html, unsafe_allow_html =True)
def attempt_download_yolo(file, repo):
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", '').lower())

# @st.cache(max_entries=2)
# def get_yolo3(weights):
#     # return torch.hub.load("/yolov3","custom",path='{}'.format(weights),source='local',force_reload=True)
#     return torch.hub.load("ultralytics/yolov3","custom",path='{}'.format(weights),force_reload=True)

# #WongKinYiu
# @st.cache(max_entries=2)
# def get_yolo5(weights):
#     return torch.hub.load('ultralytics/yolov5','custom',path = '{}'.format(weights), force_reload =True)

@st.cache(max_entries=2, allow_output_mutation=True,ttl=0.1*3600)
def get_yolo8(weights):
    return YOLO(str(weights))
    # return torch.hub.load("ultralytics/ultralytics","custom",path='{}'.format(weights),force_reload=True)


@st.cache(max_entries=10,ttl=0.1*3600)
def get_preds(img, imgsz):
    # if all_classes == False:
    #     model.conf = conf_thres
    #     model.classes = classes
    #     model.iou = iou_thres
    #     #result = model([img], size=imgsz, conf =conf_thres, iou = iou_thres, max_det = max_det, classes = classes)
    #     result = model([img], size=imgsz)
    # elif all_classes == True and weights!="yolov8.pt"and weights!="yolov7.pt":
    #     model.conf = conf_thres
    #     model.classes = None
    #     model.iou = iou_thres
    #     #result = model([img], size=imgsz, conf =conf_thres, iou = iou_thres, max_det = max_det, classes = None)
    #     result = model([img], size=imgsz)
    if weights=="yolov8.pt":
        if all_classes == False:
            model.classes = classes 
        model.conf = conf_thres
        model.iou = iou_thres
        #result = model([img], size=imgsz, conf =conf_thres, iou = iou_thres, max_det = max_det, classes = None)
        results = model([img])
        boxes = results[0].boxes
        result = boxes.data  # returns one box
        return result.numpy()
    print("hi",result.xyxy[0])
    return result.xyxy[0].numpy()


def get_colors(indexes):
    to_255 = lambda c: int(c*255)
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) 
                                                for name_color in tab_colors]
    base_colors = list(mcolors.BASE_COLORS.values())
    base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
    rgb_colors = tab_colors + base_colors
    rgb_colors = rgb_colors*5

    color_dict = {}
    for i, index in enumerate(indexes):
        if i < len(rgb_colors):
            color_dict[index] = rgb_colors[i]
        else:
            color_dict[index] = (255,0,0)

    return color_dict

def get_legend_color(class_name):
    index = CLASSES.index(class_name)
    color = rgb_colors[index]
    return 'background-color: rgb({color[0]},{color[1]},{color[2]})'.format(color=color)

st.markdown("""
        <style>
        .format-style{
            font-size:20px;
            font-family:arial;
            color:red;
        }
        </style>
        """,
        unsafe_allow_html= True
    )

st.markdown(
    """
    <style>.
    common-style{
        font-size:18px;
        font-family:arial;
        color:pink;
    }
    </style>
    """,
    unsafe_allow_html= True
)
st.sidebar.markdown(
    '<p class = "format-style"> Parameter </p>',
    unsafe_allow_html= True
)

modelSelect = st.sidebar.selectbox(
    'Model', 
    ('yolov8','yolov5','yolov3'),
    format_func = lambda a: a[:len(a)] 
)


weights = st.sidebar.selectbox(
    'Weights', 
    ('yolov8.pt','yolov5','yolov3'),
    format_func = lambda a: a[:len(a)-3] 
)


if weights == 'yolov8.pt':
    CLASSES = CLASSES1


imgsz = st.sidebar.selectbox(
    'Size Image',
    (416,512,608,896,1024,1280,1408,1536)
)

conf_thres = st.sidebar.slider(
    'Confidence Threshold', 0.00, 1.00, 0.7
)

iou_thres = st.sidebar.slider(
    'IOU Threshold', 0.00,1.00, 0.45
)
# max_det = st.sidebar.selectbox(
#     'Max detection',
#     [i for i in range(1,20)]
# )

classes = st.sidebar.multiselect(
    'Classes',
    [i for i in range(len(CLASSES1))],
    format_func= lambda index: CLASSES1[index]
)

all_classes = st.sidebar.checkbox('All classes', value =True)

with st.spinner('Loading the model...'):
    if(modelSelect==''):
        model = get_yolo8(weights)
    elif(modelSelect=='yolov8'):
        model = get_yolo8(weights)
    

st.success('Loading '+modelSelect+' model.. Done!')

prediction_mode = 'Multiple images'
# st.sidebar.radio(
#     "",
#     ('Single image','Multiple images','none'),
#     index=0)

if all_classes:
    target_class_ids = list(range(len(CLASSES1)))
elif classes:
    target_class_ids = [class_name for class_name in classes]
else:
    target_class_ids = [0]

rgb_colors = get_colors(target_class_ids)

detected_ids = None

def load_images(file_uploader, imgsz):
    uploaded_files = file_uploader("Upload image", type=['png', 'jpg', 'jpeg','tif'], accept_multiple_files=True)
    file_names = []
    images = []
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        file_names.append(uploaded_file.name)

    return [images,file_names]

def process_images(images, target_class_ids, all_classes, weights):
    results = []
    for img in images:
        result = get_preds(img, imgsz)

        result_copy = result.copy()
        result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]

        results.append(result_copy)

    return results
def draw_bounding_boxes(images, results, rgb_colors, CLASSES2, conf_thres, iou_thres):
    drawn_images = []
    for img, result_copy in zip(images, results):
        img_draw = img.copy().astype(np.uint8)
        font = cv2.FONT_HERSHEY_TRIPLEX

        res = []
        detected_ids = []

        text = "Some text in a box!"
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=0.5, thickness=2)[0]
    
        for bbox_data in result_copy:
            xmin, ymin, xmax, ymax, con, label = bbox_data

            if con >= conf_thres:
                con = round(con, 4)
                p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)

                img_draw = cv2.rectangle(img_draw, p0, p1, rgb_colors[label], 2)
                label2 = CLASSES1.index(str(label))
                print(str(label), CLASSES2[label2])
                res.append(str(label) + ' ' + CLASSES2[label2])

                box_coords = ((int(xmin) - 1, int(ymin)),
                              (int(xmin) + text_width - 80, (int(ymin) - 5) - text_height))
                img_draw = cv2.rectangle(img_draw, box_coords[0], box_coords[1], rgb_colors[label], cv2.FILLED)

                img_draw = cv2.putText(img_draw, str(label), (int(xmin), int(ymin) - 5), font, 0.5, (255, 255, 255), 1)
                img_draw = cv2.putText(img_draw, ', ' + str(con), (int(xmin) + 30, int(ymin) - 5), font, 0.5, (255, 255, 255), 1)
                detected_ids.append(label)

        drawn_images.append((res, img_draw))

    return drawn_images

def display_results(drawn_images):
    for i, (res, img_draw) in enumerate(drawn_images):
        col1, col2 = st.columns(2)
        with col1:
            st.write(res)
            download_single_image(drawn_images[i], file_names[i])
        with col2:
            st.image(img_draw, use_column_width=True)

    download_zip(drawn_images,file_names)

def download_zip(drawn_images, file_names):
    with BytesIO() as zip_buffer:
        with zipfile.ZipFile(zip_buffer, 'a') as zip_file:
            for i, (res, img_draw) in enumerate(drawn_images):
                image_bytes = image_to_bytes(img_draw)
                file_name = f"{file_names[i].rsplit('.', 1)[0]}.png"  # Use the original file name with a ".png" extension
                zip_file.writestr(file_name, image_bytes)

        zip_buffer.seek(0)
        st.download_button(label="Download All Images", data=zip_buffer, file_name='images.zip', key='download_all')

def download_single_image(drawn_image, file_name):
    _, img_draw = drawn_image
    image_bytes = image_to_bytes(img_draw)
    file_name = f"{file_name.rsplit('.', 1)[0]}.png"  # Use the original file name with a ".png" extension
    st.download_button(label="Download This Image", data=image_bytes, file_name=file_name, key='download_single')

def image_to_bytes(image):
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    return buffer.getvalue()

if prediction_mode == 'Multiple images':
    images,file_names = load_images(st.file_uploader, imgsz)

    if images:
        results = process_images(images, target_class_ids, all_classes, weights)
        drawn_images = draw_bounding_boxes(images, results, rgb_colors, CLASSES2, conf_thres, iou_thres)
        display_results(drawn_images)

