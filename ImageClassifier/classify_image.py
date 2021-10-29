import os
import streamlit as st
from kafka import KafkaConsumer
from ImageClassifier import ImageClassifier

# Setup Consumer
consumer = KafkaConsumer('img_file_path', bootstrap_servers=['localhost:9092'], 
                            auto_offset_reset='latest', enable_auto_commit=True)

# Setup classifer
img_clf = ImageClassifier()
img_clf.load_trained_model()


# Process the image
for file_path in consumer:
    # image = Image.open(file_path)
    # print(file_path.value, type(file_path.value))
    fpath = file_path.value.decode('utf-8')
    dir_path, file_name = os.path.split(fpath)
    label = img_clf.predict_from_image_files(dir_path)[0]
    st.image(fpath, caption=file_name)
    st.subheader('Classifier Output : %s ' %label)
    print(fpath)
