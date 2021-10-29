import time, os
import urllib.request
import streamlit as st
from kafka import KafkaProducer


# Setup Producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'])


def download_image():
  try:
    # Clear the download folder
    img_dir = './download'
    for file in os.listdir(img_dir):
      os.remove(os.path.join(img_dir, file))

    # Adding information about user agent
    opener=urllib.request.build_opener()
    opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36')]
    urllib.request.install_opener(opener)
    
    # Download the image in the download folder
    image_url = title
    file_name = image_url.split('/')[-1]
    (file_path, response) = urllib.request.urlretrieve(image_url, os.path.join(img_dir, file_name))

    # Send the file_path to Kafka
    producer.send('img_file_path', file_path.encode('utf-8'))

    st.success("Image downloaded : %s " %file_path)
  except Exception as ex:
    st.error("Download failed : %s " %str(ex))


st.title('Image Classifier')
title = st.text_input('Enter URL of Image', '')
button = st.button(label="Submit", on_click=download_image)

