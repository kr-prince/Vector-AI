# Vector-AI

### Setup Instructions

1. Clone the git repo locally. `git clone https://github.com/kr-prince/Vector-AI.git`
2. `cd Vector-AI`
3. Run `conda env create -f environment.yml` and then `conda activate vectorAI`

### To Run Image-Classifier

1. Download latest Kafka binary from https://kafka.apache.org/downloads and extract it. _kafka_2.12-3.0.0.tgz_ has been used here.
2. Ensure JRE and JDK are installed and Java environment variables are set. To check, open cmd or terminal and try 
    ```
    java --version
    javac --version
    ```
3. `cd kafka_2.12-3.0.0`
4. Run `bin/zookeeper-server-start.sh config/zookeeper.properties` to start zookeeper server
5. Run `bin/kafka-server-start.sh config/server.properties` in a separate terminal, to start Kafka broker
6. In the python environment, after activating it, run the below two in separate terminals
    ```
    streamlit run download_image.py 
    streamlit run classify_image.py
    ```


### To Train Image-Classifier from scratch

1. `from ImageClassifier import ImageClassifier`
2. `img_clf = ImageClassifier()`
3. If data is in compressed(.gzip format) like fashion MNIST, then follow the below steps:
   ```# file path for training and test data
      train_images_path = '../data/fashion/train-images-idx3-ubyte.gz'
      train_labels_path = '../data/fashion/train-labels-idx1-ubyte.gz'
      test_images_path = '../data/fashion/t10k-images-idx3-ubyte.gz'
      test_labels_path = '../data/fashion/t10k-labels-idx1-ubyte.gz'
      
      img_clf.load_from_raw_data(train_images_path, train_labels_path, test_images_path, test_labels_path)
    ```
    Else if you want to train on raw images, where each of the sub-directories inside a directory carry images of that class and the name of the sub-directory is the class name itself, then : 
    
    ```img_clf.load_from_image_files(data_dir)```
4.  ```img_clf.train()```
5.  Output is : 
    ```
      Training initiated..
      Training Completed..               
      Training Loss : 0.123
      Validation Loss : 0.253
      Training Accuracy : 0.955
      Validation Accuracy : 0.914
      Training history plots saved in results.
      INFO:tensorflow:Assets written to: ./model/assets
      Test Loss : 0.256
      Test Accuracy : 0.916
      Confusion Matrix saved in results.
      Classification report saved in results.
    ```

### App Preview

![Image Download](https://github.com/kr-prince/Vector-AI/blob/main/experiments/download.png)
![Image Classify](https://github.com/kr-prince/Vector-AI/blob/main/experiments/classify.png)
