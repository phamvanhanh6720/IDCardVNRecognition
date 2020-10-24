# IDCardVNRecognition
* Việc triển khai model, tôi sử dụng Tensorflow Serving và Flask. Vì vậy để chạy demo trên máy cá nhân, trước hết quý bạn đọc cần cài đặt Tensorflow Serving.
* Recommend: cuda10.1 + ubuntu (OS)

## Hướng dẫn cài đặt Tensorflow Serving
```
sudo pip install tensorflow-serving-api
```
```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
```
```
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
```
```
sudo apt-get update && sudo apt-get install tensorflow-model-server
```



## Demo
```
cd IDCardRecognition
```
Trong file serving.config trong folder models sửa lại đường dẫn tới model:
```
base_path: "absolute link to model"
eg: base_path: "/home/phamvanhanh/PycharmProjects/IDCardVNRecognition/models/cropper"
```
```
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=./models/serving.config
```
``` 
python run.py
```

