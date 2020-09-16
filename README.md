# IDCardVNRecognition

*Recommend: Ubuntu, cuda10.1 
*Việc triển khai model, tôi sử dụng tensorflow-serving và flask.Vì vậy để có thể chạy demo trên máy cá nhân,các bạn cần cài đặt tensorflow-serving.

# Hướng dẫn cài đặt tensorflow-serving

# step 1
$ echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

# step 2
$ sudo apt-get update && sudo apt-get install tensorflow-model-server

# step 3
$ pip install tensorflow-serving-api

# Demo

# step 1
$ tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=./models/serving.config

# step 2
Trong file serving.config
./models/serving.config: chỉnh lại base_path của bạn ( "/home/{user_name}/.../IDCardRecognition/models/{model_name}"

# step 3
$ python run.py


