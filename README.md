# IDCardVNRecognition
* Recommend: GPU + OS(Ubuntu)


## Update 
* I recently updated the new version that is easy to install and is improved slightly in accuracy and performance.
  
## Install Dependencies
```angular2html
# Install an ASGI server, for production such as Uvicorn or Hypercorn.
pip3 install uvicorn[standard]
```
```angular2html
pip3 install -r requirements.txt
```

## Run Server
```angular2html
uvicorn run:app --host='hostname'
```