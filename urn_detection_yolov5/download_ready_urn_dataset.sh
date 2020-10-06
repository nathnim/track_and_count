mkdir urn_dataset urn_dataset_coco

cd urn_dataset
curl -L "https://app.roboflow.ai/ds/SESRKBTPJ8?key=P0A7b9KUtr" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
cd ../urn_dataset_coco
curl -L "https://app.roboflow.ai/ds/SWoKG6vwDz?key=kj0J1ymOpt" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
cd ..
