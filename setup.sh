pip3 install -r requirements.txt --quiet

mkdir data; cd data
kaggle datasets download -d "xhlulu/140k-real-and-fake-faces"
unzip "140k-real-and-fake-faces"
cd ../