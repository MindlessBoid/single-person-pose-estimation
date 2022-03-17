if [ -d dataset/images ]; then
    echo directory already exists
else
    mkdir -p dataset/images
fi

echo Getting train2017
wget "http://images.cocodataset.org/zips/train2017.zip"
unzip train2017 -d dataset/images
rm train2017.zip

echo Getting val2017
wget "http://images.cocodataset.org/zips/val2017.zip"
unzip val2017 -d dataset/images
rm val2017.zip

#echo Getting test2017
#wget "http://images.cocodataset.org/zips/test2017.zip"
#unzip test2017 -d dataset/images
#rm test2017.zip

echo Getting annotations
wget "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
unzip annotations_trainval2017 -d dataset
rm annotations_trainval2017.zip

echo Done