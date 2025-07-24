import os
import shutil
import pandas as pd

if __name__ == '__main__':
    img_path = "/media/D/yazid/Dataset/mini-imagenet/images/"
    train_csv = pd.read_csv("/media/D/yazid/Dataset/mini-imagenet/train.csv")
    test_csv = pd.read_csv("/media/D/yazid/Dataset/mini-imagenet/test.csv")
    val_csv = pd.read_csv("/media/D/yazid/Dataset/mini-imagenet/val.csv")
    for index,row in train_csv.iterrows():
        filename, label = row['filename'], row['label']
        src_path = os.path.join(img_path,filename)
        dst_path = os.path.join(img_path,label,filename)
        if not os.path.exists(os.path.join(img_path,label)):
            os.mkdir(os.path.join(img_path,label))
        shutil.move(src_path,dst_path)
    for index,row in test_csv.iterrows():
        filename, label = row['filename'], row['label']
        src_path = os.path.join(img_path,filename)
        dst_path = os.path.join(img_path,label,filename)
        if not os.path.exists(os.path.join(img_path,label)):
            os.mkdir(os.path.join(img_path,label))
        shutil.move(src_path,dst_path)
    for index, row in val_csv.iterrows():
        filename, label = row['filename'], row['label']
        src_path = os.path.join(img_path, filename)
        dst_path = os.path.join(img_path, label, filename)
        if not os.path.exists(os.path.join(img_path, label)):
            os.mkdir(os.path.join(img_path, label))
        shutil.move(src_path, dst_path)
