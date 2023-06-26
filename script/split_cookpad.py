import os

img_dir='/srv/datasets/FoodLog/Cookpad+/meal'
subdir = [f.path for f in os.scandir(img_dir) if f.is_dir()]
train_f = open('splits/cookpad/train.txt','w')
valid_f = open('splits/cookpad/valid.txt','w')
test_f = open('splits/cookpad/test.txt','w')
count = 0
for dir in subdir:
    for path in os.listdir(dir):
        split_path = os.path.join(dir.split('/')[-1],path)
        if count % 10 < 8:
            print(split_path,file=train_f)
        elif count % 10 < 9:
            print(split_path,file=valid_f)
        else:
            print(split_path,file=test_f)
        count += 1
train_f.close()
valid_f.close()
test_f.close()