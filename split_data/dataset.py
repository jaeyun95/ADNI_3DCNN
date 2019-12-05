import csv
import os

## 데이터 포맷 맞추기
## 파일이름 클래스 형태임 ex) MRI데이터.nii AD 
train_3classes = open("/media/ailab/Backup Plus/ADNI/data/train_3classes.txt","w")
test_3classes = open("/media/ailab/Backup Plus/ADNI/data/test_3classes.txt","w")
image_file_dirs = "/media/ailab/Backup Plus/ADNI/data/AD"
for root,dirs,files in os.walk(image_file_dirs):
    for i,file_name in enumerate(files):
        if i<200:
            dataTrain = file_name + ' AD\n'
            train_3classes.write(dataTrain)
        else:
            dataTest = file_name + ' AD\n'
            test_3classes.write(dataTest)
			
			
image_file_dirs = "/media/ailab/Backup Plus/ADNI/data/MCI"
for root,dirs,files in os.walk(image_file_dirs):
    for i,file_name in enumerate(files):
        if i<450:
            dataTrain = file_name + ' MCI\n'
            train_3classes.write(dataTrain)
        else:
            dataTest = file_name + ' MCI\n'
            test_3classes.write(dataTest)
	
	
image_file_dirs = "/media/ailab/Backup Plus/ADNI/data/CN"
for root,dirs,files in os.walk(image_file_dirs):
    for i,file_name in enumerate(files):
        if i<300:
            dataTrain = file_name + ' CN\n'
            train_3classes.write(dataTrain)
        else:
            dataTest = file_name + ' CN\n'
            test_3classes.write(dataTest)

train_3classes.close()
test_3classes.close()


