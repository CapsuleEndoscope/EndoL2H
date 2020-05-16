import os 
import sys
import fnmatch
from glob import glob
import shutil

def diff(l1,l2):
    diff_list=[i for i in l1 + l2 if i in l1 and i not in l2]
    return diff_list

dirs=["A","B"]
for pth in dirs:
    files = []
    val_set=[]
    start_dir = os.getcwd()+"/x8/"+pth
    pattern   = "*.jpg"

    for dir,_,_ in sorted(os.walk(start_dir)):
        if "val" not in dir:
            files.extend(glob(os.path.join(dir,pattern))) 
        else:
            val_set.extend(glob(os.path.join(dir,pattern)))

    esophagitis=[]
    normal_pylorus=[]
    normal_z_line=[]
    polyps=[]
    ulcerative_colitis=[]

    fold1_train=[]
    fold2_train=[]
    fold3_train=[]
    fold4_train=[]
    fold5_train=[]

    trains=[]
    trains.append(fold1_train)
    trains.append(fold2_train)
    trains.append(fold3_train)
    trains.append(fold4_train)
    trains.append(fold5_train)

    fold1_test=[]
    fold2_test=[]
    fold3_test=[]
    fold4_test=[]
    fold5_test=[]

    tests=[]
    tests.append(fold1_test)
    tests.append(fold2_test)
    tests.append(fold3_test)
    tests.append(fold4_test)
    tests.append(fold5_test)

    for file in files:
        name=file[file.rfind("/")+1:]
        if "class_esophagitis" in name:
            esophagitis.append(file)

        elif "class_normal-pylorus" in name:
            normal_pylorus.append(file)

        elif "class_normal-z-line" in name:
            normal_z_line.append(file)

        elif "class_polyps" in name:
            polyps.append(file)

        elif "class_ulcerative-colitis" in name:
            ulcerative_colitis.append(file)

    for i in range(5):
        number_of_images=len(esophagitis)
        start=int(number_of_images*i/9.0)
        count=start
        for index in range(start,number_of_images):
            file=esophagitis[index]
            if count<int(number_of_images*(i+1)/9.0-34) if i!=4 else count<int(number_of_images*(i+1)/9.0-35):
                tests[i].append(file)
                count+=1
            else:
                trains[i]+=diff(esophagitis,tests[i])
                break
        
        number_of_images=len(normal_pylorus)
        start=int(number_of_images*i/9.0)
        count=start
        for index in range(start,number_of_images):
            file=normal_pylorus[index]
            if count<int(number_of_images*(i+1)/9.0-52) if i!=4 else count<int(number_of_images*(i+1)/9.0-53):
                tests[i].append(file)
                count+=1
            else:
                trains[i]+=diff(normal_pylorus,tests[i])
                break
        
        number_of_images=len(normal_z_line)
        start=int(number_of_images*i/9.0)
        count=start    
        for index in range(start,number_of_images):
            file=normal_z_line[index]
            if count<int(number_of_images*(i+1)/9.0-49):
                tests[i].append(file)
                count+=1
            else:
                trains[i]+=diff(normal_z_line,tests[i])
                break

        number_of_images=len(polyps)
        start=int(number_of_images*i/9.0)
        count=start
        for index in range(start,number_of_images):
            file=polyps[index]
            if count<int(number_of_images*(i+1)/9.0):
                tests[i].append(file)
                count+=1
            else:
                trains[i]+=diff(polyps,tests[i])
                break

        number_of_images=len(ulcerative_colitis)
        start=int(number_of_images*i/9.0)
        count=start
        for index in range(start,number_of_images):
            file=ulcerative_colitis[index]
            if count<int(number_of_images*(i+1)/9.0):
                tests[i].append(file)
                count+=1
            else:
                trains[i]+=diff(ulcerative_colitis,tests[i])
                break


    for i in range(len(trains)):
        train=trains[i]
        os.mkdir(start_dir + "/fold"+str(i+1))
        for index in range(0,len(train)):
            file=train[index]
            name=file[file.rfind("/")+1:]
            fold_path=start_dir + "/fold"+str(i+1)+"/train/"
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)
            newPath = shutil.copy(file, fold_path)
            

    for i in range(len(tests)):
        test=tests[i]
        for index in range(0,len(test)):
            file=test[index]
            name=file[file.rfind("/")+1:]
            fold_path=start_dir + "/fold"+str(i+1)+"/test/"
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)
            newPath = shutil.copy(file, fold_path)
    
    for i in range(5):
        for file in val_set:
            name=file[file.rfind("/")+1:]
            fold_path=start_dir + "/fold"+str(i+1)+"/val/"
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)
            newPath = shutil.copy(file, fold_path)

    os.rmdir(start_dir+"/test")
    os.rmdir(start_dir+"/train")
    os.rmdir(start_dir+"/val")
    