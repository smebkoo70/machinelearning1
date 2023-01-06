import glob
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    img_path = glob.glob("../dataset/faces/*/*_4.pgm")
    #img_path = glob.glob(r"..\dataset\faces\*\*_4.pgm")
    #img_path = glob.glob(r"D:\pythonProject\machineleanring1\machinelearning1\dataset\faces\*\*_4.pgm")
    target = "../dataset/data1/"
    #target = "D:\\pythonProject\\machineleanring1\\machinelearning1\\dataset\\data1\\"
    for path in tqdm(img_path):
        path = path.replace("\\", "/")
        img_name = path.split('/')[-1]
        save_path = target + img_name
        shutil.move(path, save_path)
