import csv
import os
import subprocess

#select classes you want to download at https://github.com/openimages/dataset/blob/master/dict.csv
CLASS_LIST = ('/m/01g317','/m/04yx4')
img_name = "111111111111"

# P0 安全修复（审计项 SEC-5）：
# 原实现 os.system("gsutil cp ...%s.jpg destination_path" % bbox[0])
# 把 CSV 数据字段直接拼进 shell 命令。数据文件若含 "; rm -rf /" 或 "$(...)"
# 之类元字符即可执行任意命令。现改为参数化调用（shell=False），
# 参数作为独立 argv 传递，不再经过 shell 解析。


def download_image(image_id, dest_dir="destination_path"):
    src = "gs://open-images-dataset/train/%s.jpg" % image_id
    subprocess.run(["gsutil", "cp", src, dest_dir], check=False)


#download csv from https://storage.googleapis.com/openimages/web/download.html
with open('path\\train-annotations-bbox.csv', newline='') as csvfile:
    bboxs = csv.reader(csvfile, delimiter=',', quotechar='|')
    for bbox in bboxs:
        if bbox[2] in CLASS_LIST:
            if img_name != bbox[0]:
                if not os.path.isfile("destination_path\\%s.jpg"%bbox[0]):
                    download_image(bbox[0])
                    out_file = open("destination_path\\%s.txt"%bbox[0], 'w')
                    img_name = bbox[0]
            if img_name == bbox[0]:
                out_file.write(str(CLASS_LIST.index(bbox[2])) + " " + str(float(bbox[4])+(float(bbox[5])-float(bbox[4]))/2) + " " + str(float(bbox[6])+(float(bbox[7])-float(bbox[6]))/2)+ " " + str(float(bbox[5])-float(bbox[4])) + " " + str(float(bbox[7])-float(bbox[6])) + '\n')
