import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img


def load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width, img_height):
    wrong_name = dict()
    images = dict()
    texts = dict()
    count = 0
    for f in os.listdir(txt_dir_path):
        count= count+1
        filepath = os.path.join(txt_dir_path, f)

        if os.path.isfile(filepath) and f.endswith('.txt'):
            name = f.replace('.txt', '')
            print(name)
            try:
                texts[name] = open(filepath, 'rt').read()   #.decode('iso-8859-1').encode('utf-8')
                wrong_name[name] = False
            except:
                wrong_name[name] = True

    print(count)
    for f in os.listdir(img_dir_path):
        filepath = os.path.join(img_dir_path, f)
        if os.path.isfile(filepath) and f.endswith('.jpg'):
            name = f.replace('.jpg', '')
            if wrong_name[name]:
                continue
            images[name] = filepath

    result = []
    count = 0
    for name, img_path in images.items():
        if name in texts:
            if name == "magikarp":
                continue
            if wrong_name[name]:
                continue
            if (count == 7000):
                break

            text = texts[name]
            image = img_to_array(load_img(img_path, target_size=(img_width, img_height)))
            image = (image.astype(np.float32) / 255) * 2 - 1
            result.append([image, text])
            count = count+1

    print(len(result))

    return np.array(result)


def load_normalized_flowers_img_and_its_text(data_dir, img_width, img_height):

    img_dir = data_dir +'/flowers/jpg/'
    image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]
    image_captions = {img_file: [] for img_file in image_files}
    images = {}
    texts = {}
    for i in image_files:
        name = i.replace('.jpg', '')
        filepath = os.path.join(img_dir, i)
        images[name] = filepath

    result = []

    caption_dir = data_dir + '/flowers/text_c10/'

    class_dirs = []
    for i in range(1, 103):
        class_dir_name = '/class_%.5d/' % (i)
        class_dirs.append(caption_dir + class_dir_name)

    for class_dir in class_dirs:
        caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
        for cap_file in caption_files:
            with open(class_dir + cap_file) as f:
                captions = f.read()###.split('\n')
                name = cap_file.replace('.txt', '')
                texts[name] = captions

            img_file = cap_file[0:11] + ".jpg"
            # 5 captions per image
            image_captions[img_file] += [cap for cap in captions if len(cap) > 0][0:5]

    result = []
    count = 0
    for name, img_path in images.items():
        if name in texts:
            text = texts[name]
            image = img_to_array(load_img(img_path, target_size=(img_width, img_height)))
            image = (image.astype(np.float32) / 255) * 2 - 1
            result.append([image, text])
            count = count+1

    print(len(result))
    #print(result)
    return np.array(result)


def load_normalized_img_and_its_text_with_stageII(img_dir_path, txt_dir_path, img_width, img_height,
                                                  Simg_width, Simg_height):
    wrong_name = dict()
    images = dict()
    texts = dict()

    count = 0

    for f in os.listdir(txt_dir_path):
        count= count+1
        filepath = os.path.join(txt_dir_path, f)
        if os.path.isfile(filepath) and f.endswith('.txt'):
            name = f.replace('.txt', '')
            print(name)
            try:
                texts[name] = open(filepath, 'rt').read()   #.decode('iso-8859-1').encode('utf-8')
                wrong_name[name] = False
            except:
                wrong_name[name] = True

    print(count)
    for f in os.listdir(img_dir_path):
        filepath = os.path.join(img_dir_path, f)
        if os.path.isfile(filepath) and f.endswith('.jpg'):
            name = f.replace('.jpg', '')
            if wrong_name[name]:
                continue
            images[name] = filepath

    result = []
    Sresult = []
    count = 0
    for name, img_path in images.items():
        if name in texts:
            if name == "magikarp":
                continue
            if wrong_name[name]:
                continue
            if (count == 6000):
                break

            text = texts[name]
            image = img_to_array(load_img(img_path, target_size=(img_width, img_height)))
            Simage = img_to_array(load_img(img_path, target_size=(Simg_width, Simg_height)))
            image = (image.astype(np.float32) / 255) * 2 - 1
            Simage = (Simage.astype(np.float32) / 255) * 2 - 1
            result.append([image, text])
            Sresult.append([Simage, text])
            count = count+1

    print(len(result))

    return np.array(result), np.array(Sresult)
