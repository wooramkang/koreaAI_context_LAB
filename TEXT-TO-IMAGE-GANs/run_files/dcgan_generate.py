import os
import sys
import numpy as np
from random import shuffle


def main():
    seed = 42
    np.random.seed(seed)

    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    img_dir_path = '/home/rd/recognition_research/IMAGE_CAPTION/DATASET/val2014'
    txt_dir_path = '/home/rd/recognition_research/IMAGE_CAPTION/DATASET/captions'
    model_dir_path = current_dir + '/models'

    img_width = 64
    img_height = 64

    from TXT2IMAGE.library.dcgan_v3 import DCGanV3
    from TXT2IMAGE.library.utility.image_utils import img_from_normalized_img
    from TXT2IMAGE.library.utility.img_cap_loader import load_normalized_img_and_its_text

    image_label_pairs = load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width=img_width, img_height=img_height)

    shuffle(image_label_pairs)

    gan = DCGanV3()
    gan.load_model(model_dir_path)

    for i in range(10):
        image_label_pair = image_label_pairs[i]
        normalized_image = image_label_pair[0]
        text = image_label_pair[1]

        image = img_from_normalized_img(normalized_image)
        image.save(current_dir + '/data/outputs/' + DCGanV3.model_name + '-generated-' + str(i) + '-0.png')


        for j in range(3):
            generated_image = gan.generate_image_from_text(text)
            generated_image.save(current_dir + '/data/outputs/' + DCGanV3.model_name + '-generated-' + str(i) + '-' + str(j) + '.png')


if __name__ == '__main__':
    main()
