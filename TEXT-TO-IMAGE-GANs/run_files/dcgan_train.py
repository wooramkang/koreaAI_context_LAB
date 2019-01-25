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

    #img_dir_path = current_dir + '/data/demoset/img'
    img_dir_path = '/home/rd/recognition_research/IMAGE_CAPTION/DATASET/val2014'
    txt_dir_path = '/home/rd/recognition_research/IMAGE_CAPTION/DATASET/captions'
    data_path = '/home/rd/recognition_research/IMAGE_CAPTION/DATASET'
    model_dir_path = current_dir + '/models'

    img_width = 64
    img_height = 64

    stage_width = 128
    stage_height = 128

    img_channels = 3

    from TXT2IMAGE.library.dcgan_v3 import DCGanV3
    from TXT2IMAGE.library.utility.img_cap_loader import load_normalized_img_and_its_text
    from TXT2IMAGE.library.utility.img_cap_loader import load_normalized_img_and_its_text_with_stageII
    from TXT2IMAGE.library.utility.img_cap_loader import load_normalized_flowers_img_and_its_text

    image_label_pairs = load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width=img_width, img_height=img_height)
    #image_label_pairs = load_normalized_flowers_img_and_its_text(data_path,img_width,img_height)
    #shuffle(image_label_pairs)
    #image_label_pairs, Slabel_pairs = load_normalized_img_and_its_text_with_stageII(img_dir_path, txt_dir_path, img_width=img_width,
    #                                                                  img_height=img_height,Simg_height=stage_height,Simg_width=stage_width)


    gan = DCGanV3()
    gan.img_width = img_width
    gan.img_height = img_height

    gan.img_channels = img_channels
    gan.random_input_dim = 25
    gan.glove_source_dir_path = './very_large_data'

    batch_size = 32
    epochs = 50000

    gan.fit_wasserstein(model_dir_path=model_dir_path, image_label_pairs=image_label_pairs,
                        snapshot_dir_path=current_dir + '/data/snapshots',
                        snapshot_interval=20,
                        batch_size=batch_size,
                        epochs=epochs)

    '''
    gan.fit_mono(model_dir_path=model_dir_path, image_label_pairs=image_label_pairs,
            snapshot_dir_path=current_dir + '/data/snapshots',
            snapshot_interval=20,
            batch_size=batch_size,
            epochs=epochs)

    gan.fit_DOUBLE(model_dir_path=model_dir_path, image_label_pairs=image_label_pairs,
            snapshot_dir_path=current_dir + '/data/snapshots',
            snapshot_interval=20,
            batch_size=batch_size,
            epochs=epochs)


    gan.fit_with_stageII(model_dir_path=model_dir_path, image_label_pairs=image_label_pairs,s_label_pairs=Slabel_pairs,
            snapshot_dir_path=current_dir + '/data/snapshots',
            snapshot_interval=100,
            batch_size=batch_size,
            epochs=epochs)
    '''
if __name__ == '__main__':
    main()
