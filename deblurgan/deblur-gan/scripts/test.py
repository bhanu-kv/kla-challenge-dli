import numpy as np
from PIL import Image
import click

from deblurgan.model import generator_model
from deblurgan.utils import deprocess_image
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import os
from glob import glob
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
def load_images(directory, n_images=-1):
    # Paths for blurred and sharp images
    blur_image_paths = sorted(glob(os.path.join(directory, 'blur/*.png')))
    sharp_image_paths = sorted(glob(os.path.join(directory, 'sharp/*.png')))
    
    if n_images > 0:
        blur_image_paths = blur_image_paths[:n_images]
        sharp_image_paths = sharp_image_paths[:n_images]

    blur_images = []
    sharp_images = []
    
    for blur_path, sharp_path in zip(blur_image_paths, sharp_image_paths):
        blur_img = load_img(blur_path, target_size=(256, 256))  # Resize to match model input size
        sharp_img = load_img(sharp_path, target_size=(256, 256))  # Resize to match model input size
        blur_images.append(img_to_array(blur_img))
        sharp_images.append(img_to_array(sharp_img))

    # Normalize pixel values to range [-1, 1] for the generator
    blur_images = np.array(blur_images) / 127.5 - 1
    sharp_images = np.array(sharp_images) / 127.5 - 1
    
    return {'A': blur_images, 'B': sharp_images}

def test(batch_size):
    data = load_images('/home/anirud/projects/deblurgan/deblur-gan/Data/bottle/test',2 )
    y_test, x_test = data['B'], data['A']
    g = generator_model()
    g.load_weights('/home/anirud/projects/deblurgan/deblur-gan/weights/1019/generator_3_2004.h5')
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('results{}.png'.format(i))


@click.command()
@click.option('--batch_size', default=1, help='Number of images to process')
def test_command(batch_size):
    return test(batch_size)


if __name__ == "__main__":
    test_command()
