import os
import datetime
import click
import numpy as np
import tqdm

from deblurgan.utils import  write_log
from deblurgan.losses import wasserstein_loss, perceptual_loss
from deblurgan.model import generator_model, discriminator_model, generator_containing_discriminator_multiple_outputs

from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import os
from glob import glob
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
BASE_DIR = 'weights/'

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

def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)

def train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates=5):
    data = load_images('/home/anirud/projects/deblurgan/deblur-gan/Data/bottle/train', n_images)
    y_train, x_train = data['B'], data['A']

    g = generator_model()
    d = discriminator_model()
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)

    d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    
    loss_weights = [100, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True

    output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))

    log_path = './logs'
    tensorboard_callback = TensorBoard(log_path)

    for epoch in tqdm.tqdm(range(epoch_num)):
        permutated_indexes = np.random.permutation(x_train.shape[0])

        d_losses = []
        d_on_g_losses = []
        for index in range(int(x_train.shape[0] / batch_size)):
            print("hiiiiiiiiiiiiii")
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]

            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

            for _ in range(critic_updates):
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)

            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
            d_on_g_losses.append(d_on_g_loss)
            print(d_on_g_loss,"-------------------------------")

            d.trainable = True

        mean_d_loss = np.mean(d_losses)
        mean_d_on_g_loss = np.mean(d_on_g_losses)

        # Ensure loss is not NaN before saving weights
        if not np.isnan(mean_d_loss) and not np.isnan(mean_d_on_g_loss):
            print(mean_d_loss, mean_d_on_g_loss)
            save_all_weights(d, g, epoch, int(mean_d_on_g_loss))
        else:
            print(f"Skipping saving weights at epoch {epoch} due to NaN loss.")
        
        with open('log.txt', 'a+') as f:
            f.write('{} - {} - {}\n'.format(epoch, mean_d_loss, mean_d_on_g_loss))

        #save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))


@click.command()
@click.option('--n_images', default=-1, help='Number of images to load for training')
@click.option('--batch_size', default=2, help='Size of batch')
@click.option('--log_dir', required=True, help='Path to the log_dir for Tensorboard')
@click.option('--epoch_num', default=4, help='Number of epochs for training')
@click.option('--critic_updates', default=5, help='Number of discriminator training')
def train_command(n_images, batch_size, log_dir, epoch_num, critic_updates):
    return train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates)


if __name__ == '__main__':
    train_command()
