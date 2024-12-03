from PIL import Image, ImageFile
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import torch
from matplotlib import animation
import moviepy.video.io.ImageSequenceClip
import os
import sys

def save_image(data, filename, bw):
    sizes = np.shape(data)  
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[1] / sizes[0], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if bw:
        ax.imshow(data, cmap="gray")
    else:
        ax.imshow(data)
    plt.savefig(filename, dpi = sizes[0]) 
    plt.close()

def save_animation(path_to_images, path_to_videos, num_images):
    images = []
    for img_number in range(1,num_images+1): 
        images.append(Image.open(f"{path_to_images}compressed_{img_number}.jpg"))
    images[0].save(f'{path_to_videos}anim.gif', 
               save_all = True, append_images = images[1:],  
               optimize = False, duration = num_images*10, loop = 0)

def compress(image, k = 2, bw = True):
    if bw:
        temp = image.flatten()
        kmeans = KMeans(n_clusters = k).fit(temp.reshape(-1,1))
    else:
        temp = np.reshape(image, (image.shape[0]*image.shape[1], 3 ) )
        kmeans = KMeans(n_clusters = k).fit(temp)
    labels = kmeans.labels_
    clusters = kmeans.cluster_centers_
    #print(clusters)
    new_data = np.zeros(temp.shape)
    for i in range(len(temp)):
        new_data[i] = clusters[labels[i]]
    if bw:
        new_data = torch.tensor(new_data).unflatten(dim = 0, sizes = image.shape)
    else:
        new_data = torch.tensor(new_data)
        new_data = torch.reshape(new_data, image.shape)
    return new_data.numpy(), kmeans

def measure_distortion(img1, img2):
    return np.mean((img1-img2)**2)

def compute_entropy(kmeans, img_size):
    total = len(kmeans.labels_)
    entropy = 0
    for i in range(len(kmeans.cluster_centers_)):
        prob = np.sum(kmeans.labels_ == i)/total
        entropy -= prob*np.log2(prob)
    return entropy, entropy*img_size




if __name__ == "__main__":
    image_name = "000012.JPG"
    bw = False

    save_path = image_name[:image_name.index(".")] + "/"
    try:os.mkdir(save_path)
    except:pass

    img = Image.open(image_name)
    trans = transforms.ToTensor()
    trans2 = transforms.ToPILImage()

    if bw:
        data = trans(img).numpy()[0]
    else:
        data = trans(img).numpy()
        data = torch.tensor(data).permute(1,2,0).numpy()
    save_image(data, f"{save_path}orig.jpg", bw)
    total_pixels = len(data)*len(data[0])

    num_samples = 32
    ks = [i+1 for i in range(num_samples)]
    #ks = [i for i in range(1,33)]
    distortions = []
    entropys = []
    file_sizes = []
    for k in ks:
        print(f"Compressing Image with k = {k}", end = "\r")
        new_data, kmeans = compress(data, k, bw)
        distortions.append(measure_distortion(data, new_data))
        ent, bits = compute_entropy(kmeans, total_pixels)
        entropys.append(ent)
        file_sizes.append(bits/8000) #convert to kilobytes
        save_image(new_data, f"{save_path}compressed_{k}.jpg", bw)
    save_animation(save_path, save_path, num_samples)
    print("Distortions: ", distortions)
    print("Entropies: ", entropys)
    print("File Sizes: ", file_sizes)

    plt.figure(dpi = 150)
    plt.plot(ks, distortions)
    plt.title("MSE Distortion As a Function of # Clusters")
    plt.xlabel("# Clusters")
    plt.ylabel("Distortion (MSE)")
    plt.savefig(f"{save_path}distortion.png")

    plt.figure(dpi = 150)
    plt.plot(ks, entropys)
    plt.title("Entropy As a Function of # Clusters")
    plt.xlabel("# Clusters")
    plt.ylabel("Entropy (bits per pixel)")
    plt.savefig(f"{save_path}entropy.png")

    plt.figure(dpi = 150)
    plt.plot(distortions, entropys)
    plt.title("Entropy As a Function of # Distortion")
    plt.xlabel("Distortion (MSE)")
    plt.ylabel("Entropy (bits per pixel)")
    plt.savefig(f"{save_path}entropy_distort.png")

    plt.figure(dpi = 150)
    plt.plot(ks, file_sizes)
    plt.title("File Size As a Function of # Clusters")
    plt.xlabel("# Clusters")
    plt.ylabel("File Size (kb)")
    plt.savefig(f"{save_path}file_size.png")

    plt.figure(dpi = 150)
    plt.plot(distortions, file_sizes)
    plt.title("File Size As a Function of # Distortion")
    plt.xlabel("Distortion (MSE)")
    plt.ylabel("File Size (kb)")
    plt.savefig(f"{save_path}file_size_distort.png")

    # #save_image(trans(img).numpy()[0], "test.jpg")


