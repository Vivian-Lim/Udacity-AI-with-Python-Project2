# Package Imports
import argparse
import torch
from torchvision import datasets, transforms, models
import numpy as np
import json
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

test_dir = 'flower_data/test/'

# Training data augmentation, Data normalization, Data batching, Data loading
test_transforms  = transforms.Compose([transforms.Resize(250),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_imagenet_data = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_imagenet_data, batch_size=48, shuffle=True)

def get_input_args():
    parser = argparse.ArgumentParser(prog='viv Image Classifier Project 2')
    parser.add_argument('--arch', type=str, default='vgg16', choices = ["vgg16", "densenet121"])
    parser.add_argument('--device', type=str, default='gpu', choices = ['gpu', 'cpu'])
    parser.add_argument('--save', type=str, default="checkpoint_viv.pth")
    parser.add_argument('--predict_path', type=str, default='flower_data/test/74/image_01191.jpg')
    
    return parser.parse_args()

# Loading checkpoints
def load_checkpoint(checkpoint_path, arch):
    checkpoint = torch.load(checkpoint_path)

    if arch == 'vgg16':
        model = models.vgg16(weights=True)
    elif arch == 'densenet121':
        model = models.densenet121(weights=True)

    model = checkpoint['model']
    model.classifier = checkpoint['model_classifier']
    model.load_state_dict = checkpoint['model_state_dict']
    model.optimizer = checkpoint['optimizer_state_dict']
    model.class_to_idx = checkpoint['class_to_idx']

    for param in model.parameters():
        param.requires_grad = False

    return model, model.class_to_idx


def predict(topk=5):
    in_arg = get_input_args()
    arch = in_arg.arch
    compute_device = in_arg.device
    checkpoint_path = in_arg.save
    predict_path = in_arg.predict_path

    model, class_to_idx = load_checkpoint(checkpoint_path, arch)

    # Use GPU if it is available
    if (compute_device == 'gpu') and (torch.cuda.is_available()):
        device = torch.device('cuda')
        print('Testing with GPU')
    else:
        device = torch.device('cpu')
        print('Testing with CPU')
    model.to(device)


    process_img = process_image(predict_path)
    img = torch.from_numpy(process_img)
    img = img.unsqueeze(0).float()
    img = img.to(device)

    with torch.no_grad():
        model.eval()
        model.to(device)

        log_ps = model(img)
        ps = torch.exp(log_ps)
        probs, labels = ps.topk(topk, dim=1)

        probs_list = []
        class_list = []
        for i in range(topk):
            probs_list.append(probs.data[0][i].item())
            class_list.append(labels.data[0][i].item())
                   
        class_to_idx_dict = {}
        for i in class_to_idx:
            class_to_idx_dict[class_to_idx[i]] = i
            
        classes = []
        for i in class_list:
            classes.append(class_to_idx_dict[i])
            
        return probs_list, classes

# Image Processing
def process_image(image):
    img = Image.open(image)

    if img.size[0] > img.size[1]:
        resized_img = img.resize((int(256*(img.size[0]/img.size[1])), 256))
    else:
        resized_img = img.resize(256, (256/(img.size[0]/img.size[1])))

    crop_left = (resized_img.size[0] - 224) / 2
    crop_right = (resized_img.size[0] + 224) / 2
    crop_top = (resized_img.size[1] - 224) / 2
    crop_bottom = (resized_img.size[1] + 224) / 2
    cropped_img = resized_img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Convert the image to Numpy array
    np_img = np.array(cropped_img)

    # Normalize the image
    np_img = np_img / 255.0
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - means) / stds

    process_img = np_img.transpose((2, 0, 1))

    return process_img


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def main(): 
    in_arg = get_input_args()
    predict_path = in_arg.predict_path

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict()
    print(probs)
    print(classes)

    classes_name = []
    for i in classes:
        classes_name.append(cat_to_name[i])
    print(classes_name)

    # Sanity Checking with matplotlib
    fig = plt.figure(figsize = (5, 5))
    ax = plt.subplot(2,1,1)
    ax.set_title(classes_name[0])
    plt.axis('off')
    with Image.open(predict_path) as img: 
        imshow(process_image(predict_path), ax, title="lol");

    # Plotting probs in bar chart
    classes_name_number = [str(a) + " " + b for a, b in zip(classes, classes_name)]

    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=classes_name_number);
    plt.show()


if __name__ == '__main__':
    main()