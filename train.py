# Package Imports
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
# %matplotlib inline

train_dir = 'flower_data/train/'
valid_dir = 'flower_data/valid/'

# Training data augmentation, Data normalization, Data batching, Data loading
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),                          
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(250),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

train_imagenet_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_imagenet_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

train_loader = torch.utils.data.DataLoader(train_imagenet_data, batch_size=48, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_imagenet_data, batch_size=48, shuffle=True)


def get_input_args():
    parser = argparse.ArgumentParser(prog='viv Image Classifier Project 2')
    parser.add_argument('--arch', type=str, default='vgg16', choices = ['vgg16', 'densenet121'],  help='Model Architecture')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units')
    parser.add_argument('--device', type=str, default='gpu', choices = ['gpu', 'cpu'])
    # parser.add_argument('--data_path', type=str, default='flower_data')
    parser.add_argument('--save', type=str, default='checkpoint_viv.pth')
    
    return parser.parse_args()


def train():
    in_arg = get_input_args()

    arch = in_arg.arch
    epochs = in_arg.epochs
    learning_rate = in_arg.learning_rate
    hidden_units = in_arg.hidden_units
    compute_device = in_arg.device
    # data_path = in_arg.data_path
    checkpoint_path = in_arg.save

    if arch == 'vgg16':
        model = models.vgg16(weights=True)
        num_features = model.classifier[0].in_features
    elif arch == 'densenet121':
        model = models.densenet121(weights=True)
        num_features = model.classifier.in_features

    for param in model.parameters():
        param.requires_grad = False

    # Feedforward Classifier
    classifier = nn.Sequential(
            nn.Linear(num_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
            )
    model.classifier = classifier

    # Define the loss
    criterion = nn.NLLLoss()

    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    # Use GPU if it is available
    if (compute_device == 'gpu') and (torch.cuda.is_available()):
        device = torch.device('cuda')
        print('Training with GPU')
    else:
        device = torch.device('cpu')
        print('Training with CPU')
    model.to(device)

    steps = 0
    running_loss = 0
    running_losses = []
    valid_losses = []
    accuracy_list = []

    # Training the network
    print('Training started')
    for e in range(epochs):
          model.train() 
          for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            steps += 1

            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Validation Loss and Accuracy
            if steps % 10 == 0:
                valid_loss = 0
                valid_accuracy = 0
                model.eval()   

                with torch.no_grad():
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)

                        log_ps = model(images)
                        loss = criterion(log_ps, labels)
                        valid_loss += loss.item()

                        # Calculate validation accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                running_losses.append(running_loss/len(train_loader))
                valid_losses.append(valid_loss/len(valid_loader))
                accuracy_list.append(valid_accuracy/len(valid_loader))

                # Testing Accuracy, Validation Loss and Accuracy
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_losses[-1]),
                    "Validation Loss: {:.3f}.. ".format(valid_losses[-1]),
                    "Validation Accuracy: {:.3f}".format(valid_accuracy/len(valid_loader)))

                running_loss = 0
                model.train()   

    print('Training End')

    # Saving the model
    checkpoint = {
                'model': model,
                'model_classifier': classifier,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_to_idx': train_imagenet_data.class_to_idx
                }

    torch.save(checkpoint, checkpoint_path)
    print('Checkpoint saved')

    # Data to plot graph
    plt.figure(figsize=(14,5))
    plt.subplot(1, 2, 1)

    plt.plot(running_losses, label='running loss')
    plt.plot(valid_losses, label='valid loss')
    plt.xlabel('Steps')
    plt.ylabel('Losses')
    plt.title('Losses')
    plt.legend(frameon=True)

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_list, label='accuracy')
    plt.xlabel('Steps')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.legend(frameon=True)

    plt.show()


def main():
    train()


if __name__ == '__main__':
    main()