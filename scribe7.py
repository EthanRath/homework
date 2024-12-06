#Code Adapted from Geeks for Geeks https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/

import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from torchmetrics.functional.clustering import mutual_info_score

class ANN(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size*2 ),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size*2, input_size*2),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size*2, 10),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.encoder(x)
    
class Decoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Softmax()
        )
 
    def forward(self, x):
        return self.decoder(x)
    
class Encoder(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size)
        )
 
    def forward(self, x):
        return self.encoder(x)
    
# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
    def __init__(self, encoding):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, encoding)
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
def train_ae(epochs, loss_function, loader, model, optimizer):
    losses = []

    for epoch in range(epochs):
        new_losses = []
        for (image, _) in loader:
            # Reshaping the image to (-1, 784)
            image = image.reshape(-1, 28*28).cuda()
            
            # Output of Autoencoder
            #reconstructed = model(image)
            encoding = model.encoder(image)
            reconstructed = model.decoder(encoding)

            # Calculating the loss function
            loss = loss_function(reconstructed, image)
            
            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Storing the losses in a list for plotting
            losses.append(loss.item())
            new_losses.append(loss.item())

        new_losses = torch.tensor(new_losses)

        print(f"Epoch {epoch} Loss {torch.mean(new_losses)}")
        #outputs.append((epochs, image, reconstructed))

    return reconstructed, model

def train_predictor(epochs, encoder, pred, opt, loader):
    loss_function = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        new_losses = []
        for (image, label) in loader:
            # Reshaping the image to (-1, 784)
            image = image.reshape(-1, 28*28).cuda()
            #label = image[:, 28*14+14]>= .5
            label = label.long().cuda()
            with torch.no_grad():
                encoding = encoder(image)
            yhat = pred(encoding)

            # Calculating the loss function
            loss = loss_function(yhat, label)
            
            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Storing the losses in a list for plotting
            new_losses.append(loss.item())
        new_losses = torch.tensor(new_losses)

        print(f"    Predictor Epoch {epoch} Loss {torch.mean(new_losses)}")
        #outputs.append((epochs, image, reconstructed))
    return pred

def train_decoder(epochs, encoder, decoder, opt, loader):
    loss_function = torch.nn.MSELoss()
    for epoch in range(epochs):
        new_losses = []
        for (image, _) in loader:
            # Reshaping the image to (-1, 784)
            image = image.reshape(-1, 28*28).cuda()
            with torch.no_grad():
                encoding = encoder(image)
            reconstructed = decoder(encoding)

            # Calculating the loss function
            loss = loss_function(reconstructed, image)
            
            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Storing the losses in a list for plotting
            new_losses.append(loss.item())
        new_losses = torch.tensor(new_losses)

        print(f"    Decoder Epoch {epoch} Loss {torch.mean(new_losses)}")
        #outputs.append((epochs, image, reconstructed))
    return decoder


def train_bottleneck(epochs, inner_epochs, encoder, decoder, predictor, opt_enc, loader, encoding_size, beta = .5):
    losses = []
    for epoch in range(epochs):
        dec = decoder(encoding_size).cuda()
        pred = predictor(encoding_size).cuda()
        opt_dec = torch.optim.Adam(dec.parameters(),lr = 1e-2, weight_decay = 1e-8)
        opt_pred = torch.optim.Adam(pred.parameters(),lr = 1e-2, weight_decay = 1e-8)
        mse = torch.nn.MSELoss()
        ce = torch.nn.CrossEntropyLoss()

        pred = train_predictor(inner_epochs, encoder, pred, opt_pred, loader)
        dec = train_decoder(inner_epochs, encoder, dec, opt_dec, loader)
        pred.eval()
        dec.eval()
        pred.requires_grad = False
        dec.requires_grad = False

        for inner_epoch in range(2):
            new_losses = []
            for (image, label) in loader:
                # Reshaping the image to (-1, 784)
                image = image.reshape(-1, 28*28).cuda()
                #label = image[:, 28*14+14]>= .5
                label = label.long().cuda()
                encoding = encoder(image)
                
                #with torch.no_grad():
                probs = pred(encoding)#.argmax(dim=1).long()
                reconstructed = dec(encoding)

                # Calculating the loss function
                loss = ce(probs, label) - (beta*mse(reconstructed, image)) #loss_function(reconstructed, image)
                #print(loss)
                
                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                opt_enc.zero_grad()
                loss.backward()
                opt_enc.step()
                
                # Storing the losses in a list for plotting
                new_losses.append(loss.item())
        new_losses = torch.tensor(new_losses)

        print(f"Encoder Epoch {epoch} Loss {torch.mean(new_losses)}")
        losses.append(torch.mean(new_losses))
        #outputs.append((epochs, image, reconstructed))

    plt.figure(dpi = 150)
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Curve for IT-Auto-Encoder")
    plt.savefig("mnist_images/loss.png")

    return encoder, dec, pred



def save_image(image, filename):
    data = image.reshape(-1, 28, 28)[0]
    sizes = np.shape(data)  
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[1] / sizes[0], 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ax.imshow(data, cmap="gray")
    plt.savefig(filename, dpi = sizes[0]) 
    plt.close()

if __name__ == "__main__":
    # Transforms images to a PyTorch Tensor
    try:os.mkdir("mnist_images")
    except: pass

    tensor_transform = transforms.ToTensor()
    n = 5

    # Download the MNIST Dataset
    dataset = datasets.MNIST(root = "./data", train = True, download = True, transform = tensor_transform)
    test = datasets.MNIST(root = "./data", train = False, download = True, transform = tensor_transform)

    # DataLoader is used to load the dataset 
    # for training
    loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 64,shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test, batch_size = 64,shuffle = False)

    # # Code for Training Normal AE
    # model = AE(n).cuda()
    loss_function = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-8)
    # images, model = train_ae(20, loss_function, loader, model, optimizer)
    # model.eval()

    # pred = ANN(n).cuda()
    # opt_pred = torch.optim.Adam(pred.parameters(),lr = 1e-2, weight_decay = 1e-8)
    # mse = torch.nn.MSELoss()
    # pred = train_predictor(25, model.encoder, pred, opt_pred, loader)

    # acc = 0
    # index = 0
    # count = 0
    # mse = 0
    # #Evaluate AE
    # for (image, label) in test_loader:
    #     # Reshaping the image to (-1, 784)
    #     image = image.reshape(-1, 28*28).cuda()
    #     #label = image[:, 28*14+14]>= .5
    #     label = label.long().cuda()

    #     encoding = model.encoder(image)
    #     reconstructed = model.decoder(encoding)
    #     yhat = pred(encoding).argmax(dim=1)
    #     acc += torch.sum(yhat == label)
    #     mse += loss_function(reconstructed, image)
    #     count += len(image)

    #     reconstructed = reconstructed.detach().cpu().numpy()

    #     if index < 5:
    #         save_image(image.cpu().numpy()[0], f"mnist_images/Original-{index}.png")
    #         save_image(reconstructed[0], f"mnist_images/AE-{index}.png")
    #     index += 1
    # print("Accuracy", acc/count)
    # print("MSE", mse/index)

    model = Encoder(n).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-8)
    encoder, _, _ = train_bottleneck(10, 5, model, Decoder, ANN, optimizer, loader, n, beta = .1)
    encoder.eval()
    

    #Fit Decoder and Predictor to our final encoder
    dec = Decoder(n).cuda()
    pred = ANN(n).cuda()
    opt_dec = torch.optim.Adam(dec.parameters(),lr = 1e-2, weight_decay = 1e-8)
    opt_pred = torch.optim.Adam(pred.parameters(),lr = 1e-2, weight_decay = 1e-8)
    mse = torch.nn.MSELoss()

    pred = train_predictor(25, encoder, pred, opt_pred, loader)
    dec = train_decoder(25, encoder, dec, opt_dec, loader)

    dec.eval()
    pred.eval()

    index = 0
    acc = 0
    count = 0
    mse = 0
    for (image, label) in test_loader:
        # Reshaping the image to (-1, 784)
        image = image.reshape(-1, 28*28).cuda()
        #label = image[:, 28*14+14]>= .5
        label = label.long().cuda()

        encoding = encoder(image)
        reconstructed = dec(encoding)
        yhat = pred(encoding).argmax(dim=1)
        acc += torch.sum(yhat == label)
        mse += loss_function(reconstructed, image)
        count += len(image)

        
        # Output of Autoencoder
        reconstructed = dec(model(image)).detach().cpu().numpy()

        if index < 5:
            save_image(image.cpu().numpy()[0], f"mnist_images/Original-{index}.png")
            save_image(reconstructed[0], f"mnist_images/ITAE-{index}.png")
        index += 1
    print("Accuracy", acc/count)
    print("MSE", mse/index)
    


