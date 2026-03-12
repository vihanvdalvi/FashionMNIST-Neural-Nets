import torch 
import torch.nn as nn # for building the neural networks
import torch.nn.functional as F # for activation functions like ReLU, softmax
import torch.optim as optim # for optimization algorithms like SGD, Adam, etc.
from torchvision import datasets, transforms # for loading and transforming the dataset
import random # for seeding results



def get_data_loader(training = True):
    """
    Load FashionMNIST data for training or evaluation.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if training == True:
        # download the training set of FashionMNIST dataset and apply the custom_transform to it.
        train_set = datasets.FashionMNIST('./data', train = True, download = True, transform = custom_transform)
        # so random sets of 64 will be used to train model in every epoch
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)

    elif training == False:
        # download the test set of FashionMNIST dataset and apply the custom_transform to it.
        test_set = datasets.FashionMNIST('./data', train = False, download = True, transform = custom_transform)
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False)
        
    return loader

"""
    1. Define Network
    2. Feed data through it 
    3. Compute the loss
    4. Backpropagate the gradients
    5. Update the weights (weight = weight - learning_rate * gradient)
"""

def build_model():
    """
    Build a baseline feed-forward classifier for FashionMNIST.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    
    model = nn.Sequential(
        # Flatten the input image of shape 28x28 into a dimensions of 1x784 because linear layers use 1D input and output 
        # Just serves to reformat the input data to be fed into the linear layer. It does not have any learnable parameters.
        nn.Flatten(),
        # Linear layer (which are fully connected) with 128 nodes/neurons
        # input dimension is 784 (28x28) and output dimension is 128
        nn.Linear(in_features = 28*28, out_features = 128),
        # Leaky ReLU activation function (negative slope = 0.01)
        nn.LeakyReLU(negative_slope = 0.01),
        # layer with 64 nodes
        nn.Linear(in_features = 128, out_features = 64),
        # Leaky ReLU activation function (negative slope = 0.01)
        nn.LeakyReLU(negative_slope = 0.01),
        # output layer with 10 nodes (one for each class)
        nn.Linear(in_features = 64, out_features = 10)
        # Note that we do not apply softmax activation function to the output layer since the loss function we will use (cross-entropy) will apply it for us.
    )
    
    return model


def build_deeper_model():
    """
    Build a deeper feed-forward classifier for FashionMNIST.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = 28*28, out_features = 256), # layer 1
        nn.LeakyReLU(negative_slope = 0.01),
        nn.Linear(in_features = 256, out_features = 128), # layer 2
        nn.LeakyReLU(negative_slope = 0.01),
        nn.Linear(in_features = 128, out_features = 64), # layer 3
        nn.LeakyReLU(negative_slope = 0.01),
        nn.Linear(in_features = 64, out_features = 32), # layer 4
        nn.LeakyReLU(negative_slope = 0.01),
        nn.Linear(in_features = 32, out_features = 10) # layer 5
        # again do not need to apply softmax activation function to the output layer since the loss function we will use (cross-entropy) will apply it for us.
    )
    
    return model


def train_model(model, train_loader, criterion, T):
    """
    Train a model for T epochs on the provided training DataLoader.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    
    # optimizer for updating the weights of the model. 
    # In SGD, only a random subset of training samples are used to update parameters. This converges faster than GD (uses all samples) for large datasets
    # The extra noise helps generalization and prevents overfitting (by model learning input order, etc.)
    # Momentum is simply a technique to speed up convergence and dampen oscillations. Does this by adding fraction of update vector of past time step to current update vector.
    optimizer = optim.SGD(model.parameters(), lr = 0.0195, momentum = 0.71)
    
    # T passes through the full dataset
    for epoch in range(T):
        
        # set the model to training mode
        model.train()
        
        # count how many predictions are correct in this epoch
        numberOfCorrectPredictions = 0.0
        # total images seen in this epoch
        totalImages = 0.0
        # total loss across all batches in this epoch
        sumOfLosses = 0.0
        
        # loop through the training data in random batches
        # this will return a batch of images and their corresponding labels. 
        for images, labels in train_loader:
            
            # zero the parameter gradients FIRST
            # if we do not zero them, the gradients from the previous batch will be added to the gradients of the current batch, which will lead to incorrect updates of the model parameters.
            optimizer.zero_grad()
            
            # forward pass: compute the output of the model for the input images
            outputs = model(images)
            
            # compute the loss between the predicted outputs and the true labels
            # the criterion (cross-entropy) will apply softmax activation function to the output layer for us, so we do not need to apply it here.
            loss = criterion(outputs, labels)
            
            # loss.item() returns average loss for this batch as a float since loss is tensor.
            sumOfLosses += loss.item()
            # number of correct predictions in this batch added to total number of correct predictions so far
            numberOfCorrectPredictions += (outputs.argmax(dim = 1) == labels).sum().item()
            # total number of images in this batch added to total number of images so far
            totalImages += labels.shape[0]
            
            # backward pass: compute the gradients of the loss with respect to the model parameters
            loss.backward()
            
            # update the model parameters using the optimizer
            optimizer.step()
        
        # compute the accuracy of the model on the current batch which will also be a tensor
        accuracy_percentage = round((numberOfCorrectPredictions / totalImages) * 100, 2)
        # average loss for all batches (sum of average loss for each batch / # of batches)
        avg_loss = sumOfLosses / len(train_loader)
        
        print(f'Train Epoch: {epoch} Accuracy: {int(numberOfCorrectPredictions)}/{int(totalImages)}({accuracy_percentage}%) Loss: {round(avg_loss, 3)}')


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    Evaluate a trained model on the test DataLoader.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    
    # set the model to evaluation mode
    model.eval() 
    
    # we do not need to compute gradients during evaluation since we are not updating the model parameters, so we can save memory and computation by using torch.no_grad()
    # this will disable gradient calculation
    with torch.no_grad(): 
        
        sum_of_test_loss = 0.0
        correct_image_predictions = 0.0
        totalImages = 0.0
        
        for data, labels in test_loader:
            
            # forward pass: compute the output of the model for the input images
            output = model(data)
            
            # calculate loss 
            # loss.item() returns average loss for this batch as a float since loss is tensor.
            sum_of_test_loss += criterion(output, labels).item()
            
            # calculate accuracy
            # .item() returns the number of correct predictions in this batch as a float since it is a tensor.
            correct_image_predictions += (output.argmax(dim = 1) == labels).sum().item()
            
            # total number of images in this batch added to total number of images so far
            totalImages += labels.shape[0]

        if show_loss == True:
            # calculate average loss for all batches (sum of average loss for each batch / # of batches)
            average_test_loss = sum_of_test_loss / len(test_loader)
            print(f'Average loss: {round(average_test_loss, 4)}')
            # calculate accuracy percentage 
            accuracy_percentage = round((correct_image_predictions / totalImages) * 100, 2)
            print(f'Accuracy: {accuracy_percentage}%')
        elif show_loss == False:
            # calculate only accuracy percentage 
            accuracy_percentage = round((correct_image_predictions / totalImages) * 100, 2)
            print(f'Accuracy: {accuracy_percentage}%')


def predict_label(model, test_images, index):
    """
    Print top-3 class predictions for a single test image.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    
    # access specific image
    image = test_images[index]
    
    # images already in 1x28x28 shape, so just need to add batch dimension to make it 1x1x28x28
    # model needs to process batches not a single image (make a batch of 1 image) 
    image = image.unsqueeze(0)
    
    # set model to evaluation mode
    model.eval()
    
    # class names 
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    
    # disable gradient calculation since we are not updating model parameters
    with torch.no_grad():
        
        # model outputs 10 logits (each corresponding to a class as a tensor([[2.1, -0.5, .....]])
        # higher the number, higher the probability that the image belongs to that class
        output = model(image)
        
        # convert logits to probabilities using softmax function
        # dim tells which dimension to apply softmax across. dim = 0 will be rows. dim = 1 will be columns. 
        prob = F.softmax(output, dim = 1)
        
        # print the top 3 predicted class labels and their probabilities
        # this will get the top 3 probabilities and their corresponding class indices
        top_probs, top_classes = prob.topk(3, dim = 1)
        for i in range(3):
            print(f'{class_names[top_classes[0,i]]}: {round(top_probs[0,i].item() * 100, 2)}%')
        

if __name__ == '__main__':
    '''
    Optional local run block for quick experimentation and sanity checks.
    '''
    
    # set seed so results are reproducible
    seed = 31
    random.seed(seed)
    torch.manual_seed(seed)
    
    criterion = nn.CrossEntropyLoss()
    
    # data loaders for training and test sets
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)
    print(type(test_loader))
    print(test_loader.dataset)
    
     # first model
    model = build_model()
    print(model)
    
    # train model
    # should be able to reach at least 80% accuracy after 5 epochs of training
    train_model(model, train_loader, criterion, T = 7) # 7 is max beyond that overfitting starts to occur
    # test model
    evaluate_model(model, test_loader, criterion, show_loss = False)
    evaluate_model(model, test_loader, criterion, show_loss = True)
    
    # deeper model 
    deeper_model = build_deeper_model()
    print(deeper_model)
    
    # train deeper model
    train_model(deeper_model, train_loader, criterion, T = 14)
    # test model
    evaluate_model(deeper_model, test_loader, criterion, show_loss = False)
    evaluate_model(deeper_model, test_loader, criterion, show_loss = True)
    