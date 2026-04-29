"""
Class to encapsulate all training methods 

methods as pr notebook

count_parameters
calculate_accuracy
epoch_time
train
validation

"""

from modelBuilder import *
import torch

class Trainer():
    #train: added classes list 
    def train(model, training_batch, batch_size, optimizer, criterion, device, classes_list):

        loss = 0
        acc = 0

        for i in range(batch_size):

            img = Image.open('standardized_256/' + classes_list[i] + '/' + training_batch[i]) #.convert('L')
            img = np.asarray(img)/255

            data = torch.from_numpy(img)
            x = data.to(device)

            desired_labels = [0.0] * 10
            desired_labels[i] = 1.0

            y = torch.tensor(desired_labels, dtype=torch.long, device=device)
            y_pred = model.forward(x)

            loss += criterion(y_pred[0], y.float())
            acc += calculate_accuracy(y_pred, y.float())

        # Get the average loss over the batch
        loss /= batch_size
        acc /= batch_size

        # Backpropagation
        optimizer.zero_grad() #changed from CNN_optimizer
        loss.backward()
        optimizer.step()

        epoch_loss = loss.item()
        #epoch_acc = acc.item()

        return epoch_loss, acc

        #validation
    def validation(model, validation_set, batch_size, criterion, device):

        epoch_loss = 0
        epoch_acc = 0

        with torch.no_grad():

            for i in range(10):

                img = Image.open('standardized_256/' + classes_list[i] + '/' + validation_set[i]) #.convert('L')
                img = np.asarray(img)/255

                data = torch.from_numpy(img)
                x = data.to(device)

                desired_labels = [0.0] * 10
                desired_labels[i] = 1.0

                y = torch.tensor(desired_labels, dtype=torch.long, device=device)

                y_pred = model.forward(x)

                loss = criterion(y_pred[0], y.float())

                acc = calculate_accuracy(y_pred, y.float())

                epoch_loss += loss.item()
                #epoch_acc += acc.item()

        return epoch_loss / batch_size, acc / batch_size