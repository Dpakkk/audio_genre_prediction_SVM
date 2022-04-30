'''
This nmodel is for loading the training dataset and used it for our genre prediction.
'''
import torch
torch.manual_seed(123)
from torch.autograd import Variable

from config import GENRES, DATAPATH, MODELPATH
from model import genreNet
from data import Data
from set import Set


def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated
 
# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
 
# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
 
# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

def main():
    # data loading
    data    = Data(GENRES, DATAPATH)
    data.make_raw_data()
    data.save()
    data    = Data(GENRES, DATAPATH)
    data.load()
    set_    = Set(data)
    set_.make_dataset()
    set_.save()
    set_ = Set(data)
    set_.load()

    x_train, y_train    = set_.get_train_set()
    x_valid, y_valid    = set_.get_valid_set()
    x_test,  y_test     = set_.get_test_set()

    TRAIN_SIZE  = len(x_train)
    VALID_SIZE  = len(x_valid)
    TEST_SIZE   = len(x_test)

    net = genreNet()
    net.cuda()

    criterion   = torch.nn.CrossEntropyLoss()
    optimizer   = torch.optim.RMSprop(net.parameters(), lr=1e-4)

    EPOCH_NUM   = 250
    BATCH_SIZE  = 16

    for epoch in range(EPOCH_NUM):
        inp_train, out_train    = Variable(torch.from_numpy(x_train)).float().cuda(), Variable(torch.from_numpy(y_train)).long().cuda()
        inp_valid, out_valid    = Variable(torch.from_numpy(x_valid)).float().cuda(), Variable(torch.from_numpy(y_valid)).long().cuda()
        # train phase
        train_loss = 0
        optimizer.zero_grad()  # optimizer
        for i in range(0, TRAIN_SIZE, BATCH_SIZE):
            x_train_batch, y_train_batch = inp_train[i:i + BATCH_SIZE], out_train[i:i + BATCH_SIZE]

            pred_train_batch    = net(x_train_batch)
            loss_train_batch    = criterion(pred_train_batch, y_train_batch)
            train_loss          += loss_train_batch.data.cpu().numpy()[0]

            loss_train_batch.backward()
        optimizer.step()  # optimizer

        epoch_train_loss    = (train_loss * BATCH_SIZE) / TRAIN_SIZE
        train_sum           = 0
        for i in range(0, TRAIN_SIZE, BATCH_SIZE):
            pred_train      = net(inp_train[i:i + BATCH_SIZE])
            indices_train   = pred_train.max(1)[1]
            train_sum       += (indices_train == out_train[i:i + BATCH_SIZE]).sum().data.cpu().numpy()[0]
        train_accuracy  = train_sum / float(TRAIN_SIZE)

        # validation phase stats here
        valid_loss = 0
        for i in range(0, VALID_SIZE, BATCH_SIZE):
            x_valid_batch, y_valid_batch = inp_valid[i:i + BATCH_SIZE], out_valid[i:i + BATCH_SIZE]

            pred_valid_batch    = net(x_valid_batch)
            loss_valid_batch    = criterion(pred_valid_batch, y_valid_batch)
            valid_loss          += loss_valid_batch.data.cpu().numpy()[0]

        epoch_valid_loss    = (valid_loss * BATCH_SIZE) / VALID_SIZE
        valid_sum           = 0
        for i in range(0, VALID_SIZE, BATCH_SIZE):
            pred_valid      = net(inp_valid[i:i + BATCH_SIZE])
            indices_valid   = pred_valid.max(1)[1]
            valid_sum       += (indices_valid == out_valid[i:i + BATCH_SIZE]).sum().data.cpu().numpy()[0]
        valid_accuracy  = valid_sum / float(VALID_SIZE)

        print("Epoch: %d\t\tTrain loss : %.2f\t\tValid loss : %.2f\t\tTrain acc : %.2f\t\tValid acc : %.2f" % \
              (epoch + 1, epoch_train_loss, epoch_valid_loss, train_accuracy, valid_accuracy))


    # saving the generated model
    torch.save(net.state_dict(), MODELPATH)
    print('ptorch model is saved.')

    # evaluating the model
    inp_test, out_test = Variable(torch.from_numpy(x_test)).float().cuda(), Variable(torch.from_numpy(y_test)).long().cuda()
    test_sum = 0
    for i in range(0, TEST_SIZE, BATCH_SIZE):
        pred_test       = net(inp_test[i:i + BATCH_SIZE])
        indices_test    = pred_test.max(1)[1]
        test_sum        += (indices_test == out_test[i:i + BATCH_SIZE]).sum().data.cpu().numpy()[0]
    test_accuracy   = test_sum / float(TEST_SIZE)
    print("Test accuracy is : %.2f" % test_accuracy)
    # ------------------------------------------------------------------------------------------------- #

    return

if __name__ == '__main__':
    main()





