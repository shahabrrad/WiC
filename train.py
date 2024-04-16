import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
torch.set_default_tensor_type(torch.FloatTensor)
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cpu")

'''
NOTE: Do not change any of the statements above regarding random/numpy/pytorch.
You can import other built-in libraries (e.g. collections) or pre-specified external libraries
such as pandas, nltk and gensim below. 
Also, if you'd like to declare some helper functions, please do so in utils.py and
change the last import statement below.
'''

import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

from neural_archs import DAN, RNN, LSTM
from utils import WiCDataset, get_wordnet_features, collate_function, calculate_accuracy, initialize_gensim_embedding, plot_loss_history, Vocab
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # TODO: change the `default' attribute in the following 3 lines with the choice
    # that achieved the best performance for your case
    parser.add_argument('--neural_arch', choices=['dan', 'rnn', 'lstm'], default='lstm', type=str)
    parser.add_argument('--rnn_bidirect', default=True, action='store_true')
    parser.add_argument('--init_word_embs', choices=['scratch', 'glove'], default='glove', type=str)

    args = parser.parse_args()

    embedding_dim = 50
    vocab = Vocab()
    train_dataset = WiCDataset("WiC_dataset/train/train.data.txt", "WiC_dataset/train/train.gold.txt")
    vocab = train_dataset.build_vocab(vocab, "WiC_dataset/train/train.data.txt")

    valid_dataset = WiCDataset("WiC_dataset/dev/dev.data.txt", "WiC_dataset/dev/dev.gold.txt")
    vocab = valid_dataset.build_vocab(vocab, "WiC_dataset/train/train.data.txt")

    test_dataset = WiCDataset("WiC_dataset/test/test.data.txt", "WiC_dataset/test/test.gold.txt")
    vocab = test_dataset.build_vocab(vocab, "WiC_dataset/train/train.data.txt")
    vocab_size = vocab.idx

    if args.init_word_embs == "glove":
        # TODO: Feed the GloVe embeddings to NN modules appropriately
        # for initializing the embeddings
        glove_embs = api.load("glove-wiki-gigaword-50")
    
    if args.init_word_embs == "glove":
        embeddings = initialize_gensim_embedding(glove_embs, train_dataset.vocab, vocab_size, embedding_dim=50)
    else:
        embeddings = None


    hidden_dims = [32, 64, 128, 256]
    lrs = [0.000001, 0.00001, 0.0001]
    lambda_l1s = [0, 0.00001, 0.0001]
    droputs = [0.7, 0.5, 0.3, 0.1]
    weight_decay = [0, 0.00001, 0.0001, 0.001]

    # TODO: Freely modify the inputs to the declaration of each module below
    if args.neural_arch == "dan":
        embedding_dim = 50
        hidden_dim = 128
        learning_rate = 0.0001
        batch_size = 16
        num_epochs = 100
        lambda_l1 =  0.00001
        weight_decay=1e-3
        model = DAN(vocab_size, embedding_dim, hidden_dim, embeddings=embeddings, output_dim=1).to(torch_device)
    elif args.neural_arch == "rnn":
        embedding_dim = 50
        hidden_dim = 128 #128 for RNN
        learning_rate = 0.001 #0.001 / for lstm
        learning_rate = 0.00001 #||||| 0.00001 for dan ||| 0.00001 for rnn ||| 0.00001 for lstm
        learning_rate = 0.0001
        batch_size = 16 #32 for dan |||| 16 for rnn
        num_epochs = 100
        lambda_l1 =  0.00001
        weight_decay = 1e-4
        if args.rnn_bidirect:
            model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim=1, dropout_rate=0.5, embeddings=embeddings, bidirectional=True).to(torch_device)
        else:
            model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim=1, dropout_rate=0.1, embeddings=embeddings, bidirectional=False).to(torch_device)
    elif args.neural_arch == "lstm":
        embedding_dim = 50
        hidden_dim = 128
        learning_rate = 0.0001
        batch_size = 16
        num_epochs = 100
        if args.rnn_bidirect:
            lambda_l1 =  0.000001 
            dropout=0.5
            weight_decay = 1e-3
            model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim=1, dropout_rate=0.5, embeddings=embeddings, bidirectional=True).to(torch_device).to(torch_device)
        else:
            lambda_l1 = 0.0000001
            dropout = 0.1
            weight_decay =  1e-4 
            model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim=1, dropout_rate=0.1, embeddings=embeddings, bidirectional=False).to(torch_device)

    # TODO: Read off the WiC dataset files from the `WiC_dataset' directory
    # (will be located in /homes/cs577/WiC_dataset/(train, dev, test))
    # and initialize PyTorch dataloader appropriately
    # Take a look at this page
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # and implement a PyTorch Dataset class for the WiC dataset in
    # utils.py



    

    # embedding_dim = 50
    # hidden_dim = 128 #128 for RNN
    # learning_rate = 0.001 #0.001 / for lstm
    # learning_rate = 0.00001 #||||| 0.00001 for dan ||| 0.00001 for rnn ||| 0.00001 for lstm
    # learning_rate = 0.0001
    # batch_size = 16 #32 for dan |||| 16 for rnn
    # num_epochs = 100
    # lambda_l1 =  0.00001 #0.00001 # 0.00001 for dan |||| 0.00001 for rnn ||| 0.0001 for lstm

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function)
    
    # Model, Loss, and Optimizer
    # model = SentenceMeaningComparator(vocab_size, embedding_dim, hidden_dim, output_dim=1)
    # model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim=1, dropout_rate=0.5, embeddings=embeddings, bidirectional=True).to(torch_device)
    # model = DAN(vocab_size, embedding_dim, hidden_dim, embeddings=embeddings, output_dim=1)
    # model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim=1, dropout_rate=0.5, embeddings=embeddings, bidirectional=True).to(torch_device)

    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #0.001
    # Training Loop
    # model.train()
    train_ac = []
    val_ac = []
    test_ac = []

    train_loss = []
    valid_loss = []
    test_loss = []

    for epoch in range(num_epochs):
        model.train()
        # print(dataloader[0])
        total_train_accuracy = 0
        total_train_loss = 0
        for tokens1, tokens2, word, pos, labels in dataloader:

            optimizer.zero_grad()
            outputs = model(tokens1, tokens2, word, pos)

            loss = loss_function(outputs.squeeze(), labels)
            # total_train_loss += loss.item() * tokens1.size(0)
            l1_penalty = torch.tensor(0.) #.to(inputs['s1'].device) # Ensuring the penalty is on the same device as model parameters
            for param in model.parameters():
                l1_penalty += torch.sum(torch.abs(param))
            # for param in model.parameters():
            #     l1_penalty += torch.norm(param, 1)**2
            loss += lambda_l1 * l1_penalty
            # print(outputs.squeeze(), labels)
            total_train_loss += loss.item() * tokens1.size(0)
            loss.backward()
            optimizer.step()
            accuracy = calculate_accuracy(outputs.squeeze(), labels)
            total_train_accuracy += accuracy.item()
            # total_train_loss += loss.item() * tokens1.size(0)
        avg_accuracy = total_train_accuracy / len(dataloader)
        train_ac.append(avg_accuracy)
        train_loss.append(total_train_loss/len(dataloader))
        print(f'Epoch {epoch+1}, train Loss: {loss.item()}, train accuracy: {avg_accuracy:.4f}')


        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            correct = 0
            total = 0
            total_accuracy = 0
            total_valid_loss = 0
            for tokens1, tokens2, word, pos, labels in validloader:

                # sentence1, sentence2, labels = sentence1.to(device), sentence2.to(device), labels.to(device)
                outputs = model(tokens1, tokens2, word, pos)
                # print(outputs.squeeze(), labels)
                loss = loss_function(outputs.squeeze(), labels)
                l1_penalty = torch.tensor(0.) #.to(inputs['s1'].device) # Ensuring the penalty is on the same device as model parameters
                for param in model.parameters():
                    l1_penalty += torch.sum(torch.abs(param))
                # for param in model.parameters():
                #     l1_penalty += torch.norm(param, 1)**2
                loss += lambda_l1 * l1_penalty
                total_valid_loss += loss.item() * tokens1.size(0)
                # predicted = torch.round(outputs)
                # predicted = torch.round(torch.sigmoid(outputs.squeeze()))  # Applying sigmoid to get [0,1] range and rounding off to get predictions
                accuracy = calculate_accuracy(outputs.squeeze(), labels)
                total_accuracy += accuracy.item() 
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
            # print(f'Validation Accuracy: {100 * correct / total}%')
            # avg_loss = total_loss / len(train_dataloader)
            avg_accuracy = total_accuracy / len(validloader)
            val_ac.append(avg_accuracy)
            valid_loss.append(total_valid_loss / len(validloader))
            print(f'Epoch {epoch+1} Validation Accuracy: {avg_accuracy:.4f}')
        
    
        with torch.no_grad():
            correct = 0
            total = 0
            total_accuracy = 0
            total_test_loss = 0
            for tokens1, tokens2, word, pos, labels in testloader:

                # sentence1, sentence2, labels = sentence1.to(device), sentence2.to(device), labels.to(device)
                outputs = model(tokens1, tokens2, word, pos)

                loss = loss_function(outputs.squeeze(), labels)
                l1_penalty = torch.tensor(0.) #.to(inputs['s1'].device) # Ensuring the penalty is on the same device as model parameters
                for param in model.parameters():
                    l1_penalty += torch.sum(torch.abs(param))
                # for param in model.parameters():
                #     l1_penalty += torch.norm(param, 1)**2
                loss += lambda_l1 * l1_penalty
                total_test_loss += loss.item() * tokens1.size(0)
                # predicted = torch.round(outputs)
                # predicted = torch.round(torch.sigmoid(outputs.squeeze()))  # Applying sigmoid to get [0,1] range and rounding off to get predictions
                accuracy = calculate_accuracy(outputs.squeeze(), labels)
                total_accuracy += accuracy.item()
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
            # print(f'Validation Accuracy: {100 * correct / total}%')
            # avg_loss = total_loss / len(train_dataloader)
            avg_accuracy = total_accuracy / len(testloader)
            test_ac.append(avg_accuracy)
            test_loss.append(total_test_loss / len(testloader))
            print(f'Epoch {epoch+1} test Accuracy: {avg_accuracy:.4f}')

    plot_loss_history(train_loss, valid_loss, test_loss, len(train_loss))
    plot_loss_history(train_ac, val_ac, test_ac, len(train_loss), save_path='acc_history.png', type="Accuracy")


    # TODO: Testing loop
    # Write predictions (F or T) for each test example into test.pred.txt
    # One line per each example, in the same order as test.data.txt.

    predictions = []
    with torch.no_grad():  # Disable gradient calculation for inference
        for tokens1, tokens2, word, pos, labels in testloader:
            outputs = model(tokens1, tokens2, word, pos)
            outputs = torch.round(outputs)
            predictions.extend(outputs.cpu().numpy()) 
    # Write predictions to a file
    with open('test.pred.txt', 'w') as f:
        for pred in predictions:
            # This writes each prediction on a new line. Adjust the format as needed.
            if pred == 1:
                f.write(f'T\n')
            else:
                f.write(f'F\n')
