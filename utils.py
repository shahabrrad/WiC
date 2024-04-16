from torch.utils.data import Dataset
from torch import tensor
import torch
import pandas as pd
from nltk.tokenize import PunktSentenceTokenizer,sent_tokenize, word_tokenize
from nltk.corpus import stopwords
# from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet as wn
import re
import numpy as np


# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# class WiCDataset(Dataset):



class WiCDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data = []
        self.labels = []
        lemmatizer = WordNetLemmatizer()
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                target_word, pos_tag, positions, sentence1, sentence2 = parts[0], parts[1], parts[2], parts[3], parts[4] #' '.join(parts[3:-1]), parts[-1]
                pos_tag_idx = 1 if pos_tag == 'V' else 0  # 1 for verb, 0 for noun
                sentence1 = re.sub(r"[^\w\s]", ' ', sentence1)
                sentence2 = re.sub(r"[^\w\s]", ' ', sentence2)
                sentence1_tokens = [lemmatizer.lemmatize(t) for t in sentence1.lower().split()]
                sentence2_tokens = [lemmatizer.lemmatize(t) for t in sentence2.lower().split()]
                target_word = lemmatizer.lemmatize(target_word)
                target_positions = [int(pos) for pos in positions.split('-')]
                self.data.append((sentence1_tokens, sentence2_tokens, pos_tag_idx, target_positions, target_word))

        with open(label_file, 'r', encoding='utf-8') as f:
            self.labels = [int(label.strip()=="T") for label in f]
            # print(self.labels[:10])

        assert len(self.data) == len(self.labels), "Number of data points and labels should be equal."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1_tokens, sentence2_tokens, pos_tag_idx, target_positions, target_word = self.data[idx]
        label = self.labels[idx]
        # print(sentence1_tokens)
        # print(sentence2_tokens)

        sentence1_ids = [self.vocab.stoi.get(token, self.vocab.stoi['<unk>']) for token in sentence1_tokens]
        sentence2_ids = [self.vocab.stoi.get(token, self.vocab.stoi['<unk>']) for token in sentence2_tokens]
        target_id = self.vocab.stoi.get(target_word, self.vocab.stoi['<unk>'])
        input_sequence = sentence1_ids + [self.vocab.stoi['<sep>']] + sentence2_ids + [self.vocab.stoi['<sep>']] + [target_id] 
        input_sequence_text = sentence1_tokens + ['<sep>'] + sentence2_tokens + ['<sep>'] + [target_word]

        # target_word = sentence1_tokens[target_positions[0]] if target_positions[0] < len(sentence1_tokens) else sentence2_tokens[target_positions[1] - len(sentence1_tokens) - 1]
        # wordnet_features = self.extract_wordnet_features(target_word, pos_tag_idx, sentence1_tokens, sentence2_tokens)

        # print(input_sequence_text)
        # print(sentence1_tokens, sentence2_tokens)
        return tensor(sentence1_ids), tensor(sentence2_ids), tensor([target_id]), tensor(pos_tag_idx, dtype=torch.float), tensor(label, dtype=torch.float)
        return tensor(sentence1_ids+ [self.vocab.stoi['<sep>']] + [target_id] ), tensor(sentence2_ids+ [self.vocab.stoi['<sep>']] + [target_id] ), tensor(label, dtype=torch.float)
        return tensor(input_sequence), tensor(label, dtype=torch.float)
        return tensor(input_sequence), tensor(pos_tag_idx), tensor(target_positions), tensor(label)

    def build_vocab(self, vocab, data_file):
        self.vocab = vocab #Vocab()

        self.vocab.add_token('<pad>')
        self.vocab.add_token('<unk>')
        self.vocab.add_token('<sep>')

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().lower().split('\t')
                _, _, _, sentence1, sentence2 = parts[0], parts[1], parts[2], parts[3], parts[4] #' '.join(parts[3:-1]), parts[-1]
                sentence1 = re.sub(r"[^\w\s]", ' ', sentence1)
                sentence2 = re.sub(r"[^\w\s]", ' ', sentence2)
                self.vocab.add_sentence(sentence1)
                self.vocab.add_sentence(sentence2)

        # self.vocab.add_token('<pad>')
        # self.vocab.add_token('<unk>')
        # self.vocab.add_token('<sep>')
        return self.vocab
    
    def vocab_size(self):
        return self.vocab.idx
    



class Vocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_token(self, token):
        lemmatizer = WordNetLemmatizer()
        if token not in self.word2idx:
            token = lemmatizer.lemmatize(token)
            self.word2idx[token] = self.idx
            self.idx2word[self.idx] = token
            self.idx += 1

    def add_sentence(self, sentence):
        for word in sentence.lower().split():
            self.addword(word)

    def addword(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    @property
    def stoi(self):
        return self.word2idx

    @property
    def itos(self):
        return self.idx2word




def get_wordnet_features(word):
    # Find synsets for the target word
    synsets = wn.synsets(word)
    # Example feature: the number of synsets
    num_synsets = len(synsets)
    # More sophisticated features can be calculated here
    
    # Example feature: Semantic similarity (this is a simplified example)
    if len(synsets) > 1:
        sim = synsets[0].wup_similarity(synsets[1])
    else:
        sim = 0
    
    # Return a feature vector for the word
    return [num_synsets, sim]

# Example usage
# word_features = get_wordnet_features("bank")


def collate_function(batch):
        # tokens, labels = zip(*batch)
        # tokens1, tokens2, labels = zip(*batch)
        tokens1, tokens2, word, pos, labels = zip(*batch)


        # tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
        tokens_padded1 = torch.nn.utils.rnn.pad_sequence(tokens1, batch_first=True, padding_value=0)
        tokens_padded2 = torch.nn.utils.rnn.pad_sequence(tokens2, batch_first=True, padding_value=0)

        labels = torch.stack(labels)
        word = torch.stack(word)
        pos = torch.stack(pos)

        # return tokens_padded, labels
        # return tokens_padded1, tokens_padded2, labels
        return tokens_padded1, tokens_padded2, word, pos, labels



def calculate_accuracy(y_pred, y_true):
    """Calculates accuracy of predictions."""
    predictions = torch.round(y_pred)
    correct = (predictions == y_true).float()  # convert into float for division
    accuracy = correct.sum() / len(correct)
    return accuracy

def initialize_gensim_embedding(model, vocab, vocab_size, embedding_dim=50):
    embeddings = np.zeros((vocab_size, embedding_dim))
    embeddings[2] = np.array([1 for i in range(50)])
    unknowns = []
    for i in range(3, vocab_size):
        try:
            embeddings[i] = model[vocab.itos[i]]
        except KeyError:
            unknowns.append(i)    
    unknown_emb = np.mean(embeddings,axis=0,keepdims=True)
    embeddings[1] = unknown_emb
    for j in unknowns:
        embeddings[j] = unknown_emb
    return(embeddings)



import matplotlib.pyplot as plt

def plot_loss_history(train_loss, val_loss, test_loss, epochs, save_path='loss_history.png', type="Loss"):

    if isinstance(epochs, int):
        epochs = range(1, epochs + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training '+type, color='blue', marker='o')
    plt.plot(epochs, val_loss, label='Validation '+type, color='green', marker='x')
    plt.plot(epochs, test_loss, label='Test '+type, color='red', marker='^')
    
    plt.title(type+' History over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(type)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(save_path)
    # plt.show()
