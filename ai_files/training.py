import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.language_utils import Utils
from ai_files.model import ChatDataset, NeuralNet
from sklearn.model_selection import train_test_split
from config import app_config
import csv
from datetime import datetime


class ChatbotTraining:
    def __init__(self):
        self.intents = self.load_intents()

    #Funkcja łądująca intencje którymi model będzie uczony
    def load_intents(self):
        with open(app_config.INTENTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    #Funckja przeprowadzająca trening modelu
    def train(self):
        words_list = []
        tags = []
        classes = []
        x_train = []
        y_train = []
        ignore_words = ['?', '.', ',', '!']
        utils = Utils()

        for intent in self.intents['intents']:
            tag = intent['tag']
            tags.append(tag)
            for pattern in intent['patterns']:
                word_tokenize = utils.tokenize_word(pattern)
                words_list.extend(word_tokenize)
                classes.append((word_tokenize, tag))

        ignore_words = ['?', '.', ',', '!']
        words_list = [utils.stem_word(
            word) for word in words_list if word not in ignore_words]
        words_list = sorted(set(words_list))
        tags = sorted(set(tags))

        for (pattern_sentence, tag) in classes:
            bow = utils.bag_of_words(pattern_sentence, words_list)
            x_train.append(bow)
            label = tags.index(tag)
            y_train.append(label)

        #Preparacja zbiorów do treningu modelu, walidacyjny i treningowy
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        #Hiperparametry
        epochs = 1000
        batch_size = 64
        learning_rate = 0.0005
        input_size = len(x_train[0])
        hidden_size = 28
        output_size = len(tags)

        #Zbiór danych
        dataset = ChatDataset(x_train=x_train, y_train=y_train)
        training = DataLoader(dataset=dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = NeuralNet(input_size=input_size, hidden_size=hidden_size,
                          num_classes=output_size).to(device)

        #Kryteria oceniania
        criterion = nn.CrossEntropyLoss()
        #Optymalizator
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        #Kod odpowiedzialny za uczenie modelu oraz dodatkowo zapisujący wyniki do pliku CSV
        date = datetime.now()
        date = date.strftime("%Y-%m-%d_%H-%M")
        with open(f'wyniki_{date}.csv', mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])
            for epoch in range(epochs):
                for (words, labels) in training:
                    words = words.to(device)
                    labels = labels.to(dtype=torch.long).to(device)

                    # loss - strata
                    outputs = model(words)
                    loss = criterion(outputs, labels)

                    # accuracy - skutecznosc
                    _, predicts = torch.max(outputs.data, 1)
                    correct = (predicts == labels).sum().item()
                    accuracy = correct / len(labels)

                    # cross-validation
                    val_outputs = model(x_val)
                    val_loss = criterion(val_outputs, y_val)
                    _, val_predicts = torch.max(val_outputs.data, 1)
                    val_correct = (val_predicts == y_val).sum().item()
                    val_accuracy = val_correct / len(y_val)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if (epoch+1) % 100 == 0:
                    print(f"""Epoch [{epoch+1}/{epochs}],
                            Loss: {loss.item():.4f},
                            Accuracy: {accuracy:.4f},
                            val_loss : {val_loss.item():.4f}, 
                            val_accuracy: {val_accuracy:.4f}""")
                    writer.writerow([epoch+1, f'{loss.item():.4f}', f'{accuracy:.4f}', f'{val_loss.item():.4f}', f'{val_accuracy:.4f}'])

            print(f"""final loss: {loss.item():.4f}, 
                    Accuracy: {accuracy:.4f}, 
                    val_loss : {val_loss.item():.4f}, 
                    val_accuracy: {val_accuracy:.4f}""")
            writer.writerow(['final', f'{loss.item():.4f}', f'{accuracy:.4f}', f'{val_loss.item():.4f}', f'{val_accuracy:.4f}'])

        #Końcowy model
        data = {
            "model_state": model.state_dict(),
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "words_list": words_list,
            "tags": tags
        }

        #Zapisanie modelu do pliku i zakończenie treningu
        torch.save(data, app_config.MODEL_DATA_PATH)
        print(f'training complete. file saved to {app_config.MODEL_DATA_PATH}')
