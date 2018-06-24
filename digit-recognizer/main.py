import argparse
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

from models import ConvNet_Models
from models import RecurNet_Models
from models import Feed_Forward

from train import Trainer

parser = argparse.ArgumentParser()

# Positional Arguments
parser.add_argument("DATASET_PATH", help="Path to Dataset File (.CSV)", type=str)
parser.add_argument("MODEL", help="Neural Network Model to Train", type=str)
models = ['feed_forward', 'multi_column_cnn', 'simple_rnn', 'bidirectional_rnn', 'vote_multi_column_cnn']

# Optional Arguments
parser.add_argument("--save_model", help="Where to save trained model? Default='model/model.ckpt'", type=str, default='model/model.ckpt')
parser.add_argument("--epochs", help="How many Epochs to train for? Default=10", type=int, default=10)
parser.add_argument("--batch_size", help="What will be the batch Size? Default=128", type=int, default=128)
parser.add_argument("--learning_rate", help="What will be the Learning rate? Default = 0.001", type=float, default=0.001)
parser.add_argument("--feed_forward_architecture", help="If training Feed Forward Network, what will be the Architecture? Specify Number of neurons in each hidden layer EX: [500,500,500]", default="[500,500,500]")

args = parser.parse_args()

if not args.DATASET_PATH or not (args.DATASET_PATH.endswith(".csv")):
    print("No Dataset To Train or Invalid Dataset File!")
    print("'python3 main.py -h' for help")
    sys.exit()
if not args.MODEL or not(args.MODEL in models):
    print("Invalid Model to Train")
    print("USAGE:", models)
    sys.exit()

dataset_path = args.DATASET_PATH
model_to_train = args.MODEL

path_to_save_model = args.save_model
total_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
architecture = args.feed_forward_architecture

def read_data(dataset_path):
    data = pd.read_csv(dataset_path)
    data = data.values
    values = list(data[:,0])

    labels = []
    for val in values:
        one_hot = np.zeros(10)
        one_hot[val] += 1
        labels.append(np.asarray(one_hot, np.float32))

    images = []
    for val in data:
        images.append(np.asarray(val[1:], np.float32))

    return images, labels

images, labels = read_data(dataset_path)

if model_to_train in ["multi_column_cnn", "vote_multi_column_cnn"]:
    img_w = 28
    img_h = 28
    no_classes = 10
    channels = 1

    x = tf.placeholder("float", [None, img_w, img_h, channels])
    y = tf.placeholder("float", [None, no_classes])

    if model_to_train == "multi_column_cnn":
        model_to_train = ConvNet_Models.Multi_Column_DeepNet()
    else:
        model_to_train = ConvNet_Models.Vote_Multi_Column_DeepNet()

    size = [-1, img_h, img_w, channels]





elif model_to_train in ['simple_rnn', 'bidirectional_rnn']:
    hidden_features = 512
    no_classes = 10
    timesteps = 28
    input_size = 28

    x = tf.placeholder("float", [None, timesteps, input_size])
    y = tf.placeholder("float", [None, no_classes])

    if model_to_train == "simple_rnn":
        model_to_train = RecurNet_Models.Simple_RNN(hidden_features=hidden_features, no_classes=no_classes,
                                                    timesteps=timesteps)
    else:
        model_to_train = RecurNet_Models.Bidirectional_RNN(hidden_features=hidden_features, no_classes=no_classes,
                                                    timesteps=timesteps)

    size = [-1, timesteps, input_size]

elif model_to_train == "feed_forward":
    architecture = architecture.replace("[", '')
    architecture = architecture.replace("]", '')
    architecture = architecture.split(',')

    input_size = 784
    no_classes = 10

    architecture = [int(x) for x in architecture]

    architecture = [784] + architecture + [10]

    size = [-1, input_size]

    x = tf.placeholder("float", [None, input_size])
    y = tf.placeholder("float", [None, no_classes])

    model_to_train = Feed_Forward.Feed_Forward(architecture)


images = np.reshape(images, size)

trainer = Trainer(x, y)

trainer.train_network(x, dataset=[images, labels], model=model_to_train.model,
            learning_rate=learning_rate, save_model_path=path_to_save_model,
            total_epochs=total_epochs, batch_size=batch_size)






















#
