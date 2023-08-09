import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import cairosvg
import PIL
from PIL import Image
import chess.engine
import io
from skimage import exposure
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, LeakyReLU, ReLU, BatchNormalization,
                                     Reshape, Flatten, Input, Activation, Conv2D,
                                     Conv2DTranspose, Lambda, Cropping2D)
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from skimage.transform import resize
from keras.datasets import mnist
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import os
import math
import warnings
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


stockfish_path = 'stockfish/stockfish/stockfish.exe'


def evaluate_position(board, engine, depth=20):
    analysis = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = analysis["score"].white()
    evaluation = 0
    if score.is_mate():
        num_moves_to_mate = score.mate()
        if num_moves_to_mate == 1:
            evaluation = 9999999
        elif num_moves_to_mate == -1:
            evauluation = -9999999
        elif num_moves_to_mate > 0:
            evaluation = 1000000
        else:
            evaluation = -1000000
    else:
        evaluation = score.score()
    return evaluation

def count_winning_endings(board, depth, stockfish, perspective):
    biggest_depth = -1
    def explore_moves(board, current_depth, pov):
        nonlocal biggest_depth
        if current_depth == depth:
            biggest_depth = depth+2
            return 1

        if(perspective == pov):
            analysis = stockfish.analyse(board, chess.engine.Limit(depth=20),multipv=3)
        else:
            analysis = stockfish.analyse(board, chess.engine.Limit(depth=20),multipv = 1)
        centipawns = evaluate_position(board, stockfish)
        if (centipawns == 9999999 and perspective == 'white') or (centipawns == -9999999 and perspective == 'black'):
            if biggest_depth == -1:
                biggest_depth = current_depth+2
            else:
                biggest_depth = min(biggest_depth, current_depth)
            return 1 if pov != perspective else 0
        elif (centipawns > 75 and perspective == 'white') or (centipawns < -75 and perspective == 'black'):
            top_moves = []
            for i in range(len(analysis)):
                top_moves.append(analysis[i]['pv'][0])
            total_winning_endings = 0
            for move in top_moves:
                new_board = board.copy()
                if new_board.is_legal(move):
                    new_board.push(move)
                    if (evaluate_position(new_board, stockfish) > 75 and perspective == 'white') or \
                       (evaluate_position(new_board, stockfish) < -75 and perspective == 'black') or \
                       new_board.is_checkmate():
                        total_winning_endings += explore_moves(new_board, current_depth + 1, 'white' if pov == 'black' else 'black')

            return total_winning_endings
        else:
            return 0
    if board.is_valid():
        total_winning_endings = explore_moves(board, 0, perspective)
        if(biggest_depth <= 0):
            return -1,-1
        return total_winning_endings, biggest_depth
    else:
        return -2,-1

df = pd.read_csv("lichess_db_puzzle.csv")

#decide threshold
threshold = 10000


#metric for now
df['Puzzle Score'] = 100*df['Popularity']+0.005*df['NbPlays']
df = df.sort_values(by ='Puzzle Score', ascending = False)

#number of puzzles analyzed for now
opti_df = df.head(threshold)

def editFENAndMoves(row):
    move = row['Moves'].split()[0]
    row['Moves'] = row['Moves'].split(maxsplit=1)[1:]
    board = chess.Board(row['FEN'])
    board.push(chess.Move.from_uci(move))
    row['FEN'] = board.fen()
    return row

opti_df = opti_df.apply(editFENAndMoves, axis = 1)


engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)


stockfish_path = 'stockfish/stockfish/stockfish.exe'  # Replace this with the path to your Stockfish executable
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

# Apply the evaluate_position() function to the 'FEN' column and store results in a new column 'Evaluation'
opti_df['Evaluation'] = opti_df['FEN'].apply(lambda fen: evaluate_position(chess.Board(fen), engine))

# Close the engine after all evaluations are done
engine.quit()

opti_df.to_csv('opti_df.csv', index = False)

opti_df = pd.read_csv('opti_df.csv')

opti_df = opti_df[opti_df['Evaluation'] > 0]

print(opti_df.shape)

opti_df = opti_df[opti_df['Moves'].apply(len) <= 29]
opti_df = opti_df[opti_df['Themes'].str.contains('middlegame', case=False, na=False)]
print(opti_df.shape)

opti_df.to_csv('opti_winning_count.csv', index=False)

def fen_to_one_hot_array(fen):
    piece_map = {
        'P': 1, 'p': 2,
        'R': 3, 'r': 4,
        'N': 5, 'n': 6,
        'B': 7, 'b': 8,
        'K': 9, 'k': 10,
        'Q': 11, 'q': 12,
        '.': 0,
    }

    fen_parts = fen.split(' ')[0]  the FEN
    fen_rows = fen_parts.split('/')

    chess_board = [[0 for _ in range(8)] for _ in range(8)]

    for row_index, row in enumerate(fen_rows):
        col_index = 0
        for char in row:
            if char.isdigit():
                col_index += int(char)
            else:
                chess_board[row_index][col_index] = piece_map.get(char, 0)
                col_index += 1

    max_value = 12
    num_pieces = len(piece_map)  
    chess_board_one_hot = np.zeros((8, 8, num_pieces), dtype=int)

    for i in range(8):
        for j in range(8):
            chess_board_one_hot[i, j, chess_board[i][j]] = 1

    return chess_board_one_hot

for i in range(len(opti_df)):
    array = fen_to_one_hot_array(opti_df['FEN'].iloc[i])
    np.save(f"Positions8/{i}.npy",array)

stockfish_path = 'stockfish/stockfish/stockfish.exe'
pixelArr = []
Endgame = 2660
Middlegame = 2363
for i in range(Middlegame):
    array = np.load(f"Positions8/{i}.npy")
    pixelArr.append(array)
print(pixelArr[0])
pixelArray = np.array(pixelArr)
print(pixelArray.shape)

def convertPNG (fen, step, batch, epoch):
    board = chess.Board(fen)
    boardsvg = chess.svg.board(coordinates=True, board = board, size=256, colors={"square light": "#f8dcb4", "square dark": "#b88c64"})
    img = cairosvg.svg2png(bytestring=boardsvg)
    image = Image.open(io.BytesIO(img))
    bw_image = image.convert("L")
    bw_image.save(f"GeneratedImages8/{epoch}-{batch}-{step}.png")
    fen_path = f"GeneratedFEN8/{epoch}-{batch}-{step}.txt"
    with open(fen_path, "w") as f:
        f.write(fen)



def continuous_array_to_one_hot(continuous_array):
    num_classes = continuous_array.shape[-1]
    max_indices = tf.argmax(continuous_array, axis=-1)
    one_hot_array = tf.one_hot(max_indices, num_classes)
    return one_hot_array
def one_hot_array_to_fen(chess_board_one_hot):
    piece_map = {
        1: 'P', 2: 'p', 
        3: 'R', 4: 'r', 
        5: 'N', 6: 'n', 
        7: 'B', 8: 'b', 
        9: 'K', 10: 'k',
        11: 'Q', 12: 'q',
        0: '.', 
    }

    chess_board_one_hot = chess_board_one_hot.reshape(8, 8, 13)

    chess_board = continuous_array_to_one_hot(chess_board_one_hot)
    chess_board = np.argmax(chess_board, axis=-1)
    fen_parts = []
    for row in chess_board:
        fen_row = []
        empty_count = 0
        for val in row:
            if val == 0:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row.append(str(empty_count))
                    empty_count = 0
                fen_row.append(piece_map[val])

        if empty_count > 0:
            fen_row.append(str(empty_count))

        fen_parts.append(''.join(fen_row))
    fen_string = '/'.join(fen_parts)
    return fen_string




def count_empty_squares(fen):
    fen_parts = fen.split(' ')
    board_part = fen_parts[0]
    empty_squares = 0
    for char in board_part:
        if char.isdigit():
            empty_squares += int(char)
        elif char == '/':
            continue

    return empty_squares

def count_distinct_pieces(fen):
    piece_symbols = "KQRBNPkqrbnp"
    pieces = set()
    for char in fen:
        if char in piece_symbols:
            pieces.add(char)
    return len(pieces)





class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  
        )

        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                layer.weight.data = layer.weight.data.float()
                layer.bias.data = layer.bias.data.float()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  
        )

    def encode(self, x):
        h = self.encoder(x)
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = x.float()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



def loss_function(recon_x, x, mu, logvar, batch_idx, epoch, global_step):
    batch_size = recon_x.size(0)
    
    winning_penalty_weight = 10
    depth_penalty_weight =  0.04
    empty_penalty_weight =  0.1
    distinct_penalty_weight= 0.1
    valid_penalty_weight = 100
    valid2_penalty_weight = 8
    BCE_weight =  0.01
    KLD_weight = 0.3
    anneal_rate = 0.001
    winning_penalties = []
    depth_penalties = []
    empty_penalties = []
    valid_penalties = []
    distinct_penalties = []
    
    for i in range(batch_size):
        fen_string_pred = one_hot_array_to_fen(recon_x[i].detach().numpy())
        board_pred = chess.Board(fen_string_pred)
        empty_squares_pred = count_empty_squares(fen_string_pred)
        total_winning_endings_pred, biggest_depth_pred = count_winning_endings(board_pred, 6, engine, 'white')
        distinct_pred = count_distinct_pieces(fen_string_pred)
        #CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE
        winning_penalty = np.mean((total_winning_endings_pred - 1) ** 2) * winning_penalty_weight
        depth_penalty = np.mean((biggest_depth_pred - 8) ** 2) * depth_penalty_weight
        empty_penalty = np.mean((empty_squares_pred - 0) ** 2) * empty_penalty_weight
        distinct_penalty = np.mean((empty_squares_pred - 12) ** 2) * distinct_penalty_weight
        winning_penalties.append(winning_penalty.item())
        depth_penalties.append(depth_penalty.item())
        empty_penalties.append(empty_penalty.item())
        distinct_penalties.append(distinct_penalty.item())
        valid_penalty = 0
        if (total_winning_endings_pred == -2):
            #CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE
            valid_penalty = 1 * valid_penalty_weight
            print(empty_squares_pred, end = " ")
        elif (total_winning_endings_pred == -1):
            #CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE
            valid_penalty = 1 * valid2_penalty_weight
        else:
            convertPNG(fen_string_pred, i, batch_idx, epoch)
            convertPNG(one_hot_array_to_fen(x[i]),str(i)+'r',batch_idx,epoch)
        valid_penalties.append(valid_penalty)
    print()
    mean_winning_penalty = sum(winning_penalties) / batch_size
    mean_depth_penalty = sum(depth_penalties) / batch_size
    mean_empty_penalty = sum(empty_penalties) / batch_size
    mean_valid_penalty = sum(valid_penalties) / batch_size
    mean_distinct_penalty = sum(distinct_penalties) / batch_size

    #CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE CHANGE
    BCE = nn.BCELoss(reduction='sum')(recon_x, x) * BCE_weight
    annealed_kld_weight = min(1.0, anneal_rate * global_step)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * annealed_kld_weight * KLD_weight
    
    print("Penalties:")
    print(f"Winning Penalty: {mean_winning_penalty}")
    print(f"Depth Penalty: {mean_depth_penalty}")
    print(f"Empty Penalty: {mean_empty_penalty}")
    print(f"Valid Penalty: {mean_valid_penalty}")
    print(f"Distinct Penalty: {mean_distinct_penalty}")
    print(f"BCE Loss: {BCE.item()}")
    print(f"KLD Loss: {KLD.item()}")
    total_loss = BCE + KLD + mean_winning_penalty + mean_depth_penalty + mean_empty_penalty + mean_valid_penalty + mean_distinct_penalty
    print(f"Total Loss: {total_loss.item()}")
    return total_loss
def train_vae(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    global_step = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x = data.float()  
        x = x.view(-1, input_dim)
        x = Variable(x)
        if torch.cuda.is_available():
            x = x.cuda()
    
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar, batch_idx, epoch, global_step)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        global_step += 1
    
    print('Epoch: {} Loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def plot_latent_space_pca(model, dataloader, input_dim, latent_dim=100, target_dim=2):
    model.eval()
    with torch.no_grad():
        latent_vectors = []
        labels = []
        for batch_idx, data in enumerate(dataloader):
            x = data.view(-1, input_dim)
            x = x.float() 
            if torch.cuda.is_available():
                x = x.cuda()
            mu, _ = model.encode(x)

            if mu.size(0) == 0:
                continue

            latent_vectors.append(mu.cpu().numpy())
            labels.append(batch_idx) 
        latent_vectors = np.concatenate(latent_vectors, axis=0)
    pca = PCA(n_components=target_dim)
    latent_vectors_reduced = pca.fit_transform(latent_vectors)
    cmap = get_cmap('viridis')
    colors = np.linspace(0, 1, len(latent_vectors_reduced))

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_vectors_reduced[:, 0], latent_vectors_reduced[:, 1], c=colors, cmap='viridis')
    plt.colorbar()
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('Latent Space Visualization (PCA)')
    plt.show()



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
data_array = pixelArray
custom_dataset = CustomDataset(data_array)



#CHANGE THIS
batch_size = 256
train_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
latent_dim = 100 
hidden_dim = 450
input_dim = 8 * 8 * 13 

model = VAE(input_dim, hidden_dim, latent_dim)
if torch.cuda.is_available():
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=3e-3)

num_epochs = 1000
for epoch in range(1, num_epochs + 1):
    train_vae(model, train_loader, optimizer, epoch)
    plot_latent_space_pca(model, train_loader, input_dim, latent_dim=100, target_dim=2)
    model_filename = f'GeneratedModels8/vae_model_epoch_{epoch}.pth'
    torch.save(model.state_dict(), model_filename)
