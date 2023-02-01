#適当・・・

from modules.mcts import MCTS
from modules.cboard import CBoard
import random
from tqdm import tqdm
from modules.utils import BOARD
import onnxruntime
import json

models = ["models/gen28_c128b10_aug.onnx","models/gen28_c128b10.onnx","models/gen28_c64b4_vit.onnx","models/gen28_epoch80.onnx","models/gen20_epoch60.onnx","models/gen10_epoch20.onnx","models/gen1_epoch82.onnx","random"]
members = list(range(len(models)))
elos = [1500]*len(members)
wins = [0]*len(members)
loses = [0]*len(members)
epsilon = 0.9
plays = [100,100,100,100,300,300,300,500,500,1000]
k = 32


for play in tqdm(plays):
    for i in range(1,len(members)):
        for j in range(len(members)):
            player1 = j
            player2 = (j+i)%len(members)
            board = CBoard()
            if models[player1] == "random":
                session1 = "random"
            else:
                session1 = onnxruntime.InferenceSession(models[player1],providers=['CUDAExecutionProvider','CPUExecutionProvider'])
            if models[player2] == "random":
                session2 = "random"
            else:
                session2 = onnxruntime.InferenceSession(models[player2],providers=['CUDAExecutionProvider','CPUExecutionProvider'])
            
            mcts1 = MCTS(board,session1,play,batch_size=8)
            mcts2 = MCTS(board,session2,play,batch_size=8)
            while True:
                if board.turn == 1:
                    if random.random() <= epsilon:
                        move = mcts1.move(0)
                    else:
                        move = mcts1.move(1)
                    mcts2.move_enemy(move)
                else:
                    if random.random() <= epsilon:
                        move = mcts2.move(0)
                    else:
                        move = mcts2.move(1)
                    mcts1.move_enemy(move)
                result = board.move(move)

                if result == 1:
                    num = board.board[BOARD].sum()
                    if num > 0:
                        W = 1/(10**((elos[player1]-elos[player2])/400)+1)
                        elos[player1] = elos[player1] + k*W
                        elos[player2] = elos[player2] - k*W
                        wins[player1] += 1
                        loses[player2] += 1
                    else:
                        W = 1/(10**((elos[player2]-elos[player1])/400)+1)
                        elos[player1] = elos[player1] - k*W
                        elos[player2] = elos[player2] + k*W
                        wins[player2] += 1
                        loses[player1] += 1
                    break
    results = {models[member]:f"elo:{elos[member]}, {wins[member]}勝{loses[member]}敗" for member in members}
    with open("league_results.json", "w") as f:
        json.dump(results,f)
    print(results)
