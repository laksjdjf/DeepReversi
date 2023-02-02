#random, onnxモデル, edax対応のプレイヤー、試合
import re
import onnxruntime
from modules.cboard import CBoard
from modules.mcts import MCTS
import subprocess
import numpy as np
from modules.utils import BOARD,BLACK,WHITE,pos2num,num2pos,display
import time
import random
import argparse

INIT_ELO = 1500
ELO_K = 16


#Playerクラス
class Player:
    #コンストラクタ、文字列でプレイヤーを区別する
    def __init__(self,board:CBoard, model:str):
        self.init_board = board.copy()
        self.elo = INIT_ELO
        self.wins = 0
        self.loses = 0
        self.think_time = 0
        self.model = model
        self.name = model.replace("models/","")
        
        if "edax" in model:
            self.player = Edax(board, model)
        elif model == "random":
            self.player = RandomPlayer(board, model)
        elif "models" in model:
            self.player = MCTSPlayer(board, model)
        elif model == "human":
            self.player = Human(board, model)
    
    def reset(self,board:CBoard):
        if "models" in self.model:
            self.player.mcts.reset(board)
        return 
   
    #最善手または次善手と思考時間を返す：randomで制御
    def move(self,board:CBoard, epsilon:float = 0.7) -> (int,float):
        return self.player.move(board, epsilon)
    
    def update(self,move:int):
        self.player.update(move)
        return 
            
class Edax:
    def __init__(self,board:CBoard, model:int):
        self.level = int(model[-2:]) #"edax_04"といった形を想定する
        
    def move(self,board:CBoard, epsilon:float = 0.7) -> (int,float):
        self.board2text(board)
        cmd = ["./edax-4.4","-l",str(self.level),"-solve" ,"board.txt"]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        move = out.split('\n')[2][57:].split()[0].lower()
        move = pos2num(move)
        think_time = out.split('\n')[4]
        think_time = float(re.findall(r"\d+\.\d+",think_time)[0])
        return move, think_time #丸め誤差が特に低い深さだと高くなる
    
    def update(self,move:int):
        return 
    
    def board2text(self,board):
        text = ""
        for point in board.board[BOARD]:
            if point == BLACK:
                text += "*"
            elif point == WHITE:
                text += "O"
            else:
                text += "-"
        if board.turn == BLACK:
            text += "*"
        else:
            text += "O"
        with open("board.txt" , "w") as f:
            f.write(text)
        return
    
class MCTSPlayer:
    def __init__(self, board:CBoard, model:str):
        config = model.split(":") #:で設定を分ける
        self.session = onnxruntime.InferenceSession(config[0],providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.playout = int(config[1])
        self.mcts = MCTS(board,self.session,self.playout,batch_size=8,perfect=5)
        
    def move(self,board:CBoard, epsilon:float = 0.7) -> (int,float):
        start_time = time.perf_counter()
        if random.random() <= epsilon:
            move = self.mcts.move(0)
        else:
            move = self.mcts.move(1)
        end_time = time.perf_counter()
        return move, end_time - start_time
    
    def update(self,move:int):
        self.mcts.move_enemy(move)
        return
    
class RandomPlayer:
    def __init__(self, board:CBoard, model:str):
        pass
        
    def move(self,board:CBoard, epsilon:float = 0.7) -> (int,float):
        move = random.choice(list(board.legal_moves))
        return move, 0.0 #timeはあまり意味がないので0
    
    def update(self,move:int):
        return
    
class Human:
    def __init__(self, board:CBoard, model:str):
        pass
        
    def move(self,board:CBoard, epsilon:float = 0.7) -> (int,float):
        moves = [num2pos(move) for move in list(board.legal_moves)]
        print(f"合法手は{moves}です。")
        move = None
        while move is None:
            move = input("指し手を入力してください")
            if move not in moves:
                print("合法手ではないようです。")
                move = None
        return pos2num(move), 0.0 #timeはあまり意味がないので0
    
    def update(self,move:int):
        return
    
#試合クラス
class Game:
    def __init__(self, black:Player, white:Player, board:CBoard=None,epsilon:float = 0.7, display:bool = False):
        self.init_board = board if board is not None else CBoard()
        self.board = self.init_board.copy()
        self.black = black
        self.white = white
        self.epsilon = epsilon
        self.display = display
        
    def __call__(self, reverse:bool = False):
        while True:
            if self.board.turn == (WHITE if reverse else BLACK):
                move, think_time = self.black.move(self.board, self.epsilon)
                self.black.think_time += think_time
                self.white.update(move)
            else: #この条件式一つにまとめられないんか？
                move, think_time = self.white.move(self.board, self.epsilon)
                self.white.think_time += think_time
                self.black.update(move)
            
            result = self.board.move(move)
            
            if self.display:
                display(self.board)
            
            if result == 1:
                num = self.board.board[BOARD].sum()
                self.reset()
                if (num <= 0) if reverse else (num > 0):
                    self.black.wins += 1
                    self.white.loses += 1
                    W = 1/(10**((self.black.elo-self.white.elo)/400)+1)
                    self.black.elo += ELO_K*W
                    self.white.elo -= ELO_K*W
                    return 0 if reverse else 1
                else: #この条件式(ry 引き分け・・ひとまず白勝利とする
                    self.white.wins += 1
                    self.black.loses += 1
                    W = 1/(10**((self.white.elo-self.black.elo)/400)+1)
                    self.white.elo += ELO_K*W
                    self.black.elo -= ELO_K*W
                    return 1 if reverse else 0
        
    def reset(self):
        self.board = self.init_board.copy()
        self.black.reset(self.board)
        self.white.reset(self.board)
if __name__ == "__main__":
    ###コマンドライン引数#########################################################################
    parser = argparse.ArgumentParser(description='2つのモデルを比較する')
    parser.add_argument('model1', type=str, help='プレイヤー1')
    parser.add_argument('model2', type=str, help='プレイヤー2')
    parser.add_argument('--verbose', '-v', action="store_true", help='盤面を表示する')
    parser.add_argument('--epsilon', '-e', type=float, default = 0.8, help='次善手を打つ確率 mctsのみ')
    ##############################################################################################
    args = parser.parse_args()
    
    board = CBoard()
    player1 = Player(board,args.model1)
    player2 = Player(board,args.model2)
    game = Game(player1,player2,display=args.verbose,epsilon=args.epsilon)
    game()
