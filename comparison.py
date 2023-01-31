from modules.cboard import CBoard
from modules.mcts import MCTS
from modules.utils import BOARD
import time
import onnxruntime
import argparse
from tqdm import tqdm
import random


###コマンドライン引数#########################################################################
parser = argparse.ArgumentParser(description='StableDiffusionの訓練コード')
parser.add_argument('model1', type=str, help='学習済みモデルパス1（onnx）')
parser.add_argument('model2', type=str, help='学習済みモデルパス2（onnx）')
parser.add_argument('--playout', type=int, default=100, help='プレイアウト数')
parser.add_argument('--batch_size', type=int, default=4, help='バッチサイズ')
parser.add_argument('--num_plays', type=int, default=100, help='試合数')
############################################################################################

def main(args):
    epsilon = 0.7
    model1_time = 0
    model2_time = 0
    model1_win = 0
    model2_win = 0

    black_win = 0
    white_win = 0
    
    session1 = onnxruntime.InferenceSession(args.model1,providers=['CPUExecutionProvider'])
    session2 = onnxruntime.InferenceSession(args.model2,providers=['CPUExecutionProvider'])
    
    #プログレスバー
    progress_bar = tqdm(range(args.num_plays), desc="Total Steps", leave=False)
    for i in range(args.num_plays):
        board = CBoard()
        mcts1 = MCTS(board,session1,args.playout,batch_size=args.batch_size,perfect=5)
        mcts2 = MCTS(board,session2,args.playout,batch_size=args.batch_size,perfect=5)
        
        while True:
            if board.turn == [1,-1][i%2]:
                start_time = time.perf_counter()
                if random.random() <= epsilon:
                    move = mcts1.move(0)
                else:
                    move = mcts1.move(1)
                end_time = time.perf_counter()
                model1_time += end_time - start_time    
                mcts2.move_enemy(move)
            else:
                start_time = time.perf_counter()
                if random.random() <= epsilon:
                    move = mcts2.move(0)
                else:
                    move = mcts2.move(1)
                mcts1.move_enemy(move)
                end_time = time.perf_counter()
                model2_time += end_time - start_time 
            result = board.move(move)

            if result == 1:
                num = board.board[BOARD].sum()
                if [1,-1][i%2] * num > 0:
                    model1_win += 1
                elif num != 0:
                    model2_win += 1
                if num > 0:
                    black_win += 1
                elif num < 0:
                      white_win += 1
                break
        #プログレスバー更新
        logs={"model1_win":model1_win,"model2_win":model2_win}
        progress_bar.update(1)
        progress_bar.set_postfix(logs)
        
    print("")
    print(model1_win,model2_win)
    print(black_win,white_win)
    print(model1_time)
    print(model2_time)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
