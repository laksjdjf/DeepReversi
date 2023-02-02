import argparse
from tqdm import tqdm
from game import Player, Game
from modules.cboard import CBoard

###コマンドライン引数#########################################################################
parser = argparse.ArgumentParser(description='2つのモデルを比較する')
parser.add_argument('model1', type=str, help='プレイヤー１')
parser.add_argument('model2', type=str, help='プレイヤー２')
parser.add_argument('num_plays', type=int, help='試合数、前後半やるため偶数推奨！！')
parser.add_argument('--epsilon', type=float, default = 0.8, help='次善手を打つ確率 mctsのみ')
############################################################################################

def main(args):
    #プログレスバー
    board = CBoard()
    player1 = Player(board,args.model1)
    player2 = Player(board,args.model2)
    black_win = 0
    progress_bar = tqdm(range(args.num_plays), desc="Total Steps", leave=False)
    
    for i in range(args.num_plays):
        game = Game(player1,player2,epsilon=args.epsilon)
        black_win += game(reverse = False if i%2 == 0 else True)
        
        #プログレスバー更新
        logs={"model1_win":player1.wins,"model2_win":player2.wins,"model1_time":player1.think_time,"model2_time":player2.think_time}
        progress_bar.update(1)
        progress_bar.set_postfix(logs)
        
    print("")
    print(f"player1目線で{player1.wins}勝{player2.wins}敗！！！")
    print(f"先攻：{black_win}勝,後攻：{args.num_plays - black_win}勝")
    print(f"player1思考時間：{player1.think_time}")
    print(f"player2思考時間：{player2.think_time}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
