from modules.cboard import CBoard
from tqdm import tqdm
from game import Player, Game
import json


def main():
    #コマンドライン引数でリストめんどいからコード直打ち
    board = CBoard()
    models = ["models/gen28_epoch80.onnx:100","edax:05","random"]
    players = [Player(board,model) for model in models]
    epsilon = 0.9
    plays = 10
    
    for play in tqdm(range(plays)):
        for i in range(1,len(players)):
            for j in range(len(players)):
                player1 = players[j]
                player2 = players[(j+i)%len(players)]
                
                game = Game(player1,player2,epsilon=epsilon)
                game()
        dic = {player.name:f"{player.wins}勝{player.loses}敗, elo:{round(player.elo)}" for player in players}
        print(dic)
        with open("league_results.json", "w") as f:
            json.dump(dic,f,indent = 2, ensure_ascii=False)
            
if __name__ == "__main__":
    main()
