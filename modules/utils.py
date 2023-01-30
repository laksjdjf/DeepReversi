import numpy as np

BLACK = 1
WHITE = -1

BOARD_SIZE = 10
BOARD = [x+y*BOARD_SIZE for y in range(1,9) for x in range(1,9) ]

POS = ["a","b","c","d","e","f","g","h"]
POS2NUM = {key:i+1 for i,key in enumerate(POS)}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
def logit(x):
    return np.log(x/(1-x))

def num2pos(num):
    return POS[num%10-1] + str(num//10)

def pos2num(pos):
    return int(pos[1])*10 + int(POS2NUM[pos[0]])

def count(board):
    black = (board.board[BOARD]==BLACK).sum()
    white = (board.board[BOARD]==WHITE).sum()
    return black,white

def display(board):
    print("X a b c d e f g h")
    for i in range(1,9):
        for j in range(0,9):
            if j == 0:
                print(str(i)+"\033[42m",end="")
            elif board.board[j+i*BOARD_SIZE]==BLACK:
                print("\033[30m●\033[37m",end="")
            elif board.board[j+i*BOARD_SIZE]==WHITE:
                print("\033[37m●\033[37m",end="")
            else:
                print("□",end="")
        print("\033[40m")
    print("手番:{}".format("黒" if board.turn==1 else "白"))
    print("差：",int(board.board[BOARD].sum()))
    print(list(map(num2pos,board.legal_moves)))
    #print(list(board.legal_moves))