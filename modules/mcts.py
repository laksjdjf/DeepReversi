import numpy as np
from modules.utils import sigmoid,softmax,BOARD
from modules.cboard import CBoard


class Node():
    def __init__(self):
        self.move_count = 0
        self.sum_value = 0.0
        self.child_move = None
        self.child_move_count = None
        self.child_sum_value = None
        self.child_node = None
        self.policy = None
        self.value = None

    def creat_child_node(self, index):
        self.child_node[index] = Node()

    def expand_node(self,board):
        child_num = len(self.child_move)
        self.child_move_count = np.zeros(child_num,dtype=np.int32)
        self.child_sum_value = np.zeros(child_num,dtype=np.float32)
        self.child_node = [None]*child_num

    def update(self,policy,value):
        self.policy = [policy[move%10 + move//10*8 -9].item() for move in self.child_move]
        self.value = value[0]

    def leaf(self,board):
        self.child_move = board.legal_moves
        if len(self.child_move)==0:
            self.value = sigmoid(board.turn * board.board[BOARD].sum()/10)
            return True
        return False

    def eval(self,board,model):
        self.child_move = board.legal_moves
        if len(self.child_move)==0:
            self.value = sigmoid(board.turn * board.board[BOARD].sum()/10)
        else:
            feature = np.array([board.feature()],dtype=np.float32)
            policy, value = infer(feature,model)
            self.value = value[0,0]
            self.policy = [policy[0][move%10 + move//10*8 -9].item() for move in self.child_move]

    def choice(self,c_puct):
        win_rate = np.divide(self.child_sum_value,self.child_move_count,out=np.zeros(len(self.child_move), np.float32)+0.5,where=self.child_move_count != 0)
        index = np.argmax(win_rate + (self.policy * (np.sqrt(np.float32(self.move_count)) / (self.child_move_count + 1))) * c_puct)
        if self.child_node[index] == None:
            self.creat_child_node(index)
        return index

def infer(feature,session):
    if session == "random":
        return np.random.rand(1,64) , np.random.rand(1,1)
    io_binding = session.io_binding()
    io_binding.bind_cpu_input('input', feature)
    io_binding.bind_output('policy')
    io_binding.bind_output('value')
    session.run_with_iobinding(io_binding)
    policy,value = io_binding.copy_outputs_to_cpu()
    return softmax(policy),sigmoid(value)

class MCTS():
    def __init__(self,board,session,halt=100,c_puct=1,threshold=3,batch_size=8):
        self.original_node = Node()
        self.original_board = board.copy()
        self.current_node = self.original_node
        self.current_board = board.copy()
        self.route = []
        self.indexes = []
        self.turns = [self.current_board.turn]
        self.moves = []
        self.threshold = threshold
        self.playout_number = 0
        self.session = session
        self.halt = halt
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.current_batch = 0
        self.batch_route = []
        self.batch_feature = []

    def search(self):
        while self.playout_number <= self.halt:
            if self.current_node.move_count == 0:
                self.route.append(self.current_node)
                if self.current_node.leaf(self.current_board):
                    self.playout()
                    continue
                else:
                    self.virtul_loss()
                    continue
            elif self.current_node.value == None:
                self.route.append(self.current_node)
                self.virtul_loss(False)
                continue
            elif self.current_node.move_count < self.threshold:
                self.route.append(self.current_node)
                self.playout()
                continue
            else:
                self.route.append(self.current_node)
                if self.current_node.child_node == None:
                    self.current_node.expand_node(self.current_board)
                if len(self.current_node.child_move) == 0:
                    self.playout()
                    continue
                index = self.current_node.choice(self.c_puct)
                self.moves.append(self.current_node.child_move[index])
                self.current_board.move(self.current_node.child_move[index])
                self.current_node = self.current_node.child_node[index]
                self.indexes.append(index)
                self.turns.append(self.current_board.turn)

        sorted_index = np.argsort(self.original_node.child_move_count)[::-1]
        winrate = self.original_node.child_sum_value[sorted_index[0]] / self.original_node.child_node[sorted_index[0]].move_count
        return np.array(self.original_node.child_move)[sorted_index] ,winrate
    
    def move(self,num=0,re_eval=False):
        moves,eval = self.search()
        move = moves[min(num,len(moves)-1)]
        self.original_board.move(move)
        self.current_board = self.original_board.copy()
        self.playout_number = 0
        
        index = self.original_node.child_move.index(move)
        if self.original_node.child_node[index] == None:
            self.original_node = Node()
        else:
            self.original_node = self.original_node.child_node[index]
        self.current_node = self.original_node
        if re_eval:
            return move,eval
        else:
            return move

    def move_enemy(self,move):
        self.original_board.move(move)
        self.current_board = self.original_board.copy()
        self.playout_number = 0

        if self.original_node.child_node == None:
            self.original_node = Node()
        else:
            index = self.original_node.child_move.index(move)
            if self.original_node.child_node[index] == None:
                self.original_node = Node()
            else:
                self.original_node = self.original_node.child_node[index]
        self.current_node = self.original_node

        

    def playout(self):
        self.playout_number += 1
        results = None
        while len(self.route) > 0:
            node = self.route.pop()
            if results == None:
                results = node.value
                node.sum_value += results
            else:
                index = self.indexes.pop()
                node.sum_value += results
                node.child_move_count[index] += 1
                node.child_sum_value[index] += results
            turn = self.turns.pop()
            if len(self.turns) >= 1 and turn != self.turns[-1]:
                results = 1 - results
            node.move_count += 1
        self.current_node = self.original_node
        self.current_board = self.original_board.copy()
        self.route = []
        self.indexes = []
        self.moves = []
        self.turns = [self.current_board.turn]

    def virtul_loss(self,init=True):
        self.current_batch += 1
        self.batch_route += [(self.route.copy(),self.indexes.copy(),self.turns.copy())]
        if init:
            self.batch_feature += [self.current_board.feature()]
        results = None
        while len(self.route) > 0:
            node = self.route.pop()
            if results == None:
                results = 0
            else:
                index = self.indexes.pop()
                node.child_move_count[index] += 1
            turn = self.turns.pop()
            node.move_count += 1
        self.current_node = self.original_node
        self.current_board = self.original_board.copy()
        self.route = []
        self.indexes = []
        self.moves = []
        self.turns = [self.current_board.turn]
        if self.current_batch == self.batch_size:
            self.backup()
            self.current_batch = 0

    def backup(self):
        self.playout_number += self.current_batch
        policies, values = infer(np.array(self.batch_feature,dtype=np.float32),self.session)
        i = 0
        for route,indexes,turns in self.batch_route:
            results = None
            while len(route) > 0:
                node = route.pop()
                if results == None:
                    if node.value==None:
                        results = values[i,0]
                        node.update(policies[i],values[i])
                        i+=1
                    else:
                        results = node.value
                    node.sum_value += results
                    
                else:
                    index = indexes.pop()
                    node.sum_value += results
                    node.child_sum_value[index] += results
                turn = turns.pop()
                if len(turns) >= 1 and turn != turns[-1]:
                    results = 1 - results
        self.batch_route = []
        self.batch_feature = []
            
