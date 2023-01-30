#include <string>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <pybind11/eigen.h>

using namespace std;

const int BLACK = 1;
const int BLANK = 0;
const int WHITE = -1;
const int WALL = 2;

const int BOARD_SIZE = 10;

const int RIGHT = 1;
const int LEFT = -1;
const int UP = -BOARD_SIZE;
const int DOWN = BOARD_SIZE;

const int RIGHT_UP = RIGHT+UP;
const int RIGHT_DOWN = RIGHT + DOWN;
const int LEFT_UP = LEFT+UP;
const int LEFT_DOWN = LEFT+DOWN;

const int DIRECTIONS[8] = {RIGHT,LEFT,UP,DOWN,RIGHT_UP,RIGHT_DOWN,LEFT_UP,LEFT_DOWN};

using Eigen::MatrixXd;
using Eigen::VectorXi;

class Board{
    public:
        VectorXi board = VectorXi::Zero(100);
        int turn;
        std::vector<int> legal_moves;

        Board(){
            turn = BLACK;
            for(int i=0;i<100;i++){
                if(i%10==0 || i%10 == 9 || i/10==0 || i/10 == 9){
                    board(i) = WALL;
                }
                board(44) = WHITE;
                board(55) = WHITE;
                board(45) = BLACK;
                board(54) = BLACK;
            }
            update_legal_moves();
        }


        int move(int z){
            int p,q;
            
            //空白でないならエラーをだす
            if(board(z) != BLANK){
                throw std::logic_error("エラー");
            }

            //上下左右斜めそれぞれに対して、石ひっくり返す動きをする。
            for(int i=0;i<8;i++){
                p = z + DIRECTIONS[i];
                q = 1;

                //隣が敵の石だったら、隣に移動するのという動作繰り返す。
                while(board(p) == -turn){
                    p += DIRECTIONS[i];
                    q += 1;
                }

                //自分の石を見つけたら、
                if(board(p) == turn){
                    //そこまでにあった石を全部自分の石にする。
                    //j=0からスタートなので、指した場所そのものも自分の石になる。同じこと8回やってるけどまあいいや。
                    for(int j=0;j<q;j++){
                        board(z+DIRECTIONS[i]*j) = turn;
                    }
                }
                //コードはないですが、自分の石に会う前に、空白や壁にたどり着いた場合、何も起こりません。
            }
            turn = -turn;
            update_legal_moves();
            if(legal_moves.size()==0){
                turn = -turn; //敵に合法手がない場合、彩度自分のターンになる。
                update_legal_moves();
                if(legal_moves.size()==0){
                    return 1; //両方なかったらゲーム終了。
                }
            }
            return 0;
        }

        void update_legal_moves(){

            std::vector<int> moves;
            int p,py,d;

            //すごい単純な全探索アルゴリズムです
            for(int y=1;y<9;y++){
                py = y * 10;
                for(int x=1;x<9;x++){

                    //空白なら非合法手
                    if(board(py+x)!=BLANK){
                        continue;
                    }
                    //全ての方角で
                    for(int i=0;i<8;i++){
                        d = DIRECTIONS[i];
                        p = py + x + d;
                        //隣が敵の石じゃなかったら即アウト！
                        if(board(py+x+d) != -turn){
                            continue;
                        }
                        //隣が敵の石じゃなくなるまで一直線に探索。
                        while(board(p) == -turn){
                            p += d;
                        }
                        //自分の石にたどり着いたら合法、空白や壁だったら違法
                        if(board(p) == turn){
                            moves.push_back(py+x);
                            break;
                        }
                    }
                }
            }
            legal_moves = moves;
        }

        //8×8ボードでの計算用
        int output_move(int move){
            return move%10 + move/10*8 -9;
        }

        //numpy用、機械学習のためにいらない壁を壊す
        vector<MatrixXd> feature(){
            MatrixXd feature1 = MatrixXd::Zero(8,8);
            MatrixXd feature2 = MatrixXd::Zero(8,8);
            vector<MatrixXd> features;

            features.push_back(feature1);
            features.push_back(feature2);

            int t = turn==1 ? 0:1;
            for(int y=0;y<8;y++){
                for(int x=0;x<8;x++){
                    if(board(y*10+x+11)==1)features[t](y,x)=1;
                    if(board(y*10+x+11)==-1)features[1-t](y,x)=1;
                }
            }
            return features;
        }

        //python用
        Board copy(){
            Board board;
            board = *this;
            return board;
        }

        //確認用
        void display(){
            std::string stone[4] = {"○","□","●","■"};
            for(int i=0;i<100;i++){
                if(i%10==0)std::cout << "\n";
                std::cout << stone[board(i)+1];
            }       
        }

};


namespace py = pybind11;
PYBIND11_MODULE(cboard,m) {
    py::class_<Board>(m, "CBoard")
        .def(py::init<>())
        .def("move", &Board::move)
        .def("feature", &Board::feature)
        .def("output_move", &Board::output_move)
        .def("copy", &Board::copy)
        .def_readwrite("legal_moves", &Board::legal_moves)
        .def_readwrite("board", &Board::board)
        .def_readwrite("turn", &Board::turn);
}

