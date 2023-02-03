#Resnet Vit mlp-mixerに対応

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from modules.dataset import BoardDataset
import argparse
from tqdm import tqdm
import time
import os
import wandb

###コマンドライン引数#########################################################################
parser = argparse.ArgumentParser(description='Reversiの訓練コード')
parser.add_argument('--model', type=str, default=None, help='学習済みモデルパス（pth）')
parser.add_argument('--dataset', type=str, required=True, help='データセットモデルパス')
parser.add_argument('--output', type=str, required=True, help='モデル出力先（pth）')
parser.add_argument('--epoch', type=int, default=20, help='エポック')
parser.add_argument('--batch_size', type=int, default=4096, help='バッチサイズ')
parser.add_argument('--channels', type=int, default=64, help='チャンネル:埋め込み次元')
parser.add_argument('--blocks', type=int, default=4, help='ブロック')
parser.add_argument('--fcl', type=int, default=128, help='全結合隠れ層：MLP隠れ層')
parser.add_argument('--network', type=str, required=True, help='モデル選択：「resnet,vit,mixer」')
parser.add_argument('--lr', type=float, required=True, help='学習率')
############################################################################################

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("cudaが使えません")
        return
    
    if args.network == "resnet":
        from modules.network import ResNet
        model = ResNet(channels=args.channels, blocks=args.blocks, fcl=args.fcl)
        
    elif args.network == "vit":
        from modules.network import Vit
        model = Vit(emb_dim=args.channels, num_blocks=args.blocks, hidden_dim=args.fcl)
        
    elif args.network == "mixer":
        from modules.network import MLPMixer
        model = MLPMixer(emb_dim=args.channels, num_blocks=args.blocks, channels_mlp_dim=args.fcl)
    
    if args.model is not None:
        model.load_state_dict(torch.load(args.model))
    model.to(device)
    
    #default
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    loss_cel = torch.nn.CrossEntropyLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    
    #amp
    scaler = torch.cuda.amp.GradScaler() 
    
    #dataset
    dataset = BoardDataset(args.dataset)
    train_size = int(len(dataset)*0.9)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,len(dataset)-train_size],)
    
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True,num_workers=8)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False,num_workers=8)
    
    policy_weight = 1
    
    total_steps = len(train_dataloader) * args.epoch
    
    run = wandb.init(project="DeepReversi", name=args.output,dir=os.path.join(args.output,'wandb'))
    
    #プログレスバー
    progress_bar = tqdm(range(total_steps), desc="Total Steps", leave=False)
    loss_ema = None #訓練ロスの指数平均
    global_step = 0

    for epoch in range(args.epoch):
        model.train()
        for batch_idx, (features,moves,results,evals) in enumerate(train_dataloader):
            
            b_start = time.perf_counter()
            features = features.to(device)
            moves = moves.to(device)
            results = results.to(device).unsqueeze(1)
            if evals[0] != "None":
                evals = evals.to(device).unsqueeze(1)
            
            with torch.cuda.amp.autocast(): 
                # 損失計算
                output1,output2 = model(features)
                
                #交差エントロピー損失
                loss_p = torch.sum(- moves * F.log_softmax(output1,dim=1),1)
                
                #actor critic
                if evals[0] != "None":
                    z = results - evals + 0.5
                    loss_p = (loss_p * z).mean()
                else:
                    loss_p = loss_p.mean()
                
                #エントロピー正則化
                loss_p += (F.softmax(output1, dim=1) * F.log_softmax(output1, dim=1)).sum(dim=1).mean()
                
                loss_v = loss_bce(output2,results)
                
                #boot strap
                if evals[0] != "None":
                    loss_e = loss_bce(output2,evals)
                    loss = loss_p + loss_v * 0.7 + loss_e * 0.3
                else:
                    loss = loss_p + loss_v
            
            #勾配リセット
            optimizer.zero_grad()
            
            # 誤差逆伝播
            scaler.scale(loss).backward() 

            # 勾配をアンスケールしてパラメータの更新
            scaler.step(optimizer) 

            # スケーラーの更新
            scaler.update() 
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = loss_ema * 0.9 + loss.item() * 0.1
            
            #時間計測
            b_end = time.perf_counter()
            time_per_steps = b_end - b_start
            samples_per_time = args.batch_size / time_per_steps
            
            #プログレスバー更新
            logs={"loss":loss_ema,"samples_per_second":samples_per_time}
            progress_bar.update(1)
            progress_bar.set_postfix(logs)
            global_step += 1
            run.log(logs, step=global_step)
            
        model.eval()
        test_loss = 0
        policy_correct = 0
        value_correct = 0
        with torch.no_grad():
            for features,moves,results,_ in test_dataloader:
                features = features.to(device)
                moves = moves.to(device)
                results = results.to(device).unsqueeze(1)
                output1,output2 = model(features)
                policy_correct += (output1.argmax(axis=1) == moves.argmax(axis=1)).type(torch.float).sum().item()
                value_correct += (output2>=0).eq(results>=0.5).type(torch.float).sum().item()
            policy_correct /= len(test_dataloader.dataset)
            value_correct /= len(test_dataloader.dataset)
            print(f'epoch: {epoch + 1}, policy accuracy: {policy_correct}, value accuracy: {value_correct}')
        run.log({"policy accuracy":policy_correct, "value_accuracy": value_correct}, step=global_step)
        torch.save(model.state_dict(), args.output)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
