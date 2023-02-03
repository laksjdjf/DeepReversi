import sys
import os
sys.path.append(os.getcwd())
import torch
import onnxruntime
import argparse

###コマンドライン引数#########################################################################
parser = argparse.ArgumentParser(description='onnx変換コード')
parser.add_argument('--model', type=str, required=True, help='学習済みモデルパス（pth）')
parser.add_argument('--channels', type=int, default=128, help='チャンネル:埋め込み次元')
parser.add_argument('--blocks', type=int, default=10, help='ブロック')
parser.add_argument('--fcl', type=int, default=256, help='全結合隠れ層：MLP隠れ層')
parser.add_argument('--network', type=str, required=True, help='モデル選択：「resnet,vit,mixer」')
############################################################################################


def main(args):
    if args.network == "resnet":
        from modules.network import ResNet
        model = ResNet(channels=args.channels, blocks=args.blocks, fcl=args.fcl)
        
    elif args.network == "vit":
        from modules.network import Vit
        model = Vit(emb_dim=args.channels, num_blocks=args.blocks, hidden_dim=args.fcl)
        
    elif args.network == "mixer":
        from modules.network import MLPMixer
        model = MLPMixer(emb_dim=args.channels, num_blocks=args.blocks, channels_mlp_dim=args.fcl)

    model.to("cuda")
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # Input to the model
    x = torch.randn(1, 2, 8, 8, requires_grad=True).to("cuda")
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # 対象モデル
                      x,                         # 入力サンプル
                      os.path.splitext(args.model)[0] + ".onnx",   # 保存先
                      export_params=True,        # モデルファイルに訓練した重みを保存するかどうか
                      opset_version=10,          # ONNXのバージョン
                      do_constant_folding=True,  # constant folding for optimizationを実施するかどうか
                      input_names = ['input'],   # モデルへの入力変数名
                      output_names = ['policy','value'], # モデルの出力変数名
                      dynamic_axes={
                      'input' : {0 : 'batch_size'},
                      'policy' : {0 : 'batch_size'},
                      'value' : {0 : 'batch_size'},}
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
