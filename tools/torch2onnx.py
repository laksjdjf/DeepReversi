#ひどい実装だがひとまず置いておく

import torch

import sys
import os
sys.path.append(os.getcwd())

from modules.network import PolicyValueNetwork

gen = "gen28_c128b10"
model = PolicyValueNetwork(128,10,256)
model.to("cuda")
model_path = 'models/' + gen + '.pt'
model.load_state_dict(torch.load(model_path))
model.eval()

# Input to the model
x = torch.randn(1, 2, 8, 8, requires_grad=True).to("cuda")
torch_out = model(x)

# Export the model
torch.onnx.export(model,               # 対象モデル
                  x,                         # 入力サンプル
                  "models/" + gen + ".onnx",   # 保存先
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
