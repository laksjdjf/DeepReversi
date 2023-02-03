#ResNet:https://github.com/TadaoYamaoka/python-dlshogi2/blob/main/pydlshogi2/network/policy_value_resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

#ResNetBlock:dlshogiとほぼ同等の構造。
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return torch.relu(out + x)


#ResNetBlockを並べたネットワーク、channelとblock数を設定できる。
class ResNet(nn.Module):
    def __init__(self, channels=64, blocks=4, fcl=128):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)

        self.blocks = torch.nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])

        self.p1 = nn.Conv2d(channels,1,kernel_size=1,stride=1,padding=0)

        self.v1 = nn.Linear(channels*64, fcl)
        self.v2 = nn.Linear(fcl, 1)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = torch.relu(x)
        
        x = self.blocks(x)

        policy = self.p1(x)
        policy = torch.flatten(policy,1)

        value = torch.flatten(x, 1)
        value = self.v1(value)
        value = torch.relu(value)
        value = self.v2(value)

        return policy,value
    


#Vit:https://github.com/ghmagazine/vit_book/blob/main/ch2/vit.py
class VitInputLayer(nn.Module): 
    def __init__(self, in_channels:int=2, emb_dim:int=64, num_patch_row:int=8, image_size:int=8):
        """ 
        引数:
            in_channels: 入力画像のチャンネル数
            emb_dim: 埋め込み後のベクトルの長さ
            num_patch_row: 高さ方向のパッチの数。例は2x2であるため、2をデフォルト値とした 
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定
        """
        super(VitInputLayer, self).__init__() 
        self.in_channels=in_channels 
        self.emb_dim = emb_dim 
        self.num_patch_row = num_patch_row 
        self.image_size = image_size
        
        # パッチの数
        ## 例: 入力画像を2x2のパッチに分ける場合、num_patchは4 
        ## オセロの場合:パッチは1マスごとに分ける・・・パッチ数は64
        self.num_patch = self.num_patch_row**2

        # パッチの大きさ
        ## 例: 入力画像の1辺の大きさが32の場合、patch_sizeは16 
        ## オセロの場合：パッチの大きさは1
        self.patch_size = int(self.image_size // self.num_patch_row)

        # 入力画像のパッチへの分割 & パッチの埋め込みを一気に行う層 
        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.emb_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

        # クラストークン 
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, emb_dim) 
        )

        # 位置埋め込み
        ## クラストークンが先頭に結合されているため、
        ## 長さemb_dimの位置埋め込みベクトルを(パッチ数+1)個用意 
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch+1, emb_dim) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        引数:
            x: 入力画像。形状は、(B, C, H, W)。[式(1)]
                B: バッチサイズ、C:チャンネル数、H:高さ、W:幅

        返り値:
            z_0: ViTへの入力。形状は、(B, N, D)。
                B:バッチサイズ、N:トークン数、D:埋め込みベクトルの長さ
        """
        # パッチの埋め込み & flatten [式(3)]
        ## パッチの埋め込み (B, C, H, W) -> (B, D, H/P, W/P) 
        ## ここで、Pはパッチ1辺の大きさ
        z_0 = self.patch_emb_layer(x)

        ## パッチのflatten (B, D, H/P, W/P) -> (B, D, Np) 
        ## ここで、Npはパッチの数(=H*W/Pˆ2)
        z_0 = z_0.flatten(2)

        ## 軸の入れ替え (B, D, Np) -> (B, Np, D) 
        z_0 = z_0.transpose(1, 2)

        # パッチの埋め込みの先頭にクラストークンを結合 [式(4)] 
        ## (B, Np, D) -> (B, N, D)
        ## N = (Np + 1)であることに留意
        ## また、cls_tokenの形状は(1,1,D)であるため、
        ## repeatメソッドによって(B,1,D)に変換してからパッチの埋め込みとの結合を行う 
        z_0 = torch.cat(
            [self.cls_token.repeat(repeats=(x.size(0),1,1)), z_0], dim=1)

        # 位置埋め込みの加算 [式(5)] 
        ## (B, N, D) -> (B, N, D) 
        z_0 = z_0 + self.pos_emb
        return z_0

class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, emb_dim:int=64, head:int=8, dropout:float=0.):
        """ 
        引数:
            emb_dim: 埋め込み後のベクトルの長さ 
            head: ヘッドの数
            dropout: ドロップアウト率
        """
        super(MultiHeadSelfAttention, self).__init__() 
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim ** 0.5 # D_hの二乗根。qk^Tを割るための係数

        # 入力をq,k,vに埋め込むための線形層。 [式(6)] 
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        # 式(7)にはないが、実装ではドロップアウト層も用いる 
        self.attn_drop = nn.Dropout(dropout)

        # MHSAの結果を出力に埋め込むための線形層。[式(10)]
        ## 式(10)にはないが、実装ではドロップアウト層も用いる 
        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout) 
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """ 
        引数:
            z: MHSAへの入力。形状は、(B, N, D)。
                B: バッチサイズ、N:トークンの数、D:ベクトルの長さ

        返り値:
            out: MHSAの出力。形状は、(B, N, D)。[式(10)]
                B:バッチサイズ、N:トークンの数、D:埋め込みベクトルの長さ
        """

        batch_size, num_patch, _ = z.size()

        # 埋め込み [式(6)]
        ## (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # q,k,vをヘッドに分ける [式(10)]
        ## まずベクトルをヘッドの個数(h)に分ける
        ## (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        ## Self-Attentionができるように、
        ## (バッチサイズ、ヘッド、トークン数、パッチのベクトル)の形に変更する 
        ## (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # 内積 [式(7)]
        ## (B, h, N, D//h) -> (B, h, D//h, N)
        k_T = k.transpose(2, 3)
        ## (B, h, N, D//h) x (B, h, D//h, N) -> (B, h, N, N) 
        dots = (q @ k_T) / self.sqrt_dh
        ## 列方向にソフトマックス関数
        attn = F.softmax(dots, dim=-1)
        ## ドロップアウト
        attn = self.attn_drop(attn)
        # 加重和 [式(8)]
        ## (B, h, N, N) x (B, h, N, D//h) -> (B, h, N, D//h) 
        out = attn @ v
        ## (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)
        ## (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # 出力層 [式(10)]
        ## (B, N, D) -> (B, N, D) 
        out = self.w_o(out) 
        return out

class VitEncoderBlock(nn.Module): 
    def __init__(self, emb_dim:int=64, head:int=8, hidden_dim:int=64*4, dropout: float=0.):
        """
        引数:
            emb_dim: 埋め込み後のベクトルの長さ
            head: ヘッドの数
            hidden_dim: Encoder BlockのMLPにおける中間層のベクトルの長さ 
                        原論文に従ってemb_dimの4倍をデフォルト値としている
            dropout: ドロップアウト率
        """
        super(VitEncoderBlock, self).__init__()
        # 1つ目のLayer Normalization [2-5-2項]
        self.ln1 = nn.LayerNorm(emb_dim)
        # MHSA [2-4-7項]
        self.msa = MultiHeadSelfAttention(
        emb_dim=emb_dim, head=head,
        dropout = dropout,
        )
        # 2つ目のLayer Normalization [2-5-2項] 
        self.ln2 = nn.LayerNorm(emb_dim)
        # MLP [2-5-3項]
        self.mlp = nn.Sequential( 
            nn.Linear(emb_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, emb_dim), 
            nn.Dropout(dropout)
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """ 
        引数:
            z: Encoder Blockへの入力。形状は、(B, N, D)
                B: バッチサイズ、N:トークンの数、D:ベクトルの長さ

        返り値:
            out: Encoder Blockへの出力。形状は、(B, N, D)。[式(10)]
                B:バッチサイズ、N:トークンの数、D:埋め込みベクトルの長さ 
        """
        # Encoder Blockの前半部分 [式(12)] 
        out = self.msa(self.ln1(z)) + z
        # Encoder Blockの後半部分 [式(13)] 
        out = self.mlp(self.ln2(out)) + out 
        return out

class Vit(nn.Module): 
    def __init__(self, in_channels:int=2, emb_dim:int=64, num_patch_row:int=8, image_size:int=8, num_blocks:int=4, head:int=8, hidden_dim:int=64*4, dropout:float=0.):
        """ 
        引数:
            in_channels: 入力画像のチャンネル数
            num_classes: 画像分類のクラス数
            emb_dim: 埋め込み後のベクトルの長さ
            num_patch_row: 1辺のパッチの数
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定 
            num_blocks: Encoder Blockの数
            head: ヘッドの数
            hidden_dim: Encoder BlockのMLPにおける中間層のベクトルの長さ 
            dropout: ドロップアウト率
        """
        super(Vit, self).__init__()
        # Input Layer [2-3節] 
        self.input_layer = VitInputLayer(
            in_channels, 
            emb_dim, 
            num_patch_row, 
            image_size)

        # Encoder。Encoder Blockの多段。[2-5節] 
        self.encoder = nn.Sequential(*[
            VitEncoderBlock(
                emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout = dropout
            )
            for _ in range(num_blocks)])

        # MLP Head [2-6-1項] 
        self.mlp_head_value = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 1)
        )
        
        # Policy head patchのトークンを利用する
        self.mlp_head_policy = nn.Sequential(
            nn.LayerNorm([8,8]),
            nn.Conv2d(emb_dim, 1, kernel_size=1, stride=1, padding=0)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x: ViTへの入力画像。形状は、(B, C, H, W)
                B: バッチサイズ、C:チャンネル数、H:高さ、W:幅

        返り値:
            out: ViTの出力。形状は、(B, M)。[式(10)]
                B:バッチサイズ、M:クラス数 
        """
        # Input Layer [式(14)]
        ## (B, C, H, W) -> (B, N, D)
        ## N: トークン数(=パッチの数+1), D: ベクトルの長さ 
        out = self.input_layer(x)
        
        # Encoder [式(15)、式(16)]
        ## (B, N, D) -> (B, N, D)
        out = self.encoder(out)

        # クラストークンのみ抜き出す
        ## (B, N, D) -> (B, D)
        cls_token = out[:,0]
        
        # パッチトークン
        ## (B, N, D) -> (B, D, 8, 8)
        patch_token = out[:,1:].permute(0,2,1).reshape(out.shape[0],out.shape[2],8,8)
        
        #value
        ## (B, D) -> (B, M)
        value = self.mlp_head_value(cls_token)
        
        #policy
        ## (B, D, 8, 8) -> (B, 64)
        policy = self.mlp_head_policy(patch_token).flatten(1)
        
        
        return policy, value

#mlp mixer ref:https://github.com/google-research/vision_transformer/blob/linen/vit_jax/models_mixer.py

# emb_dim -> mlp_dim -> emb_dim
class MLPBlock(nn.Module):
    def __init__(self, emb_dim:int, mlp_dim:int, dropout:float):
        super(MLPBlock, self).__init__()
        self.mlp_block = nn.Sequential( 
            nn.Linear(emb_dim, mlp_dim), 
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(mlp_dim, emb_dim), 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.mlp_block(x)
        return out
        
class MixerBlock(nn.Module):
    def __init__(self, emb_dim:int, channels_mlp_dim:int, tokens_mlp_dim:int, dropout:float):
        super(MixerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.tokens_mlp = MLPBlock(64, tokens_mlp_dim, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.channels_mlp = MLPBlock(emb_dim, channels_mlp_dim, dropout)

    def forward(self, x):
        y = self.ln1(x)
        
        #(B, 64, D)->(B, D, 64)
        y = y.transpose(1, 2)
        y = self.tokens_mlp(y)
        y = y.transpose(1, 2)
        y = x + y
        
        y = self.ln2(y)
        return x + self.channels_mlp(y)
    
class MLPMixer(nn.Module):
    def __init__(self, emb_dim:int, num_blocks:int, channels_mlp_dim:int, tokens_mlp_dim:int=256, dropout:float=0.):
        super(MLPMixer, self).__init__()
        #オセロ固定
        self.proj_in = nn.Conv2d(2, emb_dim, kernel_size = 1, stride = 1, padding = 0)
        self.mixer_blocks = nn.Sequential(*[
            MixerBlock(
                emb_dim=emb_dim,
                tokens_mlp_dim=tokens_mlp_dim,
                channels_mlp_dim=channels_mlp_dim,
                dropout = dropout
        ) for _ in range(num_blocks)])

        self.ln  = nn.LayerNorm(emb_dim)
        
        self.p1 = nn.Conv2d(emb_dim,1,kernel_size=1,stride=1,padding=0)

        self.v1 = nn.Linear(emb_dim*64, channels_mlp_dim) #なんとなくchannles_mlp_dimと同じにしてしまう。
        self.v2 = nn.Linear(channels_mlp_dim, 1)
    
    def forward(self, x):
        
        #(B,2,8,8) -> (B,D,8,8) -> (B,D,64) -> (B,64,D)
        x = self.proj_in(x).flatten(2).transpose(1,2)
        x = self.mixer_blocks(x)
        x = self.ln(x)
        
        policy = x.permute(0,2,1).reshape(x.shape[0],x.shape[2],8,8)
        policy = self.p1(policy).flatten(1)
        
        value = self.v1(x.flatten(1))
        value = self.v2(torch.relu(value))
        
        return policy, value
