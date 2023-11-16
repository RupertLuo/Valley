import torch
from torch import nn
class ResBlock(nn.Module):
    def __init__(self, feature_num, input_dim, dropout, alpha=1):
        super(ResBlock, self).__init__()
        self.linear_w = nn.Parameter(torch.randn(feature_num,input_dim,input_dim))
        self.linear_b = nn.Parameter(torch.randn(feature_num,1,input_dim))
        nn.init.kaiming_uniform_(self.linear_w, mode='fan_in', nonlinearity='leaky_relu')
        self.leaky_relu = nn.LeakyReLU(inplace=True)
#         self.dropout = nn.Dropout(p=dropout)
        self.alpha = alpha

    def forward(self, x):
        h = torch.matmul(x,self.linear_w)+self.linear_b
        h = h + self.alpha * x
        h = self.leaky_relu(h)
#         h = self.dropout(h)
        return h

class Video_Gandalf_Model_V1(torch.nn.Module):
    def __init__(self, features: int,
            llm_dim = 4096,
            num_class = 4,
            bucket_dim = 128,
            drop_prob = 0.3,
            binary = True,
            bucket_num=8,
            layer_conf=[64, 64, 64],
            alpha=1,
            output_size=128,
            use_fc=False,
            dropout=0.0,
            add_block=False,
            num_classes = 26
            ):
        super().__init__()
        self.feature_dim  = features
        feature_num = features
        self._out_dim = num_class
        self._drop_prob = drop_prob
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        #self.Dropout = nn.Dropout(p=dropout)
        self.linear1_w = nn.Parameter(torch.randn(feature_num,3,layer_conf[0]))
        self.linear1_b = nn.Parameter(torch.randn(feature_num,1,layer_conf[0]))
        self.layer = nn.Sequential(*[ResBlock(feature_num, layer_len, dropout=0, alpha=alpha) for layer_len in layer_conf]) if add_block else nn.Identity()
        self.linear2_w = nn.Parameter(torch.randn(feature_num,layer_conf[-1],bucket_num))
        self.linear2_b = nn.Parameter(torch.randn(feature_num,1,bucket_num))

        self.emb = nn.Parameter(torch.randn(feature_num,bucket_num,bucket_dim))
        self._tau_module = nn.Parameter(torch.ones([feature_num,1,bucket_num]))
        self.Softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(feature_num*bucket_dim,output_size,bias=True) if use_fc else nn.Identity()

        nn.init.kaiming_uniform_(self.linear1_w, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.linear2_w, mode='fan_in', nonlinearity='leaky_relu')

        #self.auto_dis = AutoDiscretizationLayer(features, bucket_size, embedding_dim, temperature, skip_alpha)
        # asr_length = 128
        # self.transformer = TransformerLayer(seq_length=features+asr_length)
        self.proj = nn.Sequential(nn.Linear(features * bucket_dim, 256, bias=True), nn.ReLU(inplace=True), nn.Linear(256, llm_dim, bias=True))



    def auto_dis_forward(self,x):
        #  b feature_num 1 3
        x = x.unsqueeze(2)
        x = x.expand((x.shape[0],x.shape[1],3)).unsqueeze(2) # make sure the shape is consistent with the original code

        #  b feature_num 1 layer_conf[0]
        x = torch.matmul(x,self.linear1_w)+self.linear1_b
        x = self.LeakyReLU(x)
#         x = self.Dropout(x)

        # b feature_num 1 layer_conf[-1]
        x = self.layer(x)

        #  b feature_num 1 bucket_num
        x = torch.matmul(x,self.linear2_w)+self.linear2_b
        x = self.LeakyReLU(x)

        # b feature_num bucket_num
        x = (x * self._tau_module).squeeze(2)
        x = self.Softmax(x)

        # b feature_num bucket_num bucket_dim
        x = x.unsqueeze(-1)*self.emb

        # b feature_num bucket_dim
        x = torch.sum(x,dim=-2)

        # b feature_num*bucket_num
        x = torch.flatten(x,start_dim=1)

        # b output_size if use_fc else feature_num*bucket_num
        x = self.fc(x)

        return x

    def forward(self, stat_features):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
        stat_features[stat_features > 65536.0] = torch.tensor(65536.0).to(stat_features.dtype)
        stat_features[stat_features < 0.0] = 0.0
        x = self.auto_dis_forward(stat_features)
        # x = torch.concat([asr_embeddings, mm_embeddings, y], dim=1)
        x = self.proj(x)
        
        return x