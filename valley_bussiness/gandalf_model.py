from transformers import XLMRobertaConfig, RobertaModel
from models import *
from falbert import *
#from roberta_varied_emb import RobertaVariedEmbModel
from roberta_varied import RobertaVariedEmbModel
import torch.nn.functional as F
import copy

class Live_Gandalf_Model_V1(torch.nn.Module):
    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=128,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
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
            add_block=False
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
        
        self.transformer = nn.Sequential(
            nn.Linear(asr_embedding_dim + features * bucket_dim, 256, bias=True),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=self._drop_prob, inplace=False),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=self._drop_prob, inplace=False),
        )
        if binary:
            self.classifier = nn.Sequential(nn.Linear(128, 1, bias=True))# , nn.Sigmoid()
        else:
            self.classifier = nn.Sequential(nn.Linear(128, 4, bias=True))# , nn.Sigmoid()
            
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
    
    def forward(self, stat_features, asr_embeddings):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
#         stat_features[stat_features > 65536.0] = 65536.0
        stat_features[stat_features < 0.0] = 0.0
        y = self.auto_dis_forward(stat_features)
        x = torch.concat([asr_embeddings, y],dim=1)
        x = self.transformer(x)
        z = self.classifier(x)
        return x, z
    
class Live_Gandalf_Model_V2(torch.nn.Module):
    '''
    A model conbine BertMergeModel and ctr features as linear, inspirit by wide&deep.
    '''

    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
            num_class = 4,
            bucket_dim = 128,
            drop_prob = 0.3,
            binary = True
            ):
        super().__init__()
        self.feature_dim  = features
        self._out_dim = num_class
        self._drop_prob = drop_prob
        self.auto_dis = AutoDisBucketEncoderV3(
            feature_num=features,
            use_fc=False,
            add_block=False,
            bucket_dim=bucket_dim,
            output_size=1024
        )
        #self.auto_dis = AutoDiscretizationLayer(features, bucket_size, embedding_dim, temperature, skip_alpha)
        # asr_length = 128
        # self.transformer = TransformerLayer(seq_length=features+asr_length)
        
#         self.asr_config = XLMRobertaConfig.from_pretrained("xlm-roberta-base")
#         self.asr_model = RobertaModel(self.asr_config, add_pooling_layer=False).from_pretrained('./xlmr_base_cmt')
        
        self.asr_config = XLMRobertaConfig.from_pretrained("xlm-roberta-large")
        #self.asr_model = RobertaVariedEmbModel(self.asr_config, add_pooling_layer=False).from_pretrained('./asr_model/live_asr_pretrain_512')
        self.asr_model = RobertaVariedEmbModel(self.asr_config, add_pooling_layer=False).from_pretrained('./checkpoint-24000-new-nlp-general-1201-5cls')
        asr_embedding_dim = self.asr_config.hidden_size
        self.transformer = nn.Sequential(
            nn.Linear(asr_embedding_dim + features * bucket_dim, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self._drop_prob, inplace=False),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self._drop_prob, inplace=False),
        )
        if binary:
            self.classifier = nn.Sequential(nn.Linear(128, 1, bias=True))# , nn.Sigmoid()
        else:
            self.classifier = nn.Sequential(nn.Linear(128, 4, bias=True))# , nn.Sigmoid()

    def forward(self, stat_features, asr):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
        stat_features[stat_features > 65536.0] = 65536.0
        stat_features[stat_features < 0.0] = 0.0
        #asr_embeddings = self.asr_model(input_ids=asr['input_ids'],attention_mask=asr['attention_mask']).pooler_output
        asr_embeddings = self.asr_model(input_ids=asr['input_ids'], attention_mask=asr['attention_mask'])[0][:, 0, :]
        y = self.auto_dis(stat_features)
        x = torch.concat([asr_embeddings, y],axis=1)
        x = self.transformer(x)
        z = self.classifier(x)
        return x, z
    
    
class Live_Gandalf_Model_V3(torch.nn.Module):
    '''
    A model conbine BertMergeModel and ctr features as linear, inspirit by wide&deep.
    '''

    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
            num_class = 4,
            bucket_dim = 128,
            drop_prob = 0.3,
            binary = True
            ):
        super().__init__()
        self.feature_dim  = features
        self._out_dim = num_class
        self._drop_prob = drop_prob
        self.auto_dis = AutoDisBucketEncoderV3(
            feature_num=features,
            use_fc=False,
            add_block=False,
            bucket_dim=bucket_dim,
            output_size=1024
        )
        self.asr_config = XLMRobertaConfig.from_pretrained("xlm-roberta-large")
        self.asr_model = RobertaVariedEmbModel(self.asr_config, add_pooling_layer=False).from_pretrained('./checkpoint-24000-new-nlp-general-1201-5cls')
        from types import SimpleNamespace
        import yaml
        with open('./config_backbone.yaml') as fp:
            config_backbone = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        self.backbone = FalBertModel(config_backbone)
        #state_dict = torch.load('model_state_epoch_150000.th', map_location="cpu")
        state_dict = torch.load('epoch=4-step=230000-val_loss=1.775.ckpt', map_location="cpu")['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("backbone."):
                # remove prefix
                state_dict[k[len("backbone.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        self.backbone.load_state_dict(state_dict)
        self.transformer = nn.Sequential(
            nn.Linear(asr_embedding_dim + features * bucket_dim + config_backbone.hidden_size, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self._drop_prob, inplace=False),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self._drop_prob, inplace=False),
        )
        if binary:
            self.classifier = nn.Sequential(nn.Linear(128, 1, bias=True))# , nn.Sigmoid()
        else:
            self.classifier = nn.Sequential(nn.Linear(128, 4, bias=True))# , nn.Sigmoid()
        
    def forward(self, stat_features, asr, frames_data, frames_mask, input_ids, input_masks, input_segment_ids):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
        stat_features[stat_features > 65536.0] = 65536.0
        stat_features[stat_features < 0.0] = 0.0
        #asr_embeddings = self.asr_model(input_ids=asr['input_ids'], attention_mask=asr['attention_mask']).pooler_output
        asr_embeddings = self.asr_model(input_ids=asr['input_ids'], attention_mask=asr['attention_mask'])[0][:, 0, :]
        y = self.auto_dis(stat_features)
        mm = self.backbone(input_ids=input_ids,
        input_segment_ids=input_segment_ids,
        input_mask=input_masks,
        frames=frames_data,
        frames_mask=frames_mask)
        x = torch.concat([asr_embeddings, y, mm['pooled_output']],axis=1)
        x = self.transformer(x)
        z = self.classifier(x)
        return x, z
    
class Live_Gandalf_Model_V6(torch.nn.Module):
    '''
    A model conbine BertMergeModel and ctr features as linear, inspirit by wide&deep.
    '''

    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=128,
            mm_embedding_dim: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
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
            add_block=False
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
        
        self.transformer = nn.Sequential(
            nn.Linear(asr_embedding_dim + mm_embedding_dim + features * bucket_dim, 256, bias=True),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=self._drop_prob, inplace=False),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=self._drop_prob, inplace=False),
        )
        if binary:
            self.classifier = nn.Sequential(nn.Linear(128, 1, bias=True))# , nn.Sigmoid()
        else:
            self.classifier = nn.Sequential(nn.Linear(128, 4, bias=True))# , nn.Sigmoid()
            
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
    
    def forward(self, stat_features, asr_embeddings, mm_embeddings):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
#         stat_features[stat_features > 65536.0] = 65536.0
        stat_features[stat_features < 0.0] = 0.0
        y = self.auto_dis_forward(stat_features)
        x = torch.concat([asr_embeddings, mm_embeddings, y], dim=1)
        x = self.transformer(x)
        z = self.classifier(x)
        return x, z
    
class Live_Gandalf_Model_V7(torch.nn.Module):
    '''
    A model conbine BertMergeModel and ctr features as linear, inspirit by wide&deep.
    '''

    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=128,
            mm_embedding_dim: int=768,
            hist_features_num: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
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
            time_step = 5
            ):
        super().__init__()
        self.feature_dim  = features
        self._out_dim = num_class
        self._drop_prob = drop_prob
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.score_embed_dim = 64
        self.proj = nn.Conv1d(1, self.score_embed_dim, kernel_size=time_step, stride=time_step)
        self.score_length = self.proj(torch.randn(2, hist_features_num).unsqueeze(1)).transpose(1, 2).shape[1]
        self.hist_transformer = TransformerLayer(seq_length=self.score_length, embed_dim = self.score_embed_dim, depth=4, num_heads=8)
        feature_num = features + self.score_embed_dim * (self.score_length + 1)
        #self.Dropout = nn.Dropout(p=dropout)
        self.linear1_w = nn.Parameter(torch.randn(feature_num,3,layer_conf[0]))
        self.linear1_b = nn.Parameter(torch.randn(feature_num,1,layer_conf[0]))
        self.layer = nn.Sequential(*[ResBlock(feature_num, layer_len, dropout=0, alpha=alpha) for layer_len in layer_conf]) if add_block else nn.Identity()
        self.linear2_w = nn.Parameter(torch.randn(feature_num,layer_conf[-1],bucket_num))
        self.linear2_b = nn.Parameter(torch.randn(feature_num,1,bucket_num))
        self.hist_features_num = hist_features_num
        self.emb = nn.Parameter(torch.randn(feature_num,bucket_num,bucket_dim))
        self._tau_module = nn.Parameter(torch.ones([feature_num,1,bucket_num]))
        self.Softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(feature_num*bucket_dim,output_size,bias=True) if use_fc else nn.Identity()

        nn.init.kaiming_uniform_(self.linear1_w, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.linear2_w, mode='fan_in', nonlinearity='leaky_relu')
    
        #self.auto_dis = AutoDiscretizationLayer(features, bucket_size, embedding_dim, temperature, skip_alpha)
        # asr_length = 128
        # self.transformer = TransformerLayer(seq_length=features+asr_length)
        #self.hist_transformer = TransformerLayer(seq_length=hist_features_num, embed_dim = hist_features_num, depth=4, num_heads=5)
       
        self.transformer = nn.Sequential(
            nn.Linear(asr_embedding_dim + mm_embedding_dim + feature_num * bucket_dim , 256, bias=True),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=self._drop_prob, inplace=False),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=self._drop_prob, inplace=False),
        )
        if binary:
            self.classifier = nn.Sequential(nn.Linear(128, 1, bias=True))# , nn.Sigmoid()
        else:
            self.classifier = nn.Sequential(nn.Linear(128, 4, bias=True))# , nn.Sigmoid()
            
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
    
    def forward(self, stat_features, asr_embeddings, mm_embeddings, hist_features):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
#         stat_features[stat_features > 65536.0] = 65536.0
        stat_features[stat_features < 0.0] = 0.0
        bs = len(stat_features)
        hist_features = self.proj(hist_features.unsqueeze(1)).transpose(1, 2)
        hist_features = self.hist_transformer(hist_features).reshape(bs,-1)
        y = self.auto_dis_forward(torch.concat([hist_features, stat_features],dim=1))
        x = torch.concat([asr_embeddings, mm_embeddings, y], dim=1)
        x = self.transformer(x)
        z = self.classifier(x)
        return x, z

class TopKGate(torch.nn.Module):
    """Gate module which implements TopKGating.

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 jitter_scale: float = 1e-2,
                 gate_bias=False) -> None:
        super().__init__()

        self.wg = torch.nn.Linear(model_dim, num_experts, bias=gate_bias).float()
        self.k = k
        self.num_experts = num_experts
        self.soft_temperature = False
        self.z_loss = torch.tensor(0)
        self.jitter_scale = jitter_scale

    def forward(
        self,
        input: torch.Tensor,
        k: int
    ):
        input_fp32 = input.float()
        logits = self.wg(input_fp32)
        z_loss_ep = torch.sum(torch.exp(logits), dim=1)
        self.z_loss = torch.sum(torch.log(z_loss_ep) ** 2) / logits.shape[0]
        gates = F.softmax(logits, dim=1)
        token_num = gates.shape[0] * k
        num_experts = gates.shape[1]

        topk_indices = torch.topk(gates, k, dim=1).indices
        indices = torch.jit.annotate(List[torch.Tensor], [])
        for x in topk_indices.chunk(k, dim=1):
            indices.append(x.view(-1))
            
        masks = torch.jit.annotate(List[torch.Tensor], [])
        for x in indices:
            masks.append(one_hot(x, num_classes=num_experts))
        
        weights = torch.jit.annotate(List[torch.Tensor], [])
        
        for x in masks:
            weights.append((gates * x).sum(dim=1))
        
        weight_norm = weights[0]+weights[1]
        
        weights[0] = weights[0] / weight_norm
        weights[1] = weights[1] / weight_norm

        mask = torch.cat(masks, dim=0)
        weight = torch.cat(weights, dim=0)

        token_to_which_expert = torch.cat(indices, dim=0)

        # token num of each expert, dim (1, num_experts)
        expert_token_count = torch.sum(mask, dim=0)

        # token order in each expert, dim (token_num, 1)
        token_pos_in_each_expert = torch.sum((cumsum_sub_one(mask)) * mask, dim=1)

        # calc topk gate loss
        me = torch.mean(gates, dim=0)
        ce = torch.mean(masks[0].float(), dim=0)
        l_aux = torch.sum(me * ce) * num_experts

        # redirect each token buffer to its original position
        expert_cum_token_count = torch.cumsum(expert_token_count, dim=0) - expert_token_count
        token_pos_after_transfer = token_pos_in_each_expert + \
            expert_cum_token_count[token_to_which_expert]
        token_pos_before_transfer = torch.empty_like(token_pos_after_transfer)
        token_pos_before_transfer[token_pos_after_transfer] = torch.arange(
            token_num, device=logits.device)
        return l_aux, token_pos_after_transfer, token_pos_before_transfer, expert_token_count, weight 

def one_hot(tensor: torch.Tensor, num_classes: int):
    ret = torch.zeros(tensor.shape + (num_classes,), device=tensor.device, dtype=tensor.dtype)
    ret.scatter_(-1, tensor.unsqueeze(-1), 1)
    return ret

def cumsum_sub_one(mask):
    return torch.cumsum(mask, dim=0) - torch.ones_like(mask)

def split_util(tensor, splits_tensor):
    split_dim = splits_tensor.shape[0]
    list_of_split_tensors = []
    begin_indx = 0
    for i in range(split_dim):
        end_indx = begin_indx + splits_tensor[i]
        list_of_split_tensors += [tensor[begin_indx:end_indx]]
        begin_indx = end_indx
    return list_of_split_tensors

@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor: # `input` has a same name in Sequential forward
        pass
    
class Experts(torch.nn.Module):
    def __init__(self, expert, num_local_experts=1):
        super(Experts, self).__init__()
        self.experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for _ in range(num_local_experts)])
        self.num_local_experts = num_local_experts
        
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
    
    def forward(self, inputs, splits):
        # equal to inputs.split(splits, dim=0)
        ts = split_util(inputs, splits)
        expert_outputs = []
        for index in range(len(ts)):
            submodule: ModuleInterface = self.experts[index%self.num_local_experts]
            result = submodule.forward(ts[index])
            expert_outputs += [result]
        expert_output = torch.cat(expert_outputs, dim=0)
        return expert_output

class MoE(torch.nn.Module):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self,
                 experts: torch.nn.Module,
                 num_experts: int,
                 hidden_size: int = 128,
                 k: int = 1,
                 use_capacity: bool = False,
                 capacity_factor: float = 1.0,
                 expert_shape: str = 'abc->abd') -> None:
        super().__init__()
        self.gate = TopKGate(model_dim = hidden_size,
                             num_experts = num_experts,
                             k=k)
        self.experts = Experts(experts,num_experts)
        self.num_local_experts = num_experts
        self.k = k
        self.rank = 0
        self.current_step = 0
        self.use_capacity = use_capacity
        self.l_aux = torch.tensor(0)
        self.capacity_factor = capacity_factor

    def forward(self, input):
        self.current_step += 1

        reshaped_input = input

        token_num = reshaped_input.shape[0] * self.k

        self.l_aux, token_pos_after_transfer, token_pos_before_transfer, input_splits, weight = self.gate(
            reshaped_input, self.k)
        reshaped_input = reshaped_input.repeat((self.k, 1))

        # the token number of each expert
        expert_token_count = input_splits

        output_splits = torch.clone(input_splits).detach()

        # dispatch input before alltoall, dim (token_num, hidden_dim)
        reshaped_input = reshaped_input[token_pos_before_transfer]

        # dim (token_num, output_dim)
        expert_output = self.experts(reshaped_input, output_splits)

        # gather output and combine weight
        expert_output = expert_output[token_pos_after_transfer]
        expert_output = expert_output * weight.reshape(-1, 1).type_as(expert_output)

        token_num = token_num // self.k
        tmp_output = expert_output[:token_num]
        tmp_output += expert_output[1 * token_num: 2 * token_num]
        expert_output = tmp_output
        return self.l_aux, expert_output

class HardGate(torch.nn.Module):
    """Gate module which implements hard routing.
    """

    def __init__(self, num_experts):
        super(HardGate, self).__init__()
        self.num_experts = num_experts
        self.z_loss = 0

    def forward(
        self,
        inputs: torch.Tensor,
        mapping: torch.Tensor,
    ):
        dim0 = inputs.shape[0]
        _, sorted_sample_pos = torch.sort(mapping)
        reverse_sorted_sample_pos = torch.arange(dim0, device=inputs.device)
        reverse_sorted_sample_pos[sorted_sample_pos] = torch.arange(dim0, device=inputs.device)
        expert_splits = torch.sum(torch.nn.functional.one_hot(
            mapping.long(), num_classes=self.num_experts), dim=0)
        return inputs[sorted_sample_pos], expert_splits, reverse_sorted_sample_pos
    
class HardGateMOE(torch.nn.Module):
    def __init__(self,
                 experts: torch.nn.Module,
                 hidden_size: int,
                 num_experts: int,
                 use_softmoe_expert=False,
                 assign=None,
                 gate_bias: bool = False,
                 k: int = 1
                 ) -> None:
        super().__init__()
        self.gate = HardGate(num_experts)
        self.experts = Experts(experts,num_experts)
        self.num_local_experts = num_experts
        self.topk = k
        self.hidden_size = hidden_size
        self.use_softmoe_expert = use_softmoe_expert
        self.wg = torch.nn.Linear(hidden_size, num_experts, bias=gate_bias).float()

    def forward(self, input, mapping):
        """
        self.l_aux = torch.tensor(0, device=input.device, dtype=input.dtype)
        """
        origin_shape = input.shape
        logits = F.softmax(
            self.wg(input.reshape(-1, self.hidden_size).to(self.wg.weight.dtype)), dim=0)
        input = input.reshape(-1, origin_shape[-1]).repeat(self.topk, 1)
        #mapping = mapping.reshape(mapping.shape[0], 1, self.topk).repeat(1, origin_shape[1], 1)
        mapping = mapping.reshape(-1, self.topk).permute(1, 0).reshape(-1)
        reshaped_input, input_splits, token_pos_after_transfer = self.gate(
            input, mapping=mapping)
        input_splits = torch.tensor(input_splits).cuda()

        # collect token number of other ranks
        output_splits = torch.clone(input_splits).detach()

        expert_output = self.experts(reshaped_input, output_splits)

        # gather output and combine weight
        expert_output = expert_output[token_pos_after_transfer]

        reshaped_mapping = mapping.reshape(self.topk, -1).permute(1, 0)
        weight = torch.gather(logits, dim=1, index=reshaped_mapping.long())
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)
        expert_output = expert_output * \
            weight.permute(1, 0).reshape(-1, 1).type_as(expert_output)
        token_num = expert_output.shape[0] // self.topk
        tmp_output = expert_output[:token_num]
        for i in range(1, self.topk):
            tmp_output += expert_output[i * token_num: (i + 1) * token_num]
        expert_output = tmp_output
        return expert_output

class Live_Gandalf_Model_V8(torch.nn.Module):
    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=128,
            mm_embedding_dim: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
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
            add_block=False
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
        self.proj = nn.Sequential(nn.Linear(asr_embedding_dim + mm_embedding_dim + features * bucket_dim, 256, bias=True),nn.ReLU(inplace=True))

        if binary:
            self.expert = nn.Sequential(
                nn.Linear(256, 128, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(128, 1, bias=True)
            )
        else:
            self.expert = nn.Sequential(
                nn.Linear(256, 128, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(128, 4, bias=True)
            )
        self.experts = MoE(hidden_size=256, experts=self.expert, num_experts=4, k=2, expert_shape='ab->ac')
        
        
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
    
    def forward(self, stat_features, asr_embeddings, mm_embeddings):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
#         stat_features[stat_features > 65536.0] = 65536.0
        stat_features[stat_features < 0.0] = 0.0
        y = self.auto_dis_forward(stat_features)
        x = torch.concat([asr_embeddings, mm_embeddings, y], dim=1)
        x = self.proj(x)
        w, z =  self.experts(x)
        return w.unsqueeze(0), z

class Live_Gandalf_Model_V9(torch.nn.Module):
    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=128,
            mm_embedding_dim: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
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
            add_block=False
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
        self.proj = nn.Sequential(nn.Linear(asr_embedding_dim + mm_embedding_dim + features * bucket_dim, 256, bias=True),nn.ReLU(inplace=True))
        if binary:
            self.expert = nn.Sequential(
                nn.Linear(256, 128, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1, bias=True)
            )
        else:
            self.expert = nn.Sequential(
                nn.Linear(256, 128, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(128, 4, bias=True)
            )
        self.experts = HardGateMOE(hidden_size=256, experts=self.expert, num_experts=4, k=2)
            
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
    
    def forward(self, stat_features, asr_embeddings, mm_embeddings, country_route):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
#         stat_features[stat_features > 65536.0] = 65536.0
        stat_features[stat_features < 0.0] = 0.0
        y = self.auto_dis_forward(stat_features)
        x = torch.concat([asr_embeddings, mm_embeddings, y], dim=1)
        x = self.proj(x)
        z =  self.experts(x, country_route)
        return z
    
class Live_Gandalf_Model_V10(torch.nn.Module):
    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=128,
            mm_embedding_dim: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
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
        self.proj = nn.Sequential(nn.Linear(asr_embedding_dim + mm_embedding_dim + features * bucket_dim, 256, bias=True),nn.ReLU(inplace=True))

        self.expert = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, 1, bias=True)
        )
        self.multihead = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, num_classes, bias=True)
        )
        self.experts = MoE(hidden_size=256, experts=self.expert, num_experts=4, k=2, expert_shape='ab->ac')
        
        
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
    
    def forward(self, stat_features, asr_embeddings, mm_embeddings):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
#         stat_features[stat_features > 65536.0] = 65536.0
        stat_features[stat_features < 0.0] = 0.0
        y = self.auto_dis_forward(stat_features)
        x = torch.concat([asr_embeddings, mm_embeddings, y], dim=1)
        x = self.proj(x)
        w, z =  self.experts(x)
        m = self.multihead(x)
        return w.unsqueeze(0), z, m
    
class Live_Gandalf_Model_V11(torch.nn.Module):
    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=128,
            mm_embedding_dim: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
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
        self.proj = nn.Sequential(nn.Linear(asr_embedding_dim + mm_embedding_dim + features * bucket_dim, 256, bias=True),nn.ReLU(inplace=True))

        self.expert = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, 4, bias=True)
        )
        self.multihead = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, num_classes, bias=True)
        )
        self.experts = MoE(hidden_size=256, experts=self.expert, num_experts=4, k=2, expert_shape='ab->ac')
        
        
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
    
    def forward(self, stat_features, asr_embeddings, mm_embeddings):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
#         stat_features[stat_features > 65536.0] = 65536.0
        stat_features[stat_features < 0.0] = 0.0
        y = self.auto_dis_forward(stat_features)
        x = torch.concat([asr_embeddings, mm_embeddings, y], dim=1)
        x = self.proj(x)
        w, z =  self.experts(x)
        m = self.multihead(x)
        return w.unsqueeze(0), z, m

class lgm_classifier(torch.nn.Module):
    def __init__(self, num_classes, feature_num):
        super().__init__()
        self.alpha=0.1 
        self.lambda_=0.01
        self.num_classes = num_classes
        self.centers = torch.nn.Embedding(num_classes, feature_num)
        
    def forward(self,feat,labels):
        N = feat.size(0)
        feat_len = feat.size(1)

        XY = torch.matmul(feat, self.centers.weight.t())
        XX = torch.sum(torch.square(feat), dim=1, keepdim=True)
        YY = torch.sum(torch.square(self.centers.weight), dim=1, keepdim=True)
        neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY.T)
        
        if labels is None:
            psudo_labels = torch.argmax(neg_sqr_dist, dim=1)
            means_batch = torch.index_select(self.centers.weight, 0, psudo_labels)
            likelihood_reg_loss = self.lambda_ * F.mse_loss(feat,means_batch, reduction='sum') / N
            return neg_sqr_dist, likelihood_reg_loss
        
        ALPHA = F.one_hot(labels.long().squeeze(1), self.num_classes).float() * self.alpha
        K = ALPHA.cuda() + torch.ones(N, self.num_classes).cuda()
        logits_with_margin = neg_sqr_dist * K

        means_batch = torch.index_select(self.centers.weight, 0, labels.long().squeeze(1)).cuda()
        likelihood_reg_loss = self.lambda_ * F.mse_loss(feat, means_batch, reduction='sum') / N
        return logits_with_margin, likelihood_reg_loss
        
class Live_Gandalf_Model_V12(torch.nn.Module):
    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=128,
            mm_embedding_dim: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
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
        self.proj = nn.Sequential(nn.Linear(asr_embedding_dim + mm_embedding_dim + features * bucket_dim, 256, bias=True),nn.ReLU(inplace=True))
    
        self.expert = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.classifier = lgm_classifier(2,128)
        self.multihead = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, num_classes, bias=True)
        )
        
        self.experts = MoE(hidden_size=256, experts=self.expert, num_experts=4, k=2, expert_shape='ab->ac')
        
        
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
    
    def forward(self, stat_features, asr_embeddings, mm_embeddings, labels=None):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
#         stat_features[stat_features > 65536.0] = 65536.0
        stat_features[stat_features < 0.0] = 0.0
        y = self.auto_dis_forward(stat_features)
        x = torch.concat([asr_embeddings, mm_embeddings, y], dim=1)
        x = self.proj(x)
        w, z =  self.experts(x)
        z, reg_loss = self.classifier(z, labels)
        m = self.multihead(x)
        return w.unsqueeze(0), z, m, reg_loss
    
def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

class PPO:
    def __init__(self, model, ref_model, critic_model, reward_model, criterion, criterion_mh):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_model = model
        self.ref_model = ref_model
        self.critic_model = critic_model
        self.reward_model = reward_model
        #self.optimizer = torch.optim.Adam(list(self.actor_model.parameters())+list(self.critic_model.parameters()), lr=0.001)
        #self.optimizer = torch.optim.SGD(list(self.actor_model.parameters())+list(self.critic_model.parameters()), lr=1e-5)
        # self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=0.001)
        # self.critic_optimizer = torch.optim.SGD(self.critic_model.parameters(), lr=1e-7)
        self.actor_optimizer = torch.optim.SGD(self.actor_model.parameters(), lr=1e-7)
        self.critic_optimizer = torch.optim.SGD(self.actor_model.parameters(), lr=1e-7)#torch.optim.Adam(self.critic_model.parameters(), lr=0.001)
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.generate_time = 0.0
        self.criterion_mh = torch.nn.CrossEntropyLoss(reduction='none')
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        
    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def actor_loss_fn(self, logprobs, old_logprobs, advantages):
        log_ratio = (logprobs - old_logprobs)
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))
        return pg_loss
    
    def critic_loss_fn(self, values, old_values, returns):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.mean(
            torch.max(vf_loss1, vf_loss2))
        return vf_loss
    
    def get_advantages_and_returns(self, values, rewards):
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns
    
    def compute_rewards(self, log_probs, ref_log_probs, reward_score):
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)
        rewards[:,-2:-1] += reward_clip[:,0:1]#reward_clip.sum(dim=1,keepdim=True)
        return rewards

    def forward(self, feature, asr, mm, label, multi_label, train=True):
        if train:
            self.train()
        else:
            self.eval()
            
        moe_loss, logit, multi_logit = self.actor_model(feature, asr, mm)
        logits = F.sigmoid(logit).cuda()
        multi_logits = F.softmax(multi_logit,dim=1).cuda()
        
        with torch.no_grad():
            _, ref_logit, ref_multi_logit = self.ref_model(feature.clone(), asr.clone(), mm.clone())
            ref_logits = F.sigmoid(ref_logit).cuda()
            ref_multi_logits = F.softmax(ref_multi_logit,dim=1).cuda()
        
        action = torch.where(logits>=0.5*torch.ones_like(logits),torch.ones_like(logits),torch.zeros_like(logits)).cuda()
        multi_action = torch.argmax(multi_logits,dim=1).detach()
        
        with torch.no_grad():
            reward_score = self.reward_model(feature.clone(), asr.clone(), mm.clone(), action, multi_action).cuda()
        
        values = self.critic_model(feature.clone(), asr.clone(), mm.clone(), action, multi_action).cuda()
        
#         logprobs = torch.log(torch.concat([1-logits,logits],dim=1)).gather(dim=-1,index = action.long().detach())
#         ref_logprobs = torch.log(torch.concat([1-ref_logits,ref_logits],dim=1)).gather(dim=-1,index = action.long().detach())
        
#         multi_logprobs = torch.log(multi_logits).gather(dim=-1,index = multi_action.unsqueeze(-1))
#         ref_multi_logprobs = torch.log(ref_multi_logits).gather(dim=-1,index = multi_action.unsqueeze(-1))
        
        
#         log_probs = torch.concat([logprobs,multi_logprobs],dim=1)
#         ref_log_probs = torch.concat([ref_logprobs,ref_multi_logprobs],dim=1)
          
        logprobs = torch.log(torch.concat([1-logits,logits],dim=1))
        ref_logprobs = torch.log(torch.concat([1-ref_logits,ref_logits],dim=1))
        
        multi_logprobs = torch.log(multi_logits)
        ref_multi_logprobs = torch.log(ref_multi_logits)
        
        log_probs = torch.concat([logprobs,multi_logprobs],dim=1)
        ref_log_probs = torch.concat([ref_logprobs,ref_multi_logprobs],dim=1)

        with torch.no_grad():
            old_rewards = self.compute_rewards(log_probs, ref_log_probs, reward_score)
            advantages, returns = self.get_advantages_and_returns(values, old_rewards.mean(1,True))
        actor_loss = self.actor_loss_fn(log_probs, ref_log_probs.detach(), advantages) + torch.mean(moe_loss)
        critic_loss = self.critic_loss_fn(reward_score, values, returns)
        if train:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_gradients(self.actor_model,0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_gradients(self.critic_model,0.5)
            self.critic_optimizer.step()
        else:
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
        loss = actor_loss + critic_loss
        return logits, multi_logits, loss, reward_score, values, returns, actor_loss, critic_loss, old_rewards.mean(1,True), log_probs, ref_log_probs, reward_score

class Video_Gandalf_Model_V1(torch.nn.Module):
    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=128,
            mm_embedding_dim: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
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
        self.proj = nn.Sequential(nn.Linear(asr_embedding_dim + mm_embedding_dim + features * bucket_dim, 256, bias=True),nn.ReLU(inplace=True))

        self.expert = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, 1, bias=True)
        )
        self.multihead = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, num_classes, bias=True)
        )
        self.experts = MoE(hidden_size=256, experts=self.expert, num_experts=4, k=2, expert_shape='ab->ac')
        
        
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
    
    def forward(self, stat_features, asr_embeddings, mm_embeddings):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
#         stat_features[stat_features > 65536.0] = 65536.0
        stat_features[stat_features < 0.0] = 0.0
        y = self.auto_dis_forward(stat_features)
        x = torch.concat([asr_embeddings, mm_embeddings, y], dim=1)
        x = self.proj(x)
        w, z =  self.experts(x)
        m = self.multihead(x)
        return w.unsqueeze(0), z, m

    
class Live_Gandalf_Model_Reward(torch.nn.Module):
    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=128,
            mm_embedding_dim: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
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
        self.proj = nn.Sequential(nn.Linear(asr_embedding_dim + mm_embedding_dim + features * bucket_dim, 256, bias=True),nn.ReLU(inplace=True))
        
        
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
    
    def forward(self, stat_features, asr_embeddings, mm_embeddings):
        # 参考原代码 https://reckon.bytedance.net/forge/model/17456/versions/35/commit/716135
        stat_features[torch.isnan(stat_features)] = 0.0
#         stat_features[stat_features > 65536.0] = 65536.0
        stat_features[stat_features < 0.0] = 0.0
        y = self.auto_dis_forward(stat_features)
        x = torch.concat([asr_embeddings, mm_embeddings, y], dim=1)
        x = self.proj(x)
        return x
    
class Reward(torch.nn.Module):
    def __init__(self, features_num=2100):
        super().__init__()
        self.backbone = Live_Gandalf_Model_Reward(features=features_num ,asr_embedding_dim = 640, mm_embedding_dim = 3840, embedding_dim = 64, num_classes = 26)
        self.token_embedding = nn.Embedding(2,256)
        #self.mh_token_embedding = nn.Embedding(26,256)
        from transformers import BertModel, BertConfig
        config = BertConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_labels = 1,
            num_hidden_layers = 4
        )
        self.expert = BertModel(config).encoder
        self.v_head = torch.nn.Linear(256,1,bias=False)
        
    def forward(self, feat, asr, mm, token):
        x = self.backbone(feat, asr, mm)
        token_embed = self.token_embedding(torch.tensor(token).long())
        hidden_states = self.expert((x + token_embed.squeeze(1)).unsqueeze(1)).last_hidden_state
        rewards = self.v_head(hidden_states).squeeze(-1)
        return rewards

class Live_Gandalf_Model_RNN(torch.nn.Module):
    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=128,
            mm_embedding_dim: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
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
        self.proj = nn.Sequential(nn.Linear(asr_embedding_dim + mm_embedding_dim + features * bucket_dim, 256, bias=True),nn.ReLU(inplace=True))

        
        # self.experts = MoE(hidden_size=256, experts=self.expert, num_experts=4, k=2, expert_shape='ab->ac')
        
        
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
    
    def forward(self, x):
        stat_features = x[:,0:self.feature_dim]
        asr_embeddings = x[:,self.feature_dim:self.feature_dim+self.asr_embedding_dim]
        mm_embeddings = x[:,self.feature_dim+self.asr_embedding_dim::]
        stat_features[torch.isnan(stat_features)] = 0.0
        stat_features[stat_features < 0.0] = 0.0
        y = self.auto_dis_forward(stat_features)
        x = torch.concat([asr_embeddings, mm_embeddings, y], dim=1)
        x = self.proj(x)
        return x

import torch.nn as nn
class Live_Gandalf_Model_RNN(torch.nn.Module):
    def __init__(self, features: int,
            bucket_size:int=64,
            embedding_dim:int=32, 
            asr_embedding_dim: int=128,
            mm_embedding_dim: int=768,
            temperature:float=1e-2, 
            skip_alpha:float=0.1,
            ad_type: int = 1,
            features_bn: str = 'true',
            hiddens_bn: str = 'true',
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
        self.asr_embedding_dim = asr_embedding_dim
        self.mm_embedding_dim = mm_embedding_dim
        self.feature_dim  = features - self.asr_embedding_dim - self.mm_embedding_dim
        feature_num = self.feature_dim
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
        self.proj = nn.Sequential(nn.Linear(asr_embedding_dim + mm_embedding_dim + feature_num * bucket_dim, 256, bias=True),nn.ReLU(inplace=True))

        hidden_size = asr_embedding_dim + mm_embedding_dim + feature_num * bucket_dim
        self.experts = MoE(hidden_size=hidden_size, experts=self.proj, num_experts=4, k=2, expert_shape='ab->ac')
        
        
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
    
    def forward(self, x):
        stat_features = x[:,0:self.feature_dim]
        asr_embeddings = x[:,self.feature_dim:self.feature_dim+self.asr_embedding_dim]
        mm_embeddings = x[:,self.feature_dim+self.asr_embedding_dim::]
        stat_features[torch.isnan(stat_features)] = 0.0
        stat_features[stat_features < 0.0] = 0.0
        y = self.auto_dis_forward(stat_features)
        x = torch.concat([asr_embeddings, mm_embeddings, y], dim=1)
        w, x = self.experts(x)
        return w, x

class LiveStreamRNN(nn.Module):
    def __init__(self, features_num, num_layers=1, num_classes=26):
        super(LiveStreamRNN, self).__init__()
        self.num_layers = num_layers
        self.encoder = Live_Gandalf_Model_RNN(features_num)
        self.input_size = self.encoder.proj[0].out_features
        self.hidden_size = self.input_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers, batch_first=True)
        # from rwkv import RWKV
        # self.rnn = RWKV()
        self.drop_out = 0.0
        self.expert = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.drop_out, inplace=False),
            nn.Linear(128, 1, bias=True)
        )
        self.multihead = nn.Sequential(
            nn.Linear(256, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.drop_out, inplace=False),
            nn.Linear(128, num_classes, bias=True)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        w, y = self.encoder(x.reshape(-1,x.shape[2]))
        y = y.reshape(x.shape[0],x.shape[1],-1)
        output, state = self.rnn(y)
        hidden, cell = state
        # output, shift_states, wkv_states = self.rnn(y)
        z = self.expert(output)
        m = self.multihead(output)
        return z, m, hidden, cell, w
        # return z, m, shift_states, wkv_states


