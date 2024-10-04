import torch
from wenet.transformer.encoder_cat import ConformerEncoder
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d

class Conformer(torch.nn.Module):
    def __init__(self, n_mels=80, num_blocks=6, output_size=256, embedding_dim=192, input_layer="conv2d2", 
            pos_enc_layer_type="rel_pos"):

        super(Conformer, self).__init__()
        print("input_layer: {}".format(input_layer))
        print("pos_enc_layer_type: {}".format(pos_enc_layer_type))
        self.conformer = ConformerEncoder(input_size=n_mels, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)

        self.mini_pooling_1 = AttentiveStatisticsPooling(output_size)
        self.mini_pooling_2 = AttentiveStatisticsPooling(output_size)
        self.mini_pooling_3 = AttentiveStatisticsPooling(output_size)
        self.mini_pooling_4 = AttentiveStatisticsPooling(output_size)
        self.mini_pooling_5 = AttentiveStatisticsPooling(output_size)
        self.mini_pooling_6 = AttentiveStatisticsPooling(output_size)

        self.mini_bn_1 = BatchNorm1d(input_size=output_size*2)
        self.mini_bn_2 = BatchNorm1d(input_size=output_size*2)
        self.mini_bn_3 = BatchNorm1d(input_size=output_size*2)
        self.mini_bn_4 = BatchNorm1d(input_size=output_size*2)
        self.mini_bn_5 = BatchNorm1d(input_size=output_size*2)
        self.mini_bn_6 = BatchNorm1d(input_size=output_size*2)

        self.mini_fc_1 = torch.nn.Linear(output_size*2, embedding_dim)
        self.mini_fc_2 = torch.nn.Linear(output_size*2, embedding_dim)
        self.mini_fc_3 = torch.nn.Linear(output_size*2, embedding_dim)
        self.mini_fc_4 = torch.nn.Linear(output_size*2, embedding_dim)
        self.mini_fc_5 = torch.nn.Linear(output_size*2, embedding_dim)
        self.mini_fc_6 = torch.nn.Linear(output_size*2, embedding_dim)

        self.pooling = AttentiveStatisticsPooling(output_size*num_blocks)
        self.bn = BatchNorm1d(input_size=output_size*num_blocks*2)
        self.fc = torch.nn.Linear(output_size*num_blocks*2, embedding_dim)

    
    def forward(self, feat):
        feat = feat.squeeze(1).permute(0, 2, 1)
        lens = torch.ones(feat.shape[0]).to(feat.device)
        lens = torch.round(lens*feat.shape[1]).int()
        x, cat_outputs = self.conformer(feat, lens)
        x = x.permute(0, 2, 1)
        cat_outputs = cat_outputs.permute(0, 2, 1)

        # cat_outputs = x

        embed_len = int(cat_outputs.shape[1]/6)

        pooled_output = self.pooling(x)
        normed_output = self.bn(pooled_output)
        normed_output = normed_output.permute(0, 2, 1)
        x = self.fc(normed_output)
        x = x.squeeze(1)

        cat_outputs_1 = cat_outputs[:, :embed_len, :]
        pooled_cat_outputs_1 = self.mini_pooling_1(cat_outputs_1)
        normed_cat_outputs_1 = self.mini_bn_1(pooled_cat_outputs_1)
        normed_cat_outputs_1 = normed_cat_outputs_1.permute(0, 2, 1)
        x_concat_1 = self.mini_fc_1(normed_cat_outputs_1)
        x_concat_1 = x_concat_1.squeeze(1)

        cat_outputs_2 = cat_outputs[:, embed_len:2*embed_len, :]
        pooled_cat_outputs_2 = self.mini_pooling_2(cat_outputs_2)
        normed_cat_outputs_2 = self.mini_bn_2(pooled_cat_outputs_2)
        normed_cat_outputs_2 = normed_cat_outputs_2.permute(0, 2, 1)
        x_concat_2 = self.mini_fc_2(normed_cat_outputs_2)
        x_concat_2 = x_concat_2.squeeze(1)

        cat_outputs_3 = cat_outputs[:, 2*embed_len:3*embed_len, :]
        pooled_cat_outputs_3 = self.mini_pooling_3(cat_outputs_3)
        normed_cat_outputs_3 = self.mini_bn_3(pooled_cat_outputs_3)
        normed_cat_outputs_3 = normed_cat_outputs_3.permute(0, 2, 1)
        x_concat_3 = self.mini_fc_3(normed_cat_outputs_3)
        x_concat_3 = x_concat_3.squeeze(1)

        cat_outputs_4 = cat_outputs[:, 3*embed_len:4*embed_len, :]
        pooled_cat_outputs_4 = self.mini_pooling_4(cat_outputs_4)
        normed_cat_outputs_4 = self.mini_bn_4(pooled_cat_outputs_4)
        normed_cat_outputs_4 = normed_cat_outputs_4.permute(0, 2, 1)
        x_concat_4 = self.mini_fc_4(normed_cat_outputs_4)
        x_concat_4 = x_concat_4.squeeze(1)

        cat_outputs_5 = cat_outputs[:, 4*embed_len:5*embed_len, :]
        pooled_cat_outputs_5 = self.mini_pooling_5(cat_outputs_5)
        normed_cat_outputs_5 = self.mini_bn_5(pooled_cat_outputs_5)
        normed_cat_outputs_5 = normed_cat_outputs_5.permute(0, 2, 1)
        x_concat_5 = self.mini_fc_5(normed_cat_outputs_5)
        x_concat_5 = x_concat_5.squeeze(1)

        cat_outputs_6 = cat_outputs[:, 5*embed_len:, :]
        pooled_cat_outputs_6 = self.mini_pooling_6(cat_outputs_6)
        normed_cat_outputs_6 = self.mini_bn_6(pooled_cat_outputs_6)
        normed_cat_outputs_6 = normed_cat_outputs_6.permute(0, 2, 1)
        x_concat_6 = self.mini_fc_6(normed_cat_outputs_6)
        x_concat_6 = x_concat_6.squeeze(1)

        x_concat_list = [x_concat_1, x_concat_2, x_concat_3, x_concat_4, x_concat_5, x_concat_6]
        x_concat = torch.cat(x_concat_list, dim=-1)

        return x, x_concat

def conformer_cat(n_mels=80, num_blocks=6, output_size=256, 
        embedding_dim=192, input_layer="conv2d", pos_enc_layer_type="rel_pos"):
    model = Conformer(n_mels=n_mels, num_blocks=num_blocks, output_size=output_size, 
            embedding_dim=embedding_dim, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
    return model

 
