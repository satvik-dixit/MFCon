import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import accuracy
from pytorch_metric_learning import losses

class contrastive_amsoftmax(nn.Module):
    def __init__(self, embedding_dim, num_classes, loss_type='supcon', use_outputs=True, all_blocks=True, margin=0.2, scale=30, **kwargs):
        super(contrastive_amsoftmax, self).__init__()

        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.W = torch.nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        self.loss_type = loss_type
        self.use_outputs = use_outputs
        self.all_blocks = all_blocks
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AM-Softmax m=%.3f s=%.3f'%(self.m, self.s))
        print('Embedding dim is {}, number of speakers is {}'.format(embedding_dim, num_classes))

    def forward(self, x, outputs, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        # outputs_norm = torch.norm(outputs, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        # outputs_norm = torch.div(outputs, outputs_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        acc = accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]

        # Default
        # contrastive_loss = 0

        contrastive_loss = self.contrastive_loss_function(x, outputs, label) # loss_type='supcon', use_outputs=True, all_blocks=True
        final_loss = loss + contrastive_loss

        return final_loss, acc, contrastive_loss
    

    def contrastive_loss_function(self, x, outputs, labels): # loss_type='supcon', use_outputs=True, all_blocks=True
        loss_type = self.loss_type
        use_outputs = self.use_outputs
        all_blocks = self.all_blocks
        
        if use_outputs:
            outputs = outputs
        else:
            outputs = x

        if loss_type=='triplet':
            criterion = losses.TripletMarginLoss()
        elif loss_type=='ntxent':
            criterion = losses.NTXentLoss(temperature=0.07)
        elif loss_type=='supcon':
            criterion = losses.SupConLoss(temperature=0.07)
        elif loss_type== 'npairs':
            criterion = losses.NPairsLoss()

        outputs = outputs.squeeze()

        # all_blocks = False

        if all_blocks==True:
            embed_len = int(outputs.shape[1]/6)
            output_1 = outputs[:, :embed_len]
            output_2 = outputs[:, embed_len:2*embed_len]
            output_3 = outputs[:, 2*embed_len:3*embed_len]
            output_4 = outputs[:, 3*embed_len:4*embed_len]
            output_5 = outputs[:, 4*embed_len:5*embed_len]
            output_6 = outputs[:, 5*embed_len:6*embed_len]
            contrastive_loss = (criterion(output_1, labels) + criterion(output_2, labels) + criterion(output_3, labels) + criterion(output_4, labels) + criterion(output_5, labels) + criterion(output_6, labels))/6
            # contrastive_loss = criterion(output_2, labels)
            # contrastive_loss = (1.5*criterion(output_1, labels) + 1.3*criterion(output_2, labels) + 1.1*criterion(output_3, labels) + 0.9*criterion(output_4, labels) + 0.7*criterion(output_5, labels) + 0.5*criterion(output_6, labels))/6
            # contrastive_loss = (3*criterion(output_1, labels) + 1.5*criterion(output_2, labels) + 0.75*criterion(output_3, labels) + 0.375*criterion(output_4, labels) + 0.1875*criterion(output_5, labels) + 0.009375*criterion(output_6, labels))/6
            # contrastive_loss = (3*criterion(output_1, labels) + 3*criterion(output_2, labels))/6
           
            # contrastive_loss = (6*criterion(output_6, labels))/6

        else:
            contrastive_loss = criterion(outputs, labels)

        amsupcon_loss = criterion(x, labels)

        # Change coefficients if necessary
        contrastive_loss = 0.01*contrastive_loss # + 0.01*amsupcon_loss


        return contrastive_loss






