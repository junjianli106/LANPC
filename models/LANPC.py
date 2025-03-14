import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#----> Attention module
class Attn_Net(nn.Module):
    
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

#----> Attention Gated module
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x) #[N, 256]
        b = self.attention_b(x) #[N, 256]
        A = a.mul(b) #torch.mul(a, b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class LANPC(nn.Module):
    def __init__(self, n_classes, gate = False):
        super(LANPC, self).__init__()
        fc = [nn.Linear(1024, 256), nn.ReLU()]  # 1024->512
        if gate:
            attention_net = Attn_Net_Gated(L=256, D=256, n_classes=1)
        else:
            attention_net = Attn_Net(L=256, D=256, n_classes=1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        self.fc_rad = nn.Linear(396*2, 256)
        self.classifiers = nn.Linear(256, n_classes)

    def forward(self, **kwargs):

        h = kwargs['data'].float() #[B, n, 1024]
        h = h.squeeze(0)  # [n, 1024]

        rad_h = kwargs['data_rad'].float() #[B, n, 1024]
        rad_h = rad_h.squeeze(0) #[B, n, 1024]

        Crad_h = kwargs['data_Crad'].float() #[B, n, 396]
        Crad_h = Crad_h.squeeze(0) #[B, n, 1024]

        rad_h = torch.cat((rad_h, Crad_h), dim=1)
        print(rad_h.shape)

        #---->Attention
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        h = torch.mm(A, h)

        rad_h = self.fc_rad(rad_h)
        # Crad_h = self.fc_rad(Crad_h)
        h = torch.add(h, rad_h)
        # h = torch.add(h, Crad_h)

        #---->predict output
        logits = self.classifiers(h) #[B, n_classes]
        # print(logits)
        Y_hat = torch.argmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        # print(f'S: {S}, risk: {-torch.sum(S, dim=1).cpu().item()}')
        results_dict = {'hazards': hazards, 'S': S, 'Y_hat': Y_hat}
        # print(results_dict)
        return results_dict
    #
    def forward_vis(self, **kwargs):

        h = kwargs['data'].float() #[B, n, 1024]
        h = h.squeeze(0)  # [n, 1024]

        rad_h = kwargs['data_rad'].float() #[B, n, 1024]
        rad_h = rad_h.squeeze(0) #[B, n, 1024]

        Crad_h = kwargs['data_Crad'].float() #[B, n, 396]
        Crad_h = Crad_h.squeeze(0) #[B, n, 1024]

        print(rad_h.shape)

        #---->Attention
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        return A

    def captum(self, **kwargs):

        h = kwargs['data'].float() #[B, n, 1024]
        rad_h = kwargs['data_rad'].float() #[B, n, 1024]
        rad_h = rad_h.squeeze(0) #[B, n, 1024]
        h = h.squeeze(0) #[n, 1024]

        #---->Attention
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        h = torch.mm(A, h)
        rad_h = self.fc_rad(rad_h)

        h = torch.add(h, rad_h)

        #---->predict output
        logits = self.classifiers(h) #[B, n_classes]

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        risk = -torch.sum(S, dim=1)
        # print(results_dict)
        return risk
