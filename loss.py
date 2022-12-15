import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def l2norm(x):

	norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
	return torch.div(x, norm)
 
def l1_norm(X, dim, eps=1e-8):

	norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
	X = torch.div(X, norm)
	return X


def l2_norm(X, dim=-1, eps=1e-8):

	norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
	X = torch.div(X, norm)
	return X

        
def compute_l2(x1, x2):
	l2_loss = torch.nn.MSELoss(reduction='mean')
	return l2_loss(10 * x1, 10 * x2)
 
def mutual_learning2(query1, target1, query2, target2):
    query1 = F.normalize(query1, p=2, dim=-1)
    query2 = F.normalize(query2, p=2, dim=-1)
    target1 = F.normalize(target1, p=2, dim=-1)
    target2 = F.normalize(target2, p=2, dim=-1)
    x1 = 10.0 * torch.bmm(query1, target1.repeat(query1.size(0), 1, 1).permute(0, 2, 1))
    x2 = 10.0 * torch.bmm(query2, target2.repeat(query2.size(0), 1, 1).permute(0, 2, 1))

    log_soft_x1 = F.log_softmax(x1, dim=-1)
    soft_x2 = F.softmax(torch.autograd.Variable(x2), dim=-1)
    kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')
    return kl
    

class LossModule(nn.Module):

	def __init__(self, opt):
		super(LossModule, self).__init__()

	def forward(self, scores):




		GT_labels = torch.arange(scores.shape[0]).long()
		GT_labels = torch.autograd.Variable(GT_labels)
		if torch.cuda.is_available():
			GT_labels = GT_labels.cuda()


		loss = F.cross_entropy(scores, GT_labels, reduction = 'mean')

		return loss



class ContrastiveLoss_rec(nn.Module):


	def __init__(self, opt, margin=0):
		super(ContrastiveLoss_rec, self).__init__()
		self.opt = opt
		self.embed_dim = opt.embed_dim
		self.margin = margin
		self.w = nn.Linear(opt.embed_dim, opt.embed_dim)
		

	def forward(self, A_is_t, A_em_t, m, tr_m):
		
		batch_size = m.size(0)
		
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.w.to(device=device)
		
		scores_is = (m.view(batch_size, 1, self.embed_dim) * (self.w(A_is_t))).sum(-1)
		scores_em = (m.view(batch_size, 1, self.embed_dim) * (self.w(A_em_t))).sum(-1)
		scores_is_trm = (tr_m.view(batch_size, 1, self.embed_dim) * (self.w(A_is_t))).sum(-1)
		scores_em_trm = (tr_m.view(batch_size, 1, self.embed_dim) * (self.w(A_em_t))).sum(-1)
		diagonal_is = scores_is.diag().view(batch_size, 1)
		diagonal_em = scores_em.diag().view(batch_size, 1)
		diagonal_is_trm = scores_is_trm.diag().view(batch_size, 1)
		diagonal_em_trm = scores_em_trm.diag().view(batch_size, 1)
		diagonal_is_all = (0.4 * diagonal_is + 0.6 * diagonal_is_trm)  
		diagonal_em_all = (0.6 * diagonal_em_trm + 0.4 * diagonal_em)
   
		cost_s = (self.margin + diagonal_is_all - diagonal_em_all).clamp(min=0)
		
		return cost_s.sum(0)

class ContrastiveLoss_rec2(nn.Module):

	def __init__(self, opt, margin=0):
		super(ContrastiveLoss_rec2, self).__init__()
		self.opt = opt
		self.embed_dim = opt.embed_dim
		self.margin = margin
		self.w = nn.Linear(opt.embed_dim, opt.embed_dim)
		self.saf = AttentionFiltration(opt.embed_dim)


	def forward(self, A_is_t, A_em_t, m, tr_m):
		
		batch_size = m.size(0)
		
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.w.to(device=device)
		self.saf.to(device=device)
	
		scores_is = (m.view(batch_size, 1, self.embed_dim) * (self.w(self.saf(A_is_t)))).sum(-1)
		scores_em = (m.view(batch_size, 1, self.embed_dim) * (self.w(self.saf(A_em_t)))).sum(-1)
		scores_is_trm = (tr_m.view(batch_size, 1, self.embed_dim) * (self.w(self.saf(A_is_t)))).sum(-1)
		scores_em_trm = (tr_m.view(batch_size, 1, self.embed_dim) * (self.w(self.saf(A_em_t)))).sum(-1)
		diagonal_is = scores_is.diag().view(batch_size, 1)
		diagonal_em = scores_em.diag().view(batch_size, 1)
		diagonal_is_trm = scores_is_trm.diag().view(batch_size, 1)
		diagonal_em_trm = scores_em_trm.diag().view(batch_size, 1)
		diagonal_is_all = (0.4 * diagonal_is + 0.6 * diagonal_is_trm)  
		diagonal_em_all = (0.6 * diagonal_em_trm + 0.4 * diagonal_em)
		

		
		cost_s = (self.margin + diagonal_is_all - diagonal_em_all).clamp(min=0)
		
		return cost_s.sum(0)
class ContrastiveLoss_rec3(nn.Module):
	
	def __init__(self, opt, margin=0):
		super(ContrastiveLoss_rec3, self).__init__()
		self.opt = opt
		self.embed_dim = opt.embed_dim
		self.margin = margin
		self.w = nn.Linear(opt.embed_dim, opt.embed_dim)
		self.w2 = nn.Linear(opt.embed_dim, opt.embed_dim)
	

	def forward(self, A_is_t, A_em_t, m, tr_m):
		
		batch_size = m.size(0)
		
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.w.to(device=device)
		self.w2.to(device=device)
		scores_is = ((self.w(m)).view(batch_size, 1, self.embed_dim) * (A_is_t)).sum(-1)
		
		scores_em = ((self.w(m)).view(batch_size, 1, self.embed_dim) * (A_em_t)).sum(-1)
		scores_is_trm = ((self.w(tr_m)).view(batch_size, 1, self.embed_dim) * (A_is_t)).sum(-1)
		scores_em_trm = ((self.w(tr_m)).view(batch_size, 1, self.embed_dim) * (A_em_t)).sum(-1)
		diagonal_is = scores_is.diag().view(batch_size, 1)
		diagonal_em = scores_em.diag().view(batch_size, 1)
		diagonal_is_trm = scores_is_trm.diag().view(batch_size, 1)
		diagonal_em_trm = scores_em_trm.diag().view(batch_size, 1)
		diagonal_is_all = (0.2 * diagonal_is + 0.8 * diagonal_is_trm)  
		diagonal_em_all = (0.8 * diagonal_em_trm + 0.2 * diagonal_em)
		
		cost_s = (self.margin + diagonal_is_all - diagonal_em_all).clamp(min=0)
		
		return cost_s.sum(0)

