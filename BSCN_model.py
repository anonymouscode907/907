
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BaseModel
from utils import l2norm
from math import sqrt
import numpy as np
class L2Module(nn.Module):

	def __init__(self):
		super(L2Module, self).__init__()

	def forward(self, x):
		x = l2norm(x)
		return x

def cosine_similarity(x1, x2, dim=1, eps=1e-8):

	w12 = torch.sum(x1 * x2, dim)
	w1 = torch.norm(x1, 2, dim)
	w2 = torch.norm(x2, 2, dim)
	return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def contran_attn(A_is, dim=1):
	A_mean = A_is.mean(dim=dim)

	A_is_ctr = A_mean.unsqueeze(dim=1) - A_is + A_mean.unsqueeze(dim=1)
	return A_is_ctr


class AttentionMechanism(nn.Module):


	def __init__(self, opt):
		super(AttentionMechanism, self).__init__()

		self.embed_dim = opt.embed_dim
		input_dim = self.embed_dim

		self.attention = nn.Sequential(
			nn.Linear(input_dim, self.embed_dim),
			nn.ReLU(),
			nn.Linear(self.embed_dim, self.embed_dim),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		return self.attention(x)




def l1_norm(X, dim, eps=1e-8):

	norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
	X = torch.div(X, norm)
	return X


def l2_norm(X, dim=-1, eps=1e-8):

	norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
	X = torch.div(X, norm)
	return X




class BSCN(BaseModel):
	def __init__(self, word2idx, opt):
		super(BSCN, self).__init__(word2idx, opt)


		self.Transform_m = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), L2Module())

		self.Transform_attention = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim))

		self.Attention_EM = AttentionMechanism(opt)
		self.Attention_IS = AttentionMechanism(opt)
		
		self.Attention_EM_14 = AttentionMechanism(opt)
		self.Attention_IS_14 = AttentionMechanism(opt)
		self.Attention_EM_28 = AttentionMechanism(opt)
		self.Attention_IS_28 = AttentionMechanism(opt)
		


		self.model_version = opt.model_version
		if self.model_version == "BSCN":
			self.compute_score = self.compute_score_artemis
			self.compute_score_broadcast = self.compute_score_broadcast_artemis


		self.gradcam = opt.gradcam
		self.hold_results = dict()




	def apply_attention(self, a, x):
		return l2norm(a * x)
	
	def compute_score_artemis(self, r, r_14, r_28, m, t, t_14, t_28, store_intermediary=False):
		EM, EM_14, EM_28 = self.compute_score_EM(r, r_14, r_28, m, t, t_14, t_28, store_intermediary)

		IS, IS_14, IS_28 = self.compute_score_IS(r, r_14, r_28, m, t, t_14, t_28, store_intermediary)
		if store_intermediary:
			self.hold_results["EM"] = EM
			self.hold_results["IS"] = IS

		return EM + IS, EM_14 + IS_14, EM_28 + IS_28
	def compute_score_broadcast_artemis(self, r, r_14, r_28, m, t, t_14, t_28):
		EM, EM_14, EM_28, A_EM_all_t, A_EM_all_t_14, A_EM_all_t_28, Tr_m = self.compute_score_broadcast_EM(r, r_14, r_28, m, t, t_14, t_28)
		IS, IS_14, IS_28, A_IS_all_t, A_IS_all_t_14, A_IS_all_t_28, A_IS_all_rev_t, A_IS_all_t_rev_14, A_IS_all_t_rev_28 = self.compute_score_broadcast_IS(r, r_14, r_28, m, t, t_14, t_28)
		return EM + IS, EM_14 + IS_14, EM_28 + IS_28, A_IS_all_t, A_IS_all_t_14, A_IS_all_t_28, A_EM_all_t, A_EM_all_t_14, A_EM_all_t_28, A_IS_all_rev_t, A_IS_all_t_rev_14, A_IS_all_t_rev_28, m, Tr_m


	def compute_score_EM(self, r, r_14, r_28, m, t, t_14, t_28, store_intermediary=False):
		Tr_m = self.Transform_m(m)

		A_EM = self.Attention_EM(m)
		A_EM_14 = self.Attention_EM_14(m)
		A_EM_28 = self.Attention_EM_28(m)
		A_EM_t = self.apply_attention(A_EM, t)
		A_EM_t_14 = self.apply_attention(A_EM_14, t_14)
		A_EM_t_28 = self.apply_attention(A_EM_28, t_28)
		if store_intermediary:
			self.hold_results["Tr_m"] = Tr_m
			self.hold_results["A_EM_t"] = A_EM_t
		return (Tr_m * A_EM_t).sum(-1), (Tr_m * A_EM_t_14).sum(-1), (Tr_m * A_EM_t_28).sum(-1)
	def compute_score_broadcast_EM(self, r, r_14, r_28, m, t, t_14, t_28):
		batch_size = r.size(0)

		A_EM = self.Attention_EM(m)
		A_EM_14 = self.Attention_EM_14(m)
		A_EM_28 = self.Attention_EM_28(m)

		Tr_m = self.Transform_m(m)
		A_EM_all_t = self.apply_attention(A_EM.view(batch_size, 1, self.embed_dim), t.view(1, batch_size, self.embed_dim))
		A_EM_all_t_14 = self.apply_attention(A_EM_14.view(batch_size, 1, self.embed_dim), t_14.view(1, batch_size, self.embed_dim))
		A_EM_all_t_28 = self.apply_attention(A_EM_28.view(batch_size, 1, self.embed_dim), t_28.view(1, batch_size, self.embed_dim))
		EM_score = (Tr_m.view(batch_size, 1, self.embed_dim) * A_EM_all_t).sum(-1)
		EM_score_14 = (Tr_m.view(batch_size, 1, self.embed_dim) * A_EM_all_t_14).sum(-1)
		EM_score_28 = (Tr_m.view(batch_size, 1, self.embed_dim) * A_EM_all_t_28).sum(-1)

		return EM_score, EM_score_14, EM_score_28, A_EM_all_t, A_EM_all_t_14, A_EM_all_t_28, Tr_m





	def compute_score_IS(self, r, r_14, r_28, m, t, t_14, t_28, store_intermediary=False):

		A_IS = self.Attention_IS(m)
		A_IS_14 = self.Attention_IS_14(m)
		A_IS_28 = self.Attention_IS_28(m)

		A_IS_r = self.apply_attention(A_IS, r)
		A_IS_t = self.apply_attention(A_IS, t)
		A_IS_r_14 = self.apply_attention(A_IS_14, r_14)
		A_IS_t_14 = self.apply_attention(A_IS_14, t_14)
		A_IS_r_28 = self.apply_attention(A_IS_28, r_28)
		A_IS_t_28 = self.apply_attention(A_IS_28, t_28)


		if store_intermediary:
			self.hold_results["A_IS_r"] = A_IS_r
			self.hold_results["A_IS_t"] = A_IS_t
		return (A_IS_r * A_IS_t).sum(-1), (A_IS_r_14 * A_IS_t_14).sum(-1), (A_IS_r_28 * A_IS_t_28).sum(-1)


	def compute_score_broadcast_IS(self, r, r_14, r_28, m, t, t_14, t_28):
		batch_size = r.size(0)

		A_IS = self.Attention_IS(m)
		A_IS_14 = self.Attention_IS_14(m)
		A_IS_28 = self.Attention_IS_28(m)

		A_IS_r = self.apply_attention(A_IS, r)

		A_IS_all_t = self.apply_attention(A_IS.view(batch_size, 1, self.embed_dim), t.view(1, batch_size, self.embed_dim))
		A_IS_all_rev_t = self.apply_attention(contran_attn(A_IS).view(batch_size, 1, self.embed_dim), t.view(1, batch_size, self.embed_dim))

		A_IS_r_14 = self.apply_attention(A_IS_14, r_14)

		A_IS_all_t_14 = self.apply_attention(A_IS_14.view(batch_size, 1, self.embed_dim), t_14.view(1, batch_size, self.embed_dim))
		A_IS_all_t_rev_14 = self.apply_attention(contran_attn(A_IS).view(batch_size, 1, self.embed_dim), t_14.view(1, batch_size, self.embed_dim))

		A_IS_r_28 = self.apply_attention(A_IS_28, r_28)

		A_IS_all_t_28 = self.apply_attention(A_IS_28.view(batch_size, 1, self.embed_dim), t_28.view(1, batch_size, self.embed_dim))
		A_IS_all_t_rev_28 = self.apply_attention(contran_attn(A_IS).view(batch_size, 1, self.embed_dim), t_28.view(1, batch_size, self.embed_dim))


		IS_score = (A_IS_r.view(batch_size, 1, self.embed_dim) * A_IS_all_t).sum(-1)
		IS_score_14 = (A_IS_r_14.view(batch_size, 1, self.embed_dim) * A_IS_all_t_14).sum(-1)
		IS_score_28 = (A_IS_r_28.view(batch_size, 1, self.embed_dim) * A_IS_all_t_28).sum(-1)

		return IS_score, IS_score_14, IS_score_28, A_IS_all_t, A_IS_all_t_14, A_IS_all_t_28, A_IS_all_rev_t, A_IS_all_t_rev_14, A_IS_all_t_rev_28

	def compute_score_arithmetic(self, r, m, t, store_intermediary=False):
		return (l2norm(r + m) * t).sum(-1)
	def compute_score_broadcast_arithmetic(self, r, m, t):
		return (l2norm(r + m)).mm(t.t())

	def compute_score_crossmodal(self, r, m, t, store_intermediary=False):
		return (m * t).sum(-1)
	def compute_score_broadcast_crossmodal(self, r, m, t):
		return m.mm(t.t())

	def compute_score_visualsearch(self, r, r_14, r_28, m, t, t_14, t_28, store_intermediary=False):
		return (r * t).sum(-1) + (r_14 * t_14).sum(-1) + (r_28 * t_28).sum(-1)
	def compute_score_broadcast_visualsearch(self, r, r_14, r_28, m, t, t_14, t_28):
		return r.mm(t.t()) + r_14.mm(t_14.t()) + r_28.mm(t_28.t())




	def forward_save_intermediary(self, images_src, images_trg, sentences, lengths):


		self.hold_results.clear()


		r, r_14, r_28 = self.get_image_embedding(images_src)
		if self.gradcam:
			self.hold_results["r_activation"] = self.img_enc.get_activation()
		t, t_14, t_28 = self.get_image_embedding(images_trg)
		if self.gradcam:
			self.hold_results["t_activation"] = self.img_enc.get_activation()
		m = self.get_txt_embedding(sentences, lengths)

		return self.compute_score(r, r_14, r_28, m, t, t_14, t_28, store_intermediary=True)