import warnings
warnings.filterwarnings("ignore")
import torch, faiss
import numpy as np

def loss_select(loss, opt, to_optim):
    if loss == 'recallatk':
        loss_params  = {'anneal':opt.sigmoid_temperature, 'batch_size':opt.bs, "num_id":int(opt.bs / opt.samples_per_class), 'feat_dims':opt.embed_dim, 'k_vals': opt.k_vals_train, 'k_temperatures': opt.k_temperatures, 'mixup':opt.mixup}
        criterion = RecallatK(**loss_params)
    else:
        raise Exception('Loss {} not available!'.format(loss))

    return criterion, to_optim

def sigmoid(tensor, temp=1.0):
    exponent = -tensor / temp
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

class RecallatK(torch.nn.Module):
    def __init__(self, anneal, batch_size, num_id, feat_dims, k_vals, k_temperatures, mixup):
        super(RecallatK, self).__init__()
        assert(batch_size%num_id==0)
        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims
        self.k_vals = [min(batch_size, k) for k in k_vals]
        self.k_temperatures = k_temperatures
        self.mixup = mixup
        self.samples_per_class = int(batch_size/num_id)

    def forward(self, preds, q_id):
        batch_size = preds.shape[0]
        num_id = self.num_id
        anneal = self.anneal
        feat_dims = self.feat_dims
        k_vals = self.k_vals
        k_temperatures = self.k_temperatures
        samples_per_class = int(batch_size/num_id)
        norm_vals = torch.Tensor([min(k, (samples_per_class-1)) for k in k_vals]).cuda()
        group_num = int(q_id/samples_per_class)
        q_id_ = group_num*samples_per_class

        sim_all = (preds[q_id]*preds).sum(1)
        sim_all_g = sim_all.view(num_id, int(batch_size/num_id))
        sim_diff_all = sim_all.unsqueeze(-1) - sim_all_g[group_num, :].unsqueeze(0).repeat(batch_size,1)
        sim_sg = sigmoid(sim_diff_all, temp=anneal)
        for i in range(samples_per_class): sim_sg[group_num*samples_per_class+i,i] = 0.
        sim_all_rk = (1.0 + torch.sum(sim_sg, dim=0)).unsqueeze(dim=0)

        sim_all_rk[:, q_id%samples_per_class] = 0.
        sim_all_rk = sim_all_rk.unsqueeze(dim=-1).repeat(1,1,len(k_vals))
        k_vals = torch.Tensor(k_vals).cuda()
        k_vals = k_vals.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1, samples_per_class, 1)
        sim_all_rk = k_vals - sim_all_rk
        for given_k in range(0, len(self.k_vals)):
            sim_all_rk[:,:,given_k] = sigmoid(sim_all_rk[:,:,given_k], temp=float(k_temperatures[given_k]))

        sim_all_rk[:,q_id%samples_per_class,:] = 0.
        k_vals_loss = torch.Tensor(self.k_vals).cuda()
        k_vals_loss = k_vals_loss.unsqueeze(dim=0)
        recall = torch.sum(sim_all_rk, dim=1)
        recall = torch.minimum(recall, k_vals_loss)
        recall = torch.sum(recall, dim=0)
        recall = torch.div(recall, norm_vals)
        recall = torch.sum(recall)/len(self.k_vals)
        return (1.-recall)/batch_size
