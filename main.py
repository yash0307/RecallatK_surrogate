import warnings
warnings.filterwarnings("ignore")
import os, numpy as np, argparse, random, matplotlib, datetime
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from pathlib import Path
matplotlib.use('agg')
from tqdm import tqdm
import auxiliaries as aux
import datasets as data
import netlib as netlib
import losses as losses
import evaluate as eval
import time
import copy
import pdb
from tensorboardX import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      default='Inaturalist',   type=str, help='Dataset to use.', choices=['Inaturalist','vehicle_id', 'sop', 'cars196'])
parser.add_argument('--lr',                default=0.00001,  type=float, help='Learning Rate for network parameters.')
parser.add_argument('--fc_lr_mul',         default=5,        type=float, help='OPTIONAL: Multiply the embedding layer learning rate by this value. If set to 0, the embedding layer shares the same learning rate.')
parser.add_argument('--n_epochs',          default=400,       type=int,   help='Number of training epochs.')
parser.add_argument('--kernels',           default=16,        type=int,   help='Number of workers for pytorch dataloader.')
parser.add_argument('--bs',                default=112 ,     type=int,   help='Mini-Batchsize to use.')
parser.add_argument('--bs_base',                default=200 ,     type=int,   help='Mini-Batchsize to use for evaluation.')
parser.add_argument('--samples_per_class', default=4,        type=int,   help='Number of samples in one class drawn before choosing the next class')
parser.add_argument('--seed',              default=1,        type=int,   help='Random seed for reproducibility.')
parser.add_argument('--scheduler',         default='step',   type=str,   help='Type of learning rate scheduling. Currently: step & exp.')
parser.add_argument('--gamma',             default=0.3,      type=float, help='Learning rate reduction after tau epochs.')
parser.add_argument('--decay',             default=0.0004,   type=float, help='Weight decay for optimizer.')
parser.add_argument('--tau',               default= [200,300],nargs='+',type=int,help='Stepsize(s) before reducing learning rate.')
parser.add_argument('--infrequent_eval', default=1,type=int, help='only compute evaluation metrics every 10 epochs')
parser.add_argument('--opt', default = 'adam',help='adam or adamW')
parser.add_argument('--loss',         default='recallatk', type=str)
parser.add_argument('--sigmoid_temperature', default=0.01, type=float, help='tau_{2}, the temperature applied on the difference of similarity values')
parser.add_argument('--k_vals',       nargs='+', default=[1,2,4,8], type=int, help='set of k values to be used for Recall@k surrogate.')
parser.add_argument('--k_vals_train',       nargs='+', default=[1,2,4,8,16], type=int, help='Training recall@k vals.')
parser.add_argument('--k_temperatures',       nargs='+', default=[1,2,4,8,16], type=int, help='tau_{1}, the temperatures applied on the ranks.')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint to load weights from (if empty then ImageNet pre-trained weights are loaded')
parser.add_argument('--embed_dim',    default=512,         type=int,   help='Embedding dimensionality of the network')
parser.add_argument('--arch',         default='resnet50',  type=str,   help='Network backend choice: resnet50, googlenet, BNinception, ViT')
parser.add_argument('--grad_measure',                      action='store_true', help='If added, gradients passed from embedding layer to the last conv-layer are stored in each iteration.')
parser.add_argument('--dist_measure',                      action='store_true', help='If added, the ratio between intra- and interclass distances is stored after each epoch.')
parser.add_argument('--not_pretrained',                    action='store_true', help='If added, the network will be trained WITHOUT ImageNet-pretrained weights.')
parser.add_argument('--gpu',          default=0,           type=int,   help='GPU-id for GPU to use.')
parser.add_argument('--savename',     default='',          type=str,   help='Save folder name if any special information is to be included.')
parser.add_argument('--source_path',  default='',         type=str, help='Path to data')
parser.add_argument('--save_path',    default=os.getcwd()+'/Training_Results', type=str, help='Where to save the checkpoints')

opt = parser.parse_args()
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset

# Training hyper-parameters
opt.k_vals_train = [1, 2, 4, 8, 16]
opt.k_temperatures = [1.0, 1.0, 1.0, 1.0, 1.0]
opt.gamma = 0.3
opt.samples_per_class = 4
opt.fc_lr_mul = 1.

if opt.dataset== 'Inaturalist':
    opt.k_vals = [1,4,16,32]
    opt.bs = 4000
    opt.n_epochs = 90
    if opt.arch == 'resnet50':
        opt.tau = [40,70]
        opt.bs_base = 200
        opt.lr = 0.0001
        opt.opt = 'adam'
    if opt.arch == 'ViTB32':
        opt.tau = [10,40,70]
        opt.bs_base = 200
        opt.lr = 0.00005
        opt.opt = 'adamW'
    if opt.arch == 'ViTB16':
        opt.tau = [10,40,70]
        opt.bs_base = 100
        opt.lr = 0.00005
        opt.opt = 'adamW'

if opt.dataset=='sop':
    opt.tau = [25,50]
    opt.k_vals = [1,10,100,1000]
    opt.bs = 4000
    opt.n_epochs = 55
    if opt.arch == 'resnet50':
        opt.bs_base = 200
        opt.lr = 0.0002
        opt.opt = 'adam'
    if opt.arch == 'ViTB32':
        opt.bs_base = 200
        opt.lr = 0.00005
        opt.opt = 'adamW'
    if opt.arch == 'ViTB16':
        opt.bs_base = 100
        opt.lr = 0.00005
        opt.opt = 'adamW'

if opt.dataset=='vehicle_id':
    opt.tau = [40,70]
    opt.k_vals = [1,5]
    opt.bs = 4000
    opt.n_epochs = 90
    if opt.arch == 'resnet50':
        opt.bs_base = 200
        opt.lr = 0.0001
        opt.opt = 'adam'
    if opt.arch == 'ViTB32':
        opt.bs_base = 200
        opt.lr = 0.0001
        opt.opt = 'adamW'
    if opt.arch == 'ViTB16':
        opt.bs_base = 100
        opt.lr = 0.00005
        opt.opt = 'adamW'

if opt.dataset == 'cars196':
    opt.k_vals = [1,2,4,8,16]
    opt.bs = 392
    opt.bs_base = 98
    if opt.arch == 'resnet50':
        opt.n_epochs = 170
        opt.tau = [80, 140] 
        opt.lr = 0.0001
        opt.opt = 'adam'
    if opt.arch == 'ViTB32':
        opt.n_epochs = 50
        opt.tau = [20,30,40]
        opt.lr = 0.00003
        opt.opt = 'adamW'
    if opt.arch == 'ViTB16':
        opt.n_epochs = 50
        opt.tau = [20,30,40]
        opt.lr = 0.00001
        opt.opt = 'adamW'

timestamp = datetime.datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
exp_name = aux.args2exp_name(opt)
opt.save_name = f"weights_{exp_name}" +'/'+ timestamp
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)
tensorboard_path = Path(f"logs/logs_{exp_name}") / timestamp

tensorboard_path.parent.mkdir(exist_ok=True, parents=True)
global writer;
writer = SummaryWriter(tensorboard_path)
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu)

opt.device = torch.device('cuda')
model      = netlib.networkselect(opt)

_          = model.to(opt.device)

if 'fc_lr_mul' in vars(opt).keys() and opt.fc_lr_mul!=0:
    if opt.arch == 'resnet50':
        all_but_fc_params = list(filter(lambda x: 'last_linear' not in x[0],model.named_parameters()))
        for ind, param in enumerate(all_but_fc_params):
            all_but_fc_params[ind] = param[1]
        fc_params         = model.model.last_linear.parameters()
        to_optim          = [{'params':all_but_fc_params,'lr':opt.lr,'weight_decay':opt.decay},
                            {'params':fc_params,'lr':opt.lr*opt.fc_lr_mul,'weight_decay':opt.decay}]
    elif opt.arch == 'ViTB16' or opt.arch == 'ViTB32' or opt.arch == 'DeiTB':
        all_but_fc_params = list(filter(lambda x: 'head' not in x[0],model.named_parameters()))
        for ind, param in enumerate(all_but_fc_params):
            all_but_fc_params[ind] = param[1]
        fc_params         = model.model.head.parameters()
        to_optim          = [{'params':all_but_fc_params,'lr':opt.lr,'weight_decay':opt.decay},
                            {'params':fc_params,'lr':opt.lr*opt.fc_lr_mul,'weight_decay':opt.decay}]
else:
    to_optim   = [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]
dataloaders      = data.give_dataloaders(opt.dataset, opt)
opt.num_classes  = len(dataloaders['training'].dataset.avail_classes)
metrics_to_log = aux.metrics_to_examine(opt.dataset, opt.k_vals)
LOG = aux.LOGGER(opt, metrics_to_log, name='Base', start_new=True)

criterion, to_optim = losses.loss_select(opt.loss, opt, to_optim)
_ = criterion.to(opt.device)

if opt.grad_measure:
    grad_measure = eval.GradientMeasure(opt, name='baseline')
if opt.dist_measure:
    distance_measure = eval.DistanceMeasure(dataloaders['evaluation'], opt, name='Train', update_epochs=1)

if opt.opt == 'adamW':
    optimizer    = torch.optim.AdamW(to_optim)
elif opt.opt == 'adam':
    optimizer    = torch.optim.Adam(to_optim)
else:
    raise Exception('unknown optimiser')
if opt.scheduler=='exp':
    scheduler    = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)
elif opt.scheduler=='step':
    scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)
elif opt.scheduler=='none':
    print('No scheduling used!')
else:
    raise Exception('No scheduling option for input: {}'.format(opt.scheduler))

def same_model(model1,model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def train_one_epoch(train_dataloader, model, optimizer, criterion, opt, epoch):
    loss_collect = []
    start = time.time()
    data_iterator = tqdm(train_dataloader, desc='Epoch {} Training...'.format(epoch))
    optimizer.zero_grad()
    for i,(class_labels, input) in enumerate(data_iterator):
        output = torch.zeros((len(input), opt.embed_dim)).to(opt.device)
        for j in range(0, len(input), opt.bs_base):
            input_x = input[j:j+opt.bs_base,:].to(opt.device)
            x = model(input_x)
            output[j:j+opt.bs_base,:] = copy.copy(x)
            del x
            torch.cuda.empty_cache()
        num_samples = output.shape[0]
        output.retain_grad()
        loss = 0.

        for q in range(0, num_samples):
            loss += criterion(output, q)
        loss_collect.append(loss.item())
        loss.backward()
        output_grad = copy.copy(output.grad)
        del loss
        del output
        torch.cuda.empty_cache()

        for j in range(0, len(input), opt.bs_base):
            input_x = input[j:j+opt.bs_base,:].to(opt.device)
            x = model(input_x)
            x.backward(output_grad[j:j+opt.bs_base,:])
        optimizer.step()
        optimizer.zero_grad()
        if opt.grad_measure:
            if opt.arch == 'resnet50':
                grad_measure.include(model.model.last_linear)
            else:
                grad_measure.include(model.model.head)
        if i==len(train_dataloader)-1:
            data_iterator.set_description('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect)))
    LOG.log('train', LOG.metrics_to_log['train'], [epoch, np.round(time.time()-start,4), np.mean(loss_collect)])
    writer.add_scalar('global/training_loss',np.mean(loss_collect),epoch)
    if opt.grad_measure:
        grad_measure.dump(epoch)

print('\n-----\n')
if opt.dataset in ['Inaturalist', 'sop', 'cars196']:
    eval_params = {'dataloader': dataloaders['testing'], 'model': model, 'opt': opt, 'epoch': 0}
elif opt.dataset == 'vehicle_id':
    eval_params = {
        'dataloaders': [dataloaders['testing_set1'], dataloaders['testing_set2'], dataloaders['testing_set3']],
        'model': model, 'opt': opt, 'epoch': 0}
print('epochs -> '+str(opt.n_epochs))

for epoch in range(opt.n_epochs):
    if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))
    _ = model.train()
    train_one_epoch(dataloaders['training'], model, optimizer, criterion, opt, epoch)
    dataloaders['training'].dataset.reshuffle()
    _ = model.eval()
    if opt.dataset in ['Inaturalist', 'sop', 'cars196']:
        eval_params = {'dataloader':dataloaders['testing'], 'model':model, 'opt':opt, 'epoch':epoch}
    elif opt.dataset=='vehicle_id':
        eval_params = {'dataloaders':[dataloaders['testing_set1'], dataloaders['testing_set2'], dataloaders['testing_set3']], 'model':model, 'opt':opt, 'epoch':epoch}
    if opt.infrequent_eval == 1:
        epoch_freq = 5
    else:
        epoch_freq = 1
    if not opt.dataset == 'vehicle_id':
        if epoch%epoch_freq == 0 or epoch == opt.n_epochs - 1:
            results = eval.evaluate(opt.dataset, LOG, save=True, **eval_params)
            writer.add_scalar('global/recall1',results[0][0],epoch+1)
            writer.add_scalar('global/recall2',results[0][1],epoch+1)
            writer.add_scalar('global/recall3',results[0][2],epoch+1)
            writer.add_scalar('global/recall4',results[0][3],epoch+1)
            writer.add_scalar('global/NMI',results[1],epoch+1)
            writer.add_scalar('global/F1',results[2],epoch+1)

    else:
        if epoch%epoch_freq == 0 or epoch == opt.n_epochs - 1:
            results = eval.evaluate(opt.dataset, LOG, save=True, **eval_params)
            writer.add_scalar('global/recall1',results[2],epoch+1)
            writer.add_scalar('global/recall2',results[3],epoch+1)#writer.add_scalar('global/recall3',results[0][2],0)
            writer.add_scalar('global/recall3',results[6],epoch+1)
            writer.add_scalar('global/recall4',results[7],epoch+1)
            writer.add_scalar('global/recall5',results[10],epoch+1)
            writer.add_scalar('global/recall6',results[11],epoch+1)
    if opt.dist_measure:
        distance_measure.measure(model, epoch)
    if opt.scheduler != 'none':
        scheduler.step()
    print('\n-----\n')
