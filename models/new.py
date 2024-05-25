import random
import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import torch.nn as nn
import wandb
import utils
import model_base
from train_utils import InitTrain


class CorrelationAlignmentLoss(nn.Module):

    def __init__(self):
        super(CorrelationAlignmentLoss, self).__init__()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        mean_s = f_s.mean(0, keepdim=True)
        mean_t = f_t.mean(0, keepdim=True)
        cent_s = f_s - mean_s
        cent_t = f_t - mean_t
        cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1)

        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()

        return mean_diff + cov_diff



class load_disc(nn.Module):

    def __init__(self,
                 in_channel=64, kernel_size=3, stride=1, padding=1,
                 mp_kernel_size=2, mp_stride=2, dropout=0.):
        super(load_disc, self).__init__()


        self.conv_1 = nn.Conv1d(in_channel, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn_1 = nn.BatchNorm1d(64)
        self.relu_1 = nn.ReLU(inplace=True)
        self.pool_1 = nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride)

        self.conv_2 = nn.Conv1d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn_2 = nn.BatchNorm1d(128)
        self.relu_2 = nn.ReLU(inplace=True)
        self.pool_2 = nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride)

        self.conv_3 = nn.Conv1d(128, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn_3 = nn.BatchNorm1d(128)
        self.relu_3 = nn.ReLU(inplace=True)
        self.pool_3 = nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride)
        
        self.fl = nn.Flatten()


    def forward(self, tar, x=None, y=None):
        
        out = self.conv_1(tar)
        out = self.bn_1(out)
        out = self.relu_1(out)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu_2(out)
        out = self.pool_2(out)

        out = self.conv_3(out)
        out = self.bn_3(out)
        out = self.relu_3(out)
        # out = self.pool_3(out)
        
        out = self.fl(out)

        return out

class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 512
        self.d_l = load_disc().to(self.device)
        self.d_m = model_base.extractor().to(self.device)
        self.d_m_c = model_base.ClassifierMLP(input_size=output_size, output_size=1,
                                          dropout=args.dropout, last='sigmoid').to(self.device)
        self.e_c = model_base.extractor().to(self.device)
        self.C = model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
                                          dropout=args.dropout, last='sm').to(self.device)
        self.load_discri = model_base.ClassifierMLP(input_size=output_size, output_size=4,
                        dropout=args.dropout, last = 'sm').to(self.device)
        grl = utils.GradientReverseLayer() 
        self.domain_adv = utils.new_da(self.load_discri, grl=grl)
        self.coral = CorrelationAlignmentLoss()
        self._init_data()
    def save_model(self):
        torch.save({
            'd_l': self.d_l.state_dict(),
            'd_m': self.d_m.state_dict(),
            'd_m_c': self.d_m.state_dict(),
            'e_c': self.e_c.state_dict(),
            'load_discri': self.load_discri.state_dict(),
            'C': self.C.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def entropy(self, tensor, domain_label = False):
        if domain_label:
            entropy = -torch.mean(torch.sum(tensor *(torch.log(tensor + 1e-5)), 1))
            entropy += -torch.mean(torch.sum((1-tensor) *(torch.log((1-tensor) + 1e-5)), 1))
        else:
            tensor = F.softmax(tensor, dim=-1)
            entropy = -torch.mean(torch.sum(tensor *(torch.log(tensor + 1e-5)), 1))
        return entropy

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.e_c.load_state_dict(ckpt['e_c'])
        self.C.load_state_dict(ckpt['C'])

    def train(self):
        args = self.args
        src = args.source_name
        self.optimizer = self._get_optimizer([self.d_l, self.d_m, self.load_discri, self.d_m_c, self.e_c, self.C])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        
        best_acc = 0.0
        best_epoch = 0
   
        for epoch in range(1, args.max_epoch+1):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
   
            # Each epoch has a training and val phase
            epoch_acc = defaultdict(float)
   
            # Set model to train mode or evaluate mode
            self.d_l.train()
            self.d_m.train()
            self.d_m_c.train()
            self.e_c.train()
            self.C.train()
            self.load_discri.train()
            epoch_loss = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
            num_iter = len(self.dataloaders['train'])
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels = utils.get_next_batch(self.dataloaders,
                						 self.iters, 'train', self.device)                 
                source_data, source_labels = [], []
                if args.train_mode == 'source_combine':
                    source_data_item, source_labels_item = utils.get_next_batch(self.dataloaders,
                						     self.iters, src, self.device)
                    source_data.append(source_data_item)
                    source_labels.append(source_labels_item)
                else:
                    for idx in range(self.num_source):
                        source_data_item, source_labels_item = utils.get_next_batch(self.dataloaders,
                    						     self.iters, src[idx], self.device)
                        source_data.append(source_data_item)
                        source_labels.append(source_labels_item)
                # forward
                self.optimizer.zero_grad()

                input = torch.cat((source_data[0], source_data[1], source_data[2], target_data), axis = 0)
                
                # machine discriminator
                f_dm, h_dm = self.d_m(input)
                pred_dm = self.d_m_c(f_dm)
                p_dm_1, p_dm_2, p_dm_3, t_p = pred_dm.chunk(4, dim = 0)
                s_p = torch.cat((p_dm_1, p_dm_2, p_dm_3), dim = 0)
                
                # machine domain label
                s_mch = torch.ones(s_p.size(0), 1).to(self.device)
                t_mch = torch.zeros(t_p.size(0), 1).to(self.device)
                
                # load discriminator
                f_dl = self.d_l(h_dm)
                f_d1_1, f_d1_2, f_d1_3, f_d1_4 = f_dl.chunk(4, dim = 0)
                               
                inputs = torch.cat((source_data[0], source_data[1], source_data[2], target_data), axis = 0)
                
                # class classifier
                f_ec, _ = self.e_c(inputs)
                f_s_1, f_s_2, f_s_3, f_t = f_ec.chunk(4, dim = 0)
                f_s = torch.cat((f_s_1, f_s_2, f_s_3), axis= 0)
                pred = self.C(f_ec)
                pred_s1, pred_s2, pred_s3, pred_t = pred.chunk(4, dim = 0)
                
                # source class loss
                loss_t = F.cross_entropy(pred_s1, source_labels[0]) + \
                         F.cross_entropy(pred_s2, source_labels[1]) + \
                         F.cross_entropy(pred_s3, source_labels[2])
                
                # distribution align loss
                loss_dist = self.coral(f_s, f_t)

                # machine domain loss
                loss_dm = F.binary_cross_entropy(s_p, s_mch) + F.binary_cross_entropy(t_p, t_mch)

                # factorization loss - 아직 불안정해서 사용안함
                loss_f = torch.abs(torch.inner(f_ec, f_dm)).mean(dim = 0).mean()

                # load domain adversarial loss
                loss_d = self.domain_adv(f_d1_1, f_d1_2, f_d1_3, f_d1_4)
                
                # perserving loss - factorization loss 의 trivial soultion 을 방지
                loss_n = torch.norm(torch.diagonal(torch.inner(f_ec, f_ec))).mean(dim = 0) + \
                         torch.norm(torch.diagonal(torch.inner(f_dm, f_dm))).mean(dim = 0)
                
                loss = loss_t + 30 * loss_f + loss_dm + loss_d + 1000 * loss_dist
                y_s = [pred_s1, pred_s2, pred_s3]
                for j in range(self.num_source):
                    epoch_acc['Source Data %d'%j] += utils.get_accuracy(y_s[j], source_labels[j])
                epoch_loss['Source Classifier'] += loss_t
                epoch_loss['adv loss'] += loss_d
                epoch_loss['facto loss'] += loss_f
                epoch_loss['machine domain loss'] += loss_dm
                epoch_loss['info preserve'] += loss_n
                epoch_loss['distribution gap'] += loss_dist

                # backward
                loss.backward()
                self.optimizer.step()
                
           # Print the train and val information via each epoch
            for key in epoch_acc.keys():
                avg_acc = epoch_acc[key] / num_iter
                logging.info('Train-Acc {}: {:.4f}'.format(key, avg_acc))
                wandb.log({f'Train-Acc {key}': avg_acc}, commit=False)  # Log to wandb
            for key in epoch_loss.keys():
                logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/num_iter))
                # wandb.log({f'Train-loss {key}': epoch_loss/num_iter}, commit=False)  # Log to wandb
                
                
            #log the best model according to the val accuracy
            new_acc = self.test()
            
            last_acc_formatted = f"{new_acc:.3f}"
            wandb.log({"last_target_acc": float(last_acc_formatted)})
            
            
            if new_acc >= best_acc:
                best_acc = new_acc
                best_epoch = epoch
            logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
            
            best_acc_formatted = f"{best_acc:.3f}"
            wandb.log({"best_target_acc": float(best_acc_formatted)})
             
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
             
            if self.args.tsne:
                self.epoch = epoch
                if epoch == 1 or epoch % 5 == 0:
                    self.test_tsne()
                
     
        acc=self.test()
        acc_formatted = f"{acc:.3f}"
        wandb.log({"target_acc": float(acc_formatted)})    

    def test(self):
        self.e_c.eval()
        self.C.eval()
        acc = 0.0
        iters = iter(self.dataloaders['val'])
        num_iter = len(iters)
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels, _ = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                f, _ = self.e_c(target_data)
                pred = self.C(f)
                acc += utils.get_accuracy(pred, target_labels)
        acc /= num_iter
        logging.info('Val-Acc Target Data: {:.4f}'.format(acc))
        return acc