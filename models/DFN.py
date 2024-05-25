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


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 512
        self.ps_1 = model_base.extractor().to(self.device)
        self.ps_2 = model_base.extractor().to(self.device)
        self.pt = model_base.extractor().to(self.device)
        self.sh = model_base.extractor().to(self.device)
        self.C = model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
                                          dropout=args.dropout, last=None).to(self.device)
        self.de = model_base.Shared_feature_decoder().to(self.device)
        self.domain_discri = model_base.ClassifierMLP(input_size=output_size, output_size=1,
                        dropout=args.dropout, last = 'sigmoid').to(self.device)
        grl = utils.GradientReverseLayer() 
        self.domain_adv = utils.dfn_da(self.domain_discri, grl=grl)
        self._init_data(concat_src=True)
    def save_model(self):
        torch.save({
            'ps_1': self.ps_1.state_dict(),
            'ps_2': self.ps_2.state_dict(),
            'pt': self.pt.state_dict(),
            'sh': self.sh.state_dict(),
            'de': self.de.state_dict(),
            'domain_discri': self.domain_discri.state_dict(),
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
        self.G.load_state_dict(ckpt['G'])
        self.C.load_state_dict(ckpt['C'])

    def train(self):
        args = self.args
        self._init_data()
        src = args.source_name
        self.optimizer = self._get_optimizer([self.ps_1, self.ps_2, self.pt, self.sh, self.de, self.domain_discri, self.C])
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
            self.ps_1.train()
            self.ps_2.train()
            self.pt.train()
            self.sh.train()
            self.de.train()
            self.C.train()
            self.domain_discri.train()
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
                # private encoder forward
                # 근데 사실 factorization 이 불안전해서 사용은 안함
                fs_1 = self.ps_1(source_data[0])
                fs_2 = self.ps_2(source_data[1])
                ft = self.pt(target_data)
                fs = torch.cat((fs_1, fs_2, ft), dim = 0)    
                
                # shared encoder forward
                f_sh = self.sh(torch.cat((source_data[0], source_data[1], target_data), dim = 0))
                
                # reconstruction with decoder
                recon = self.de(f_sh+ fs)

                # domain 별 분리
                f_sh_s1, f_sh_s2, f_sh_t = f_sh.chunk(3, dim = 0)
                raw_s1, raw_s2, raw_t = recon.chunk(3, dim = 0)

                # shared encoder 에서 나온 feature 로 class prediction
                cs_1 = self.C(f_sh_s1)
                cs_2 = self.C(f_sh_s2)
                ct = self.C(f_sh_t)
                y_s = [cs_1, cs_2]
                loss_d, ds_1, ds_2, dt, d_acc = self.domain_adv(f_sh_s1, f_sh_s2, f_sh_t) 
                # source class loss
                loss_t = (F.cross_entropy(cs_1, source_labels[0]) + \
                    F.cross_entropy(cs_2, source_labels[1]))
                # factorization loss - 아직 불안정해서 사용안함
                loss_f = 1/3 * (torch.norm(torch.inner(fs_1, f_sh_s1)).mean(dim = 0) + \
                                torch.norm(torch.inner(fs_2, f_sh_s2)).mean(dim = 0) + \
                                torch.norm(torch.inner(ft, f_sh_t)).mean(dim = 0))
                # reconstruction loss
                loss_r = 1/3 * (F.mse_loss(raw_s1, source_data[0]) + \
                                F.mse_loss(raw_s2, source_data[1]) + \
                                F.mse_loss(raw_t, target_data))
                # iet loss 
                
                loss_i = 1/3 * ((1 + self.entropy(ds_1, domain_label = True)) * self.entropy(cs_1) + \
                                (1 + self.entropy(ds_2, domain_label = True)) * self.entropy(cs_2) + \
                                (1 + self.entropy(dt, domain_label = True)) * self.entropy(ct))
                #loss = loss_t + 0.3 * loss_d + 0.01 * loss_i
                
    

                loss = loss_t + (self.args.adv * loss_d) + (self.args.iet * loss_i) + (self.args.fac * loss_f) + self.args.recon * loss_r



                for j in range(self.num_source):
                    epoch_acc['Source Data %d'%j] += utils.get_accuracy(y_s[j], source_labels[j])
                epoch_loss['Source Classifier'] += loss_t
                epoch_loss['adv loss'] += loss_d
                epoch_loss['facto loss'] += loss_f
                epoch_loss['iet loss'] += loss_i
                epoch_loss['recon loss'] += loss_r
                epoch_loss['domain acc'] += d_acc               

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
        self.sh.eval()
        self.C.eval()
        acc = 0.0
        iters = iter(self.dataloaders['val'])
        num_iter = len(iters)
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels, _ = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                pred = self.C(self.sh(target_data))
                acc += utils.get_accuracy(pred, target_labels)
        acc /= num_iter
        logging.info('Val-Acc Target Data: {:.4f}'.format(acc))
        return acc