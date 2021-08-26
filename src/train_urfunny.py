import torch
from torch import nn
import sys
from src import models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle

from src.eval_metrics import *


#######################
#
# Construct the model
#
#######################

text_dim, audio_dim,video_dim = 300, 81, 75
def crop(a): return a[:,:,:, 0:text_dim],a[:, :, :, text_dim:(text_dim + audio_dim)], a[:, :, :, (text_dim+audio_dim)::]

def initiate(hyp_params, train_loader, valid_loader, test_loader,test_only=False,verbose=True,onlypunch=False):
    model = getattr(models, 'SPModel')(hyp_params)
    paramnum=sum(p.numel() for p in list(model.parameters()) if p.requires_grad)
    if verbose: print('Param num',paramnum)
    with open('./pre_trained_models/'+hyp_params.name+'_log.txt','w') as f: 
        f.write('Param num: '+str(paramnum)+'\n')
    
    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader,test_only,verbose,onlypunch)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader,test_only=False,verbose=True,onlypunch=False):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = nn.BCEWithLogitsLoss()
    
    scheduler = settings['scheduler']
    
    log=''

    def train(model, optimizer, criterion, verbose=True,onlypunch=False):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (x_c, x_p, eval_attr) in enumerate(train_loader):
            x_p = torch.unsqueeze(x_p, dim=1)
            combined = x_p if onlypunch else torch.cat([x_c, x_p], dim=1) 
            t,a,v = crop(combined)
            bs,segs,lens,_=a.shape
            audio,text,vision=a.reshape(bs,segs*lens,-1),t.reshape(bs,segs*lens,-1),v.reshape(bs,segs*lens,-1)
            
            model.zero_grad()
                
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
            
            batch_size = text.size(0)
            batch_chunk = hyp_params.batch_chunk
                
            combined_loss = 0
            net = nn.DataParallel(model) if batch_size > 10 else model
            if batch_chunk > 1:
                raw_loss = combined_loss = 0
                text_chunks = text.chunk(batch_chunk, dim=0)
                audio_chunks = audio.chunk(batch_chunk, dim=0)
                vision_chunks = vision.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                
                for i in range(len(text_chunks)):
                    text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                    eval_attr_i = eval_attr_chunks[i]
                    preds_i, hiddens_i = net(text_i, audio_i, vision_i)
                    raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                    raw_loss += raw_loss_i
                    raw_loss_i.backward()
                combined_loss = raw_loss
            else:
                preds, hiddens = net(text, audio, vision)
                raw_loss = criterion(preds, eval_attr)
                combined_loss = raw_loss
                combined_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                loginfo='Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.format(
                    epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss)
                if verbose: print(loginfo)
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False,onlypunch=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (x_c, x_p, eval_attr) in enumerate(loader):
                x_p = torch.unsqueeze(x_p, dim=1)
                combined = x_p if onlypunch else torch.cat([x_c, x_p], dim=1)
                t,a,v = crop(combined)
                bs,segs,lens,_=a.shape
                audio,text,vision=a.reshape(bs,segs*lens,-1),t.reshape(bs,segs*lens,-1),v.reshape(bs,segs*lens,-1)
            
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        
                batch_size = text.size(0)
                
                net = nn.DataParallel(model) if batch_size > 10 else model
                preds, _ = net(text, audio, vision)
                total_loss += criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    if not test_only:
        best_valid = 1e8
        for epoch in range(1, hyp_params.num_epochs+1):
            start = time.time()
            train_loss=train(model, optimizer, criterion, verbose,onlypunch=onlypunch)
            val_loss, _, _ = evaluate(model, criterion, test=False,onlypunch=onlypunch)
            test_loss, _, _ = evaluate(model, criterion, test=True,onlypunch=onlypunch)
            
            end = time.time()
            duration = end-start
            scheduler.step(val_loss)    # Decay learning rate by validation loss

            loginfo="-"*50+'\n'
            loginfo='Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}\n'.format(
                        epoch, duration, train_loss, val_loss, test_loss)
            loginfo+="-"*50
            if verbose: print(loginfo)
            log+=loginfo+'\n'
            
            if val_loss < best_valid:
                loginfo=f"Saved model at pre_trained_models/{hyp_params.name}.pt!"
                if verbose: print(loginfo)
                log+=loginfo+'\n'
                save_model(hyp_params, model, name=hyp_params.name)
                best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, criterion, test=True, onlypunch=onlypunch)
    metric,loginfo=eval_ur_funny(results, truths)
    if verbose: print(loginfo)
    log+=loginfo

    with open('./pre_trained_models/'+hyp_params.name+'_log.txt','w') as f: f.write(log)
    return metric

