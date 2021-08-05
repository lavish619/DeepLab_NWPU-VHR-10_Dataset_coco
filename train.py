import time
import copy
from collections import defaultdict
from tqdm.notebook import tqdm

import torch
from loss_metrics import calc_loss, mIoU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    

def print_ious(IOUs, NUM_IMGS, phase):    
    outputs = []
    outputs.append('mIOU classes')
    for k in sorted(IOUs.keys()):
        outputs.append("cls_{}: {:4f}".format(k, IOUs[k] / NUM_IMGS[k]))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    

def train_model(model, dataloaders, optimizer, scheduler = None, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_miou = 0

    losses = {'train':[], 'val': [] }
    class_miou = {phase :{i:[] for i in range(1,11)} for phase in ["train", "val"]}
    total_miou= {'train':[], 'val': [] }
    not_imp = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            IOUs = defaultdict(float)
            NUM_IMGS = defaultdict(int)
            epoch_samples = 0
            
            for inputs, labels,_ in tqdm(dataloaders[phase]):
                # print(inputs.shape, labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    miou = mIoU(outputs, labels, metrics, IOUs, NUM_IMGS)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            if scheduler is not None and phase=="train":
                scheduler.step()
                
            print_metrics(metrics, epoch_samples, phase)
            print_ious(IOUs, NUM_IMGS, phase)

            epoch_miou = metrics['mIoU']/epoch_samples
            losses[phase].append(metrics['loss'] / epoch_samples)
            total_miou[phase].append(epoch_miou)
            
            for k in sorted(IOUs.keys()):
                class_miou[phase][k].append(IOUs[k] / NUM_IMGS[k])
           
            # deep copy the model
            if phase == 'val' and epoch_miou > best_miou:
                not_imp=0
                print("saving best model")
                best_miou = epoch_miou
                best_model_wts = copy.deepcopy(model.state_dict())
            elif phase=='val' and epoch_miou < best_miou:
                not_imp += 1

            if not_imp > 4:
                break 

        if not_imp > 4:
            break        
        
        # print(labels.shape, outputs.shape)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val mIOU: {:4f}'.format(best_miou))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, losses, class_miou, total_miou