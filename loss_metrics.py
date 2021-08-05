import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMOOTH = 1e-6
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x H x W shape
    with torch.no_grad():
        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
        
        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0    
    return iou 

def mIoU(pred, target, metrics, IOUs, NUM_IMGS):
    #n_classes ：the number of classes in your dataset
    # for mask and ground-truth label, not probability map

    N,C,H,W = target.shape
    bg_channel = torch.zeros((N,1,H,W)).to(device)
    pred = torch.cat((bg_channel,pred),axis=1).argmax(1)
    target = torch.cat((bg_channel,target),axis=1).argmax(1)

    # Ignore IoU for background class ("0")
    class_iou =torch.zeros((N,C)).to(device)

    for cls in range(1,C+1):
        # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred==cls
        target_inds = target==cls
        iou = iou_pytorch(pred_inds, target_inds)
        a = target_inds.view(N,-1).any(dim=1) # find classes with ground truths
        class_iou[:,cls-1] = iou*a
        img_in_cls = class_iou[:,cls-1].count_nonzero()
        if img_in_cls!=0:

            IOUs[cls] += (class_iou[:,cls-1].sum()).data.item()
            NUM_IMGS[cls] += img_in_cls.data.item()

    cls_in_img = class_iou.count_nonzero(dim=1)
    miou_imgs = class_iou.sum(dim=1)/cls_in_img
    
    metrics['mIoU'] += miou_imgs.sum().data.cpu().numpy()
    return miou_imgs.mean()

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):

    bce = F.binary_cross_entropy_with_logits(pred, target)
    
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight) 
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss