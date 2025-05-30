import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from spearman import spearman, spearman_diff
from utils import AverageMeter, cal_deltaE
from reloss import ReLoss
from dataset import DatasetReloss as D


def calc_gradient_penalty(loss_module, S0, Fuse):
    Fuse.requires_grad_()
    loss = loss_module(S0, Fuse)
    Fuse_gradients = autograd.grad(outputs=loss,
                              inputs=Fuse,
                              grad_outputs=torch.ones(loss.size(),
                                                         device=loss.device),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]

    gradients = Fuse_gradients
    penalty = ((gradients.norm(2, dim=None) - 1)**2).mean()
    return penalty


def train_epoch(train_loader, loss_module, loss_optimizer, loss_scheduler):
    loss_module.train()
    current_step = 0
    for idx, train_data in enumerate(train_loader):
        current_step += 1
        S0s = train_data['S0']
        Fuses = train_data['Fuse']
        S0s = S0s.cuda()
        Fuses = Fuses.cuda()
        losses = []
        metrics1 = []
        metrics2 = []
        penalty = []
        for S0_i, Fuse_i in zip(S0s, Fuses):
            # calculate loss and metric for each batch
            S0_i = torch.unsqueeze(S0_i, 0)
            Fuse_i = torch.unsqueeze(Fuse_i, 0)
            loss = loss_module(S0_i, Fuse_i)
            mse_ = F.mse_loss(Fuse_i, S0_i, reduction='mean')
            Delta_E = cal_deltaE(S0_i, Fuse_i)

            losses.append(loss)
            metrics1.append(mse_*10)
            metrics2.append(Delta_E*10)

            penalty_ = calc_gradient_penalty(loss_module, S0_i, Fuse_i)
            penalty.append(penalty_)

        penalty = sum(penalty) / S0s.shape[0]
        losses = torch.stack(losses)
        metrics1 = torch.tensor(metrics1, device=losses.device)
        metrics2 = torch.tensor(metrics2, device=losses.device)

        diff_spea1 = spearman_diff(losses.unsqueeze(0), metrics1.unsqueeze(0))
        diff_spea2 = spearman_diff(losses.unsqueeze(0), metrics2.unsqueeze(0))
        diff_spea = 0*diff_spea1 + diff_spea2
        spea1 = spearman(losses.unsqueeze(0).detach(),
                        metrics1.unsqueeze(0).detach())
        spea2 = spearman(losses.unsqueeze(0).detach(),
                         metrics2.unsqueeze(0).detach())
        spea = 0*spea1 + spea2

        obj = -diff_spea + 10 * penalty
        loss_optimizer.zero_grad()
        obj.backward()
        loss_optimizer.step()
        loss_scheduler.step()

        if idx % 2 == 0:
            print(
                f'Train: [{idx}/{len(train_loader)}] diff_spea {diff_spea.item():.4f} spea {spea.item():.4f}'
            )
            print(f'loss_value   {losses[:5].detach().cpu()}')
            print(f'metric_value {metrics2[:5].detach().cpu()}')
            print(f'penalty {penalty.detach().cpu()}')
            print(f'objective_function {obj.detach().cpu()}')



def val_epoch(val_loader, loss_module):
    loss_module.eval()
    spea_meter = AverageMeter()
    ce_spea_meter = AverageMeter()
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for idx, val_data in enumerate(val_loader):
            S0s = val_data['S0']
            Fuses = val_data['Fuse']
            S0s = S0s.cuda()
            DoLPs = DoLPs.cuda()
            Fuses = Fuses.cuda()
            losses = []
            metrics = []
            ce_losses = []
            for S0_i, DoLP_i, Fuse_i in zip(S0s, DoLPs, Fuses):
                S0_i = torch.unsqueeze(S0_i, 0)
                Fuse_i = torch.unsqueeze(Fuse_i, 0)
                loss = loss_module(S0_i, Fuse_i)
                Delta_E = cal_deltaE(S0_i, Fuse_i)
                Delta_E = torch.tensor(Delta_E).cuda()
                losses.append(loss)
                metrics.append(Delta_E*100)
                ce_losses.append(Delta_E)

            losses = torch.stack(losses)
            ce_losses = torch.stack(ce_losses)
            metrics = torch.tensor(metrics, device=losses.device)

            diff_spea = spearman_diff(losses.unsqueeze(0),
                                  metrics.unsqueeze(0)).item()
            spea = spearman(losses.unsqueeze(0), metrics.unsqueeze(0)).item()
            ce_spea = spearman_diff(ce_losses.unsqueeze(0),
                               metrics.unsqueeze(0)).item()

            spea_meter.update(spea, losses.shape[0])
            ce_spea_meter.update(ce_spea, losses.shape[0])

            if idx % 2 == 0:
                print(
                    f'Val: [{idx}/{len(val_loader)}] diff_spea {diff_spea:.4f} spea {spea:.4f} ce_spea {ce_spea:.4f}'
                )

    print(f'Val: spea {spea_meter.avg:.4f} ce_spea {ce_spea_meter.avg:.4f}')
    return spea_meter.avg


def main():
    # load data
    train_set = D('train')
    val_set = D('val')
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                              num_workers=8, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False,
                              num_workers=1, drop_last=False, pin_memory=True)

    # initialize reloss module
    loss_module = ReLoss()
    loss_module.cuda()
    loss_optimizer = torch.optim.Adam(loss_module.parameters(), 0.0001, weight_decay=1e-3)
    loss_scheduler = lr_scheduler.ExponentialLR(loss_optimizer, gamma=0.95)

    # train reloss
    best_spearman = -1
    total_epochs = 10
    for epoch in range(total_epochs):
        print(f'epoch: {epoch}')
        train_epoch(train_loader, loss_module, loss_optimizer, loss_scheduler)
        spearman = val_epoch(val_loader, loss_module)

        if spearman >= best_spearman:
            # save the best checkpoint
            torch.save(loss_module.state_dict(), 'loss_module_Delta_E.ckpt')
            best_spearman = spearman


if __name__ == '__main__':
    main()
