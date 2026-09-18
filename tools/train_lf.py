import argparse
import csv
import json
import time
import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _init_paths  # noqa: F401

from mvface.data.facescape_multiview import FaceScapeMultiView as FaceScapeMultiViewV2
from mvface.losses import decoder_losses, mpjpe_mm
from mvface.model import MultiViewLandmark3D


def make_cfg(args):
    cfg = types.SimpleNamespace()
    cfg.DATASET = types.SimpleNamespace(
        ROOT=args.root,
        NUM_VIEWS=args.num_views,
        USE_RETINAFACE=args.retinaface,
        DEPTH_SCALE=200.0,
        RETINAFACE_ROOT=args.retinaface_root,
        RETINAFACE_CHECKPOINT=args.retinaface_ckpt,
    )
    cfg.NETWORK = types.SimpleNamespace(
        IMAGE_SIZE=(args.img_size, args.img_size),
    )
    return cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root',   default='/nfs/turbo/coe-igmr-pub/seoin/FaceScape')
    p.add_argument('--assets', default='/nfs/turbo/coe-igmr-pub/seoin/MVFace/assets')
    p.add_argument('--out',    default='output/run')
    p.add_argument('--retinaface', action='store_true')
    p.add_argument('--retinaface-root',
                   default='/nfs/turbo/coe-igmr-pub/seoin/Pytorch_Retinaface')
    p.add_argument('--retinaface-ckpt',
                   default='/nfs/turbo/coe-igmr-pub/seoin/trained_weights/Resnet50_Final.pth')
    p.add_argument('--no-depth',    action='store_true')
    p.add_argument('--late-fusion', action='store_true')
    p.add_argument('--epochs',    type=int,   default=150)
    p.add_argument('--bs',        type=int,   default=2)
    p.add_argument('--lr',        type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=1e-3)
    p.add_argument('--lambda-2d', type=float, default=1e-4,
                   help='weight on the 2D reprojection loss term')
    p.add_argument('--num-layers',type=int,   default=4)
    p.add_argument('--img-size',  type=int,   default=256)
    p.add_argument('--num-views', type=int,   default=5)
    p.add_argument('--workers',   type=int,   default=4)
    p.add_argument('--val-freq',  type=int,   default=5)
    p.add_argument('--seed',      type=int,   default=0)
    p.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume',    default='')
    return p.parse_args()


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def nme_interocular(pred_3d, gt_3d):
    iod = (gt_3d[:, 45] - gt_3d[:, 36]).norm(dim=-1, keepdim=True).unsqueeze(1)
    iod = iod.clamp(min=1e-6)
    err = (pred_3d - gt_3d).norm(dim=-1)
    return float((err / iod.squeeze(1)).mean())


@torch.no_grad()
def evaluate(model, loader, device, use_late_fusion, lambda_2d):
    model.eval()
    loss_sum, nme_sum, mpjpe_sum, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        batch = move(batch, device)
        hw = (batch['rgbd'].shape[-2], batch['rgbd'].shape[-1])
        depth_maps = batch['depth_raw'] if use_late_fusion else None
        preds_3d, preds_2d = model(batch['rgbd'], batch['proj'], hw,
                                   depth_maps=depth_maps)
        b = batch['rgbd'].shape[0]
        losses = decoder_losses(preds_3d, preds_2d,
                                batch['landmarks_3d'],
                                batch['landmarks_2d'],
                                batch['vis'], lambda_2d=lambda_2d)
        loss_sum  += float(losses['total']) * b
        nme_sum   += nme_interocular(preds_3d[-1], batch['landmarks_3d']) * b
        mpjpe_sum += float(mpjpe_mm(preds_3d[-1], batch['landmarks_3d'])) * b
        n += b
    n = max(n, 1)
    return loss_sum / n, nme_sum / n, mpjpe_sum / n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    use_depth       = not args.no_depth
    use_late_fusion = args.late_fusion and use_depth
    mode = 'RGB-only' if not use_depth else ('late-fusion' if use_late_fusion else 'early-fusion')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    logf = open(out / 'train.log', 'w')
    def logprint(msg):
        print(msg); logf.write(msg + '\n'); logf.flush()

    csvf   = open(out / 'metrics.csv', 'w', newline='')
    writer = csv.writer(csvf)
    writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss',
                     'val_nme_pct', 'val_mpjpe_mm', 'skipped', 'sec'])
    csvf.flush()

    (out / 'run_config.json').write_text(json.dumps(vars(args), indent=2))
    logprint(f'mode={mode}  retinaface={args.retinaface}  '
             f'epochs={args.epochs}  bs={args.bs}  lr={args.lr}')

    cfg = make_cfg(args)
    train_ds = FaceScapeMultiViewV2(cfg, image_set='train', is_train=True)
    val_ds   = FaceScapeMultiViewV2(cfg, image_set='val',   is_train=False)
    logprint(f'train={len(train_ds)} samples  val={len(val_ds)} samples')

    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=args.workers, drop_last=True, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    model = MultiViewLandmark3D(
        args.assets, num_layers=args.num_layers,
        use_depth=use_depth, img_size=args.img_size,
    ).to(args.device)

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_nme    = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        if 'opt'   in ckpt: opt.load_state_dict(ckpt['opt'])
        if 'sched' in ckpt: sched.load_state_dict(ckpt['sched'])
        start_epoch = ckpt.get('epoch', 0)
        best_nme    = ckpt.get('best_nme', float('inf'))
        logprint(f'resumed from epoch {start_epoch}, best_nme={best_nme:.4f}')

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0, running, skipped = time.time(), 0.0, 0

        for batch in train_ld:
            batch = move(batch, args.device)
            hw    = (batch['rgbd'].shape[-2], batch['rgbd'].shape[-1])
            depth_maps = batch['depth_raw'] if use_late_fusion else None
            preds_3d, preds_2d = model(batch['rgbd'], batch['proj'], hw,
                                       depth_maps=depth_maps)
            losses = decoder_losses(preds_3d, preds_2d,
                                    batch['landmarks_3d'],
                                    batch['landmarks_2d'],
                                    batch['vis'], lambda_2d=args.lambda_2d)
            loss = losses['total']
            if not torch.isfinite(loss):
                skipped += 1; continue
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if torch.isfinite(gnorm):
                opt.step(); running += loss.item()
            else:
                skipped += 1

        lr         = opt.param_groups[0]['lr']
        sched.step()
        train_loss = running / max(len(train_ld) - skipped, 1)
        sec        = time.time() - t0

        if epoch % args.val_freq == 0 or epoch == args.epochs:
            val_loss, val_nme, val_mpjpe = evaluate(model, val_ld,
                                                     args.device, use_late_fusion,
                                                     args.lambda_2d)
            skip_note = f'  skipped={skipped}' if skipped else ''
            logprint(f'epoch {epoch:3d}  train_loss {train_loss:8.4f}  '
                     f'val_loss {val_loss:8.4f}  '
                     f'NME {val_nme*100:6.3f}%  '
                     f'MPJPE {val_mpjpe:7.2f}mm  '
                     f'({sec:.0f}s){skip_note}')
            writer.writerow([epoch, f'{lr:.3e}', f'{train_loss:.6f}',
                             f'{val_loss:.6f}', f'{val_nme*100:.4f}',
                             f'{val_mpjpe:.4f}', skipped, f'{sec:.1f}'])
            csvf.flush()
            ckpt = {'epoch': epoch, 'model': model.state_dict(),
                    'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'val_nme': val_nme, 'val_mpjpe': val_mpjpe,
                    'best_nme': best_nme, 'args': vars(args)}
            torch.save(ckpt, out / 'last.pth')
            if val_nme < best_nme:
                best_nme = val_nme
                torch.save(ckpt, out / 'best.pth')
                logprint(f'  *** new best NME {best_nme*100:.3f}% ***')
        else:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'opt': opt.state_dict(), 'sched': sched.state_dict(),
                        'best_nme': best_nme, 'args': vars(args)},
                       out / 'last.pth')
            logprint(f'epoch {epoch:3d}  train_loss {train_loss:8.4f}  ({sec:.0f}s)  [no val]')

    logprint(f'\ndone. best NME {best_nme*100:.3f}%  ->  {out/"best.pth"}')
    csvf.close(); logf.close()


if __name__ == '__main__':
    main()
