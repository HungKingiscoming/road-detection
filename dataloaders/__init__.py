from dataloaders.datasets import spacenet
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    if args.dataset == 'spacenet':
        train_loader, val_loader = spacenet.build_spacenet_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=kwargs.get('num_workers', 4),
            base_size=args.base_size,
            crop_size=args.crop_size,
        )
        test_loader = None
        num_classes = 2
        return train_loader, val_loader, test_loader, num_classes
    else:
        raise NotImplementedError(f"Dataset {args.dataset} chưa được hỗ trợ!")
