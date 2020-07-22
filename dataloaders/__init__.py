from DR_deeplab_xception.dataloaders.datasets import idrid,pascal
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class
    #
    # elif args.dataset == 'cityscapes':
    #     train_set = cityscapes.CityscapesSegmentation(args, split='train')
    #     val_set = cityscapes.CityscapesSegmentation(args, split='val')
    #     test_set = cityscapes.CityscapesSegmentation(args, split='test')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #
    #     return train_loader, val_loader, test_loader, num_class
    #
    # elif args.dataset == 'coco':
    #     train_set = coco.COCOSegmentation(args, split='train')
    #     val_set = coco.COCOSegmentation(args, split='val')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None
    #     return train_loader, val_loader, test_loader, num_class
    if args.dataset == 'idrid':
        train_set = idrid.IDRiDegmentation(args, split='train')
        val_set = idrid.IDRiDegmentation(args, split='test')
        seg_num_class = train_set.seg_NUM_CLASSES
        grade_num_class  =train_set.grade_num_classes
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, seg_num_class

    else:
        raise NotImplementedError

