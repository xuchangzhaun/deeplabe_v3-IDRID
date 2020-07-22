
class Path(object):
    @staticmethod
    def db_root_dir(dataset,kaggle=False):
        if dataset == 'pascal':
            if kaggle:
                return '../input/voctrainval-2012/vocdevkit/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
            else:
                return '/Volumes/backup/Users/jcoral/Desktop/DRnetwork_research/voc2012-segmentation/'
        elif dataset == 'sbd':
            if kaggle:
                return '../input/voctrainval-2012/vocdevkit/VOCdevkit/VOC2012/' # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'idrid':
            if kaggle:
                return '../input/IDRID_Segmentation/Original_Images/'
            else:
                return '/Volumes/backup/Users/jcoral/Desktop/DRnetwork_research/IDRID_Segmentation'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
