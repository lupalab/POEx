import os
import pickle

def get_dataset(args, split):
    if args.dataset == 'omniglot':
        from .omniglot import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size, args.mask_type)
    elif args.dataset == 'mnist':
        from .mnist import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size, args.mask_type)
    elif args.dataset == 'video':
        from .video import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size, args.t_dim)
    elif args.dataset == 'youtube':
        from .youtube import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size, args.mask_type, args.t_dim)
    elif args.dataset == 'knee':
        from .knee import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size, args.t_dim)
    elif args.dataset == 'modelnet':
        from .modelnet import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size, args.category, args.mask_type)
    elif args.dataset == 'sig_modelnet':
        from .modelnet import mDataset
        dataset = mDataset(split, args.batch_size, args.set_size, args.category, args.mask_type)
    elif args.dataset == 'modelnet40':
        from .modelnet40 import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size, args.mask_type)
    elif args.dataset == 'colon':
        from .colon import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size, args.mask_type)
    elif args.dataset == 'occo':
        from .occo import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size)
    elif args.dataset == 'molecule':
        from .molecule import Dataset
        dataset = Dataset(split, args.batch_size, args.mask_type)
    elif args.dataset == 'shapenet':
        from .shapenet import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size)
    elif args.dataset == 'gp':
        from .gp_curve import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size)
    elif args.dataset == 'mtgp':
        from .gp_curve import MTDataset
        dataset = MTDataset(split, args.batch_size, args.set_size)
    elif args.dataset == 'img_curve':
        from .img_curve import Dataset
        dataset = Dataset(split, args.batch_size, args.set_size)
    elif args.dataset == 'uci':
        from .uci import Dataset
        dataset = Dataset(args.data, split, args.batch_size)
    else:
        raise ValueError()

    return dataset
    
def cache(args, split, fname):
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            batches = pickle.load(f)
    else:
        batches = []
        dataset = get_dataset(args, split)
        dataset.initialize()
        for _ in range(dataset.num_batches):
            batch = dataset.next_batch()
            batches.append(batch)
        with open(fname, 'wb') as f:
            pickle.dump(batches, f)

    return batches


def get_cached_data(args, split):
    if args.dataset in ['omniglot', 'mnist']:
        fname = f'.{args.dataset}_{split}_{args.mask_type}'
    elif args.dataset in ['video', 'youtube', 'knee']:
        fname = f'.{args.dataset}_{split}_t{args.t_dim}'
    elif args.dataset in ['shapenet']:
        fname = f'.{args.dataset}_{split}'
    elif args.dataset in ['modelnet40', 'colon']:
        fname = f'.{args.dataset}_{args.set_size}_{split}_{args.mask_type}'
    elif args.dataset in ['modelnet', 'sig_modelnet']:
        fname = f'.{args.dataset}_{args.category}_{args.set_size}_{split}_{args.mask_type}'
    elif args.dataset in ['occo']:
        fname = f'.{args.dataset}_{args.set_size}_{split}'
    elif args.dataset in ['molecule']:
        fname = f'.{args.dataset}_{split}_{args.mask_type}'
    elif args.dataset in ['gp', 'mtgp', 'img_curve']:
        fname = f'.{args.dataset}_{args.set_size}_{split}'
    else:
        raise ValueError()

    return cache(args, split, fname)

