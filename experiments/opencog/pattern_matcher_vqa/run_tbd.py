from pathlib import Path
import torch

from tbd.tbd.module_net import load_tbd_net
from pattern_matcher_vqa import TBD
from util import initialize_atomspace_by_facts
from tbd.utils.clevr import load_vocab, ClevrDataLoaderNumpy, ClevrDataLoaderH5
import tbd_helpers
# imports for calling from atomspace
from tbd_helpers import init_scene, filter, intersect, classify, relate


device = 'cuda' if torch.cuda.is_available() else 'cpu'

answers_str = {23: 'metal', 27: 'rubber', 22: 'large', 29: 'sphere', 28: 'small', 19: 'cylinder', 17: 'cube',
               30:'yellow', 26:'red', 25: 'purple', 21:'green', 20:'gray', 18:'cyan',16:'brown',15:'blue'}

answers_str_int = {v: k for (k, v) in answers_str.items()}


def main():
    torch.set_grad_enabled(False)
    atomspace = initialize_atomspace_by_facts("tbdas.scm")
    tbd = TBD(None, None, atomspace, None)
    tbd_net_checkpoint = './models/clevr-reg.pt'
    vocab = load_vocab(Path('data/vocab.json'))
    tdb_net = load_tbd_net(tbd_net_checkpoint, vocab, feature_dim=(1024, 14, 14))
    BATCH_SIZE = 64
    val_loader_kwargs = {
        'question_h5': Path('/mnt/fileserver/shared/datasets/CLEVR_v1/data/val_questions_query_ending.h5'),
        'feature_h5': Path('/mnt/fileserver/shared/datasets/CLEVR_v1/data/val_features.h5'),
        'batch_size': BATCH_SIZE,
        'num_workers': 1,
        'shuffle': False
    }

    tbd_helpers.tbd = tdb_net
    loader = ClevrDataLoaderH5(**val_loader_kwargs)
    for batch in loader:
        _, _, feats, answers, programs = batch
        feats = feats.to(device)
        programs = programs.to(device)

        results = tbd.answerByPrograms(tdb_net, feats, programs)
        import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
