from pathlib import Path
import torch
from opencog.utilities import tmp_atomspace

from tbd.tbd.module_net import TbDNet
from tbd.utils.clevr import load_vocab, ClevrDataLoaderH5
from opencog.utilities import initialize_opencog
from opencog.scheme_wrapper import scheme_eval
import tbd_helpers
from cognets import CogModel, CogModule, InputModule, get_value
from opencog.type_constructors import *
from opencog.bindlink import execute_atom

device = 'cuda' if torch.cuda.is_available() else 'cpu'

clevr_answer_map_int_str = {23: 'metal', 27: 'rubber', 22: 'large', 29: 'sphere', 28: 'small', 19: 'cylinder', 17: 'cube',
                            30:'yellow', 26:'red', 25: 'purple', 21:'green', 20:'gray', 18:'cyan', 16:'brown', 15:'blue'}

clevr_answers_map_str_int = {v: k for (k, v) in clevr_answer_map_int_str.items()}


def load_tbd_net(checkpoint, vocab, feature_dim=(1024,14,14)):
    tbd_net = TbDNet(vocab, feature_dim)
    tbd_net.load_state_dict(torch.load(str(checkpoint), map_location={'cuda:0': 'cpu'}))
    if torch.cuda.is_available():
        tbd_net.cuda()
    return tbd_net


class TBDModel(CogModel):

    def answer_by_programs(self, tdb_net, features, programs):
        """
        Compute answers from image features and programs

        :param tdb_net: tbd.TbDNet
            Object holding neural networks
        :param features: torch.Tensor
            Images features
        :param programs: torch.Tensor
            Programs in numeric form
        :return: List[str]
            answers as strings
        """
        batch_size = features.size(0)
        feat_input_volume = tdb_net.stem(features)
        results = []
        for n in range(batch_size):
            feat_input = feat_input_volume[n:n + 1]
            output = feat_input
            program = []
            for i in programs.data[n].cpu().numpy():
                module_type = tdb_net.vocab['program_idx_to_token'][i]
                if module_type == '<NULL>':
                    continue
                program.append(module_type)
            result = self.run_program(output, program, tdb_net.ones_var)
            results.append(result)
        return results


    def argmax(self, items):
        items.sort(reverse=True)
        return items[0][1]

    def run_program(self, features, program, attention):
        with tmp_atomspace() as tmp:
            prog, left = tbd_helpers.return_prog(program)
            query = tbd_helpers.ast_to_query(prog, tmp)
            scene = InputModule(ConceptNode("scene"), {'features':features,
                                                       'attention': attention.clone(),
                                                       'name': ConceptNode("scene").name})
            result = execute_atom(tmp, query)
            answer = self.argmax(self.extract(result))
            return answer

    def extract(self, result):
        tmp = [get_value(x) for x in result.out]
        results = []
        for item in tmp:
            name = tbd_helpers.filter_reg.match(item['name']).group(2)
            results.append((item['attention'].sum().numpy(), name))
        return results


def test(atomspace):
    commands = ['<END>', 'scene', 'filter_size[large]', 'filter_color[purple]', 'unique', 'query_material', '<START>']
    prog, left = tbd_helpers.return_prog(list(reversed(commands)))

    # query = tbd_helpers.ast_to_query(prog, atomspace)
    commands = ['<END>', 'scene', 'filter_color[blue]', 'filter_shape[cube]', 'unique', 'relate[left]', 'scene', 'filter_size[large]', 'filter_color[green]', 'filter_material[rubber]', 'filter_shape[cube]', 'unique', 'relate[right]', 'intersect', 'unique', 'query_material', '<START>']
    prog, left = tbd_helpers.return_prog(list(reversed(commands)))
    query = tbd_helpers.ast_to_query(prog, atomspace)

def main():
    torch.set_grad_enabled(False)
    atomspace = AtomSpace()
    initialize_opencog(atomspace)
    tbd_net_checkpoint = '/mnt/fileserver/shared/models/tbd-nets-models/clevr-reg.pt'
    vocab = load_vocab(Path('/mnt/fileserver/shared/models/tbd-nets-models/data/vocab.json'))
    tbd_net = load_tbd_net(tbd_net_checkpoint, vocab, feature_dim=(1024, 14, 14))
    wrappers = tbd_helpers.generate_wrappers(tbd_net.function_modules, atomspace)
    tbd_helpers.setup_inheritance(tbd_net.function_modules, atomspace)
    test(atomspace)
    BATCH_SIZE = 64
    val_loader_kwargs = {
        'question_h5': Path('/mnt/fileserver/shared/datasets/CLEVR_v1/data/val_questions_query_ending.h5'),
        'feature_h5': Path('/mnt/fileserver/shared/datasets/CLEVR_v1/data/val_features.h5'),
        'batch_size': BATCH_SIZE,
        'num_workers': 0,
        'shuffle': False
    }


    scheme_eval(atomspace, '(use-modules (opencog logger))')
    scheme_eval(atomspace, '(cog-logger-set-level! "fine")')
    loader = ClevrDataLoaderH5(**val_loader_kwargs)
    model = TBDModel(atomspace)
    total_acc = 0
    for i, batch in enumerate(loader):
        _, _, feats, expected_answers, programs = batch
        feats = feats.to(device)
        programs = programs.to(device)
        results = model.answer_by_programs(tbd_net, feats, programs)
        correct = 0
        clevr_numeric_actual_answers = [clevr_answers_map_str_int[x] for x in results]
        for (actual, expected) in zip(clevr_numeric_actual_answers, expected_answers):
            correct += 1 if actual == expected else 0
        acc = float(correct) / len(programs)
        total_acc = total_acc * (i / (i + 1)) + acc/(i + 1)
        print("Accuracy average: {0}".format(total_acc))


if __name__ == '__main__':
    main()
