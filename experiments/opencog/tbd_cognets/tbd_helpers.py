import re
from opencog.atomspace import AtomSpace, types

from cognets import InputModule, CogModule, execute

filter_reg = re.compile('[a-z]+_([a-z]+)\[([a-z]+)\]')


class TbdWrapper(CogModule):
    def __init__(self, atom, tbd_module):
        super().__init__(atom)
        self.tbd_module = tbd_module

    def forward(self, args):
        features, attention = args['features'], args['attention']
        att = self.tbd_module.forward(features, attention)
        result = {'features': features,
                  'attention': att,
                  'name': self.atom.name}
        return result


class TbdIntersect(TbdWrapper):
    def forward(self, arg0, arg1):
        att0, att1 = arg0['attention'], arg1['attention']
        features = arg0['features']
        return {'features': features,
                'attention': self.tbd_module.forward(att0, att1),
                'name': self.atom.name}


def generate_wrappers(function_modules, atomspace):
    result = dict()
    for modulename, module in function_modules.items():
        atom = atomspace.add_node(types.ConceptNode, modulename)
        if modulename == 'intersect':
            result[modulename] = TbdIntersect(atom, module)
        else:
            result[modulename] = TbdWrapper(atom, module)
    return result


def setup_inheritance(module_dict, atomspace):
    """
    Creates inheritances links for filter modules e.g.

    InheritanceLink(metal, material)
    EvaluationLink(filters, metal, filter_material[metal])
    """
    predicate = atomspace.add_node(types.PredicateNode, 'filters')
    for modulename in module_dict.keys():
        if modulename.startswith('filter'):
            # e.g. 'material', 'metal'
            parent, child = filter_reg.match(modulename).groups()
            parent_atom = atomspace.add_node(types.ConceptNode, parent)
            child_atom = atomspace.add_node(types.ConceptNode, child)
            module_atom = atomspace.add_node(types.ConceptNode, modulename)
            inh_link = atomspace.add_link(types.InheritanceLink, [child_atom, parent_atom])
            # filter_material[metal] filters metal
            eval_link = atomspace.add_link(types.EvaluationLink,
                               [predicate, atomspace.add_link(types.ListLink, [child_atom, module_atom])])


def ast_to_query(ast, atomspace):
    """
    Accepts ast in form of nested lists, convert it to query
    """
    current, rest = ast[0], ast[1:]
    if isinstance(current, list):
        result = []
        for item in current:
            result.append(*ast_to_query(item, atomspace))
        assert not rest
        return result
    elif current.startswith('query'):
        query_type = current.split('_')[-1]
        query_type_atom = atomspace.add_node(types.ConceptNode, query_type)
        material = atomspace.add_node(types.VariableNode, "$material")
        modulename = atomspace.add_node(types.VariableNode, "$modulename")
        inh_material = atomspace.add_link(types.InheritanceLink, [material, query_type_atom])
        predicate = atomspace.add_node(types.PredicateNode, 'filters')
        filters = atomspace.add_link(types.EvaluationLink,
                               [predicate,
                               atomspace.add_link(types.ListLink, [material, modulename])])
        constraints = [inh_material, filters]
        query = execute(modulename, *ast_to_query(rest, atomspace))
        andlink = atomspace.add_link(types.AndLink, constraints)
        bindlink = atomspace.add_link(types.BindLink, [andlink, query])
        return bindlink
    elif (current.startswith('filter') or
        current.startswith('relate') or
        current.startswith('same') or
        current.startswith('intersect')):
        atom = atomspace.add_node(types.ConceptNode, current)
        return [execute(atom, *ast_to_query(rest, atomspace))]
    elif current.startswith('scene'):
        atom = atomspace.add_node(types.ConceptNode, current)
        assert not rest
        return [execute(atom)]
    raise NotImplementedError(current)


def return_prog(program):
    current, rest = program[0], program[1:]
    if (current.startswith('query') or
       current.startswith('filter') or
       current.startswith('relate') or
       current.startswith('same')):
        arg, left = return_prog(rest)
        return [current] + arg, left
    if current.startswith('scene'):
        return [current], rest
    elif current == '<START>':
        return return_prog(rest)
    elif current == 'unique':
        return return_prog(rest)
    elif current.startswith('intersect'):
        sub_prog0, left = return_prog(rest)
        sub_prog1, right = return_prog(left)
        return [current, [sub_prog0, sub_prog1]], right
    elif current == '<END>':
        return [], []
    raise NotImplementedError(current)


if __name__ == '__main__':
    commands = ['<END>', 'scene', 'filter_size[large]', 'filter_color[purple]', 'unique', 'query_material', '<START>']
    prog, left = return_prog(list(reversed(commands)))
    commands = ['<END>', 'scene', 'filter_color[blue]', 'filter_shape[cube]', 'unique', 'relate[left]', 'scene', 'filter_size[large]', 'filter_color[green]', 'filter_material[rubber]', 'filter_shape[cube]', 'unique', 'relate[right]', 'intersect', 'unique', 'query_material', '<START>']
    prog, left = return_prog(list(reversed(commands)))
    import pdb;pdb.set_trace()
