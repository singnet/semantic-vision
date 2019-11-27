from opencog.type_constructors import *

def atom_to_truth_value(atom):
    return TruthValue(atom.name == "True", 1)
    
def condition(gb, fn, args):
    llargs = ListLink(*[ConceptNode(a) if type(a) == str else a for a in args])
    return EvaluationLink( GroundedPredicateNode("py: atom_to_truth_value"), ApplyLink(MethodOfLink(gb, ConceptNode(fn)), llargs))

def opencog_reactive_automation_bindlinks(gb_current_event):  
    
    GON = GroundedObjectNode
    
    bls = []
    bls.append(BindLink(AndLink(EvaluationLink(PredicateNode ("placed-in"), ListLink(VariableNode("button"), ConceptNode("room2"))),
                                EvaluationLink(PredicateNode ("placed-in"), ListLink(VariableNode("lamp"), ConceptNode("room1"))),
                                EvaluationLink(PredicateNode ("placed-in"), ListLink(VariableNode("door_sensor"), ConceptNode("front-door"))),
                                InheritanceLink(VariableNode("lamp"), ConceptNode("light")),
                                InheritanceLink(VariableNode("button1"), ConceptNode("button")),
                                InheritanceLink(VariableNode("door_sensor"), ConceptNode("door_windows_sensor")),
                                condition(gb_current_event,  "is_event_type_equal_cn", ["click"]),
                                condition(gb_current_event,  "is_data_equal_cn",   ["entity_id", VariableNode("button1")]),
                                condition(gb_current_event,  "is_data_equal_cn",   ["click_type", "single"]),
                                condition(VariableNode("door_sensor"), "is_state_equal_cn", ["off"]),
                               condition(VariableNode("lamp"),   "is_state_equal_cn", ["off"])),
               ApplyLink(MethodOfLink(VariableNode("lamp"), ConceptNode("send_simple_command")), ListLink(ConceptNode("turn_on")))))
    bls.append(BindLink(AndLink(EvaluationLink(PredicateNode ("placed-in"), ListLink(VariableNode("button"), ConceptNode("room2"))),
                                EvaluationLink(PredicateNode ("placed-in"), ListLink(VariableNode("lamp"), ConceptNode("room1"))),
                                EvaluationLink(PredicateNode ("placed-in"), ListLink(VariableNode("door_sensor"), ConceptNode("front-door"))),
                                InheritanceLink(VariableNode("lamp"), ConceptNode("light")),
                                InheritanceLink(VariableNode("button1"), ConceptNode("button")),
                                InheritanceLink(VariableNode("door_sensor"), ConceptNode("door_windows_sensor")),
                                condition(gb_current_event,  "is_event_type_equal_cn", ["click"]),
                                condition(gb_current_event,  "is_data_equal_cn",   ["entity_id", VariableNode("button1")]),
                                condition(gb_current_event,  "is_data_equal_cn",   ["click_type", "double"]),
                                condition(VariableNode("door_sensor"), "is_state_equal_cn", ["off"]),
                               condition(VariableNode("lamp"),   "is_state_equal_cn", ["on"])),
               ApplyLink(MethodOfLink(VariableNode("lamp"), ConceptNode("send_simple_command")), ListLink(ConceptNode("turn_off")))))               
    return bls
