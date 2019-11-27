from opencog.type_constructors import *

def atom_to_truth_value(atom):
    return TruthValue(atom.name == "True", 1)
    
def condition(gb, fn, args):
    llargs = ListLink(*[ConceptNode(a) for a in args])
    return EvaluationLink( GroundedPredicateNode("py: atom_to_truth_value"), ApplyLink(MethodOfLink(gb, ConceptNode(fn)), llargs))

def opencog_reactive_automation_bindlinks(gb_current_event):  
    button1 = "binary_sensor.switch_158d00039928d1"
    door_sensor1 = "binary_sensor.door_window_sensor_158d0003973109"
    lamp1  = "light.experiment_lamp_1"  
    
    GON = GroundedObjectNode
    
    bls = []
    bls.append(BindLink(AndLink(condition(gb_current_event,  "is_event_type_equal_cn", ["click"]),
                       condition(gb_current_event,  "is_data_equal_cn",   ["entity_id", button1]),
                       condition(gb_current_event,  "is_data_equal_cn",   ["click_type", "single"]),
                       condition(GON(door_sensor1), "is_state_equal_cn", ["off"]),
                       condition(GON(lamp1),        "is_state_equal_cn", ["off"])),
               ApplyLink(MethodOfLink(GON(lamp1), ConceptNode("send_simple_command")), ListLink(ConceptNode("turn_on")))))
    bls.append(BindLink(AndLink(condition(gb_current_event,  "is_event_type_equal_cn", ["click"]),
                       condition(gb_current_event,  "is_data_equal_cn",   ["entity_id", button1]),
                       condition(gb_current_event,  "is_data_equal_cn",   ["click_type", "double"]),
                       condition(GON(door_sensor1), "is_state_equal_cn", ["off"]),
                       condition(GON(lamp1),        "is_state_equal_cn", ["on"])),
               ApplyLink(MethodOfLink(GON(lamp1), ConceptNode("send_simple_command")), ListLink(ConceptNode("turn_off")))))
    return bls
