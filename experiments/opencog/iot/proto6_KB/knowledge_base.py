from opencog.type_constructors import *


def knowledge_base():
    GON = GroundedObjectNode
    button1 = GON("binary_sensor.switch_158d00039928d1")
    door_sensor1 = GON("binary_sensor.door_window_sensor_158d0003973109")
    lamp1  = GON("light.experiment_lamp_1")
        
    InheritanceLink(ConceptNode("room1"), ConceptNode("room"))
    InheritanceLink(ConceptNode("room2"), ConceptNode("room"))
    InheritanceLink(lamp1, ConceptNode("light"))
    InheritanceLink(door_sensor1, ConceptNode("door_windows_sensor"))
    InheritanceLink(button1, ConceptNode("button"))     
    EvaluationLink(PredicateNode ("placed-in"), ListLink(lamp1, ConceptNode("room1")))
    EvaluationLink(PredicateNode ("placed-in"), ListLink(button1, ConceptNode("room2")))
    EvaluationLink(PredicateNode ("placed-in"), ListLink(door_sensor1, ConceptNode("front-door")))
    
