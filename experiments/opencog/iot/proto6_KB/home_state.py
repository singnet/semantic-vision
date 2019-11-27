from opencog.type_constructors import ConceptNode, GroundedObjectNode
from entity import Entity, entity_unavailable_state

class HomeState:
    def __init__(self, initial_state, queue_send, known_entity_ids):
        self.entities = {s["entity_id"] : Entity(s, queue_send) for s in initial_state["result"]}
            
        # if some known entities are not in initial_state we create an Entity object (with unavailable state) anyway 
        for entity_id in known_entity_ids:
            if (entity_id not in self.entities):
                self.entities[entity_id] = Entity(entity_unavailable_state(entity_id), queue_send)
            
        for entity_id, entity in self.entities.items():            
            GroundedObjectNode(entity_id, entity , unwrap_args = False)

    def get_all_entity_ids(self):
        return list(self.entities.keys())
    
    def state_changed_event_handler(self, msg):
        data = msg["data"]["new_state"]
        if (data["entity_id"] not in self.entities):
            raise Exception("We do not support new devices yet")
        self.entities[data["entity_id"]].set_state(data)
