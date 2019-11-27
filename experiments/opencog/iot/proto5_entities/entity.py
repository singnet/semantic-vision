from opencog.type_constructors import ConceptNode

class Entity:
    def __init__(self, state_dict, queue_send):
        self.set_state(state_dict)
        self.queue_send = queue_send
        
    def set_state(self, state_dict):
        self.entity_id = state_dict["entity_id"]
        self.state = state_dict["state"]
        self.attributes = state_dict["attributes"]
        self.last_changed = state_dict["last_changed"]
        self.last_updated = state_dict["last_updated"]
        
    def send_simple_command(self, service):
        msg = {"type": "call_service",  "domain": self.get_domain(), "service": service.name, "service_data": { "entity_id": self.entity_id}}
        self.queue_send.put(msg)
        return ConceptNode("wait #119 merge into master")
        
    def is_state_equal_cn(self, cn):
        return ConceptNode(str(self.state == cn.name))
    
    def get_domain(self):
        return self.entity_id.split(".")[0]

    
def entity_unavailable_state(entity_id):
    return {'entity_id': entity_id, 'state': 'unavailable', 'last_changed': '0000-00-00T00:00:00.000000+00:00', 'last_updated': '0000-00-00T00:00:00.000000+00:00', 'attributes' : {}}


def safe_load_entity_ids(fname):
    try:
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip() for x in content] 
            return content
    except:
        return []
    
def safe_save_entity_ids(entity_ids, fname):
    try:
        with open(fname, 'w') as f:
            for e in entity_ids:
                f.write("%s\n"%e)
    except:
        pass
    
