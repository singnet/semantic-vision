from opencog.type_constructors import ConceptNode

class Event:
    def __init__(self, event):
        assert "event_type" in event and "data" in event, "Wrong event structure" 
        self.event_type = event["event_type"].split(".")[-1]
        self.data = event["data"]
        
    def is_event_type_equal_cn(self, cn):
        return ConceptNode(str(cn.name == self.event_type))
    
    def is_data_equal_cn(self, key, cn):
        return ConceptNode(str((key.name in self.data) and cn.name == self.data[key.name]))
                
