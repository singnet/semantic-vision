from queue import Queue
from hass_communicator import start_HassCommunicator_in_thread
from home_state import HomeState
from event import Event
from entity import safe_load_entity_ids, safe_save_entity_ids
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog
from opencog.bindlink import execute_atom
import configparser
from opencog_reactive_automation_bindlinks import opencog_reactive_automation_bindlinks, atom_to_truth_value

config = configparser.ConfigParser()
config.read('config.cfg')


atomspace = AtomSpace()
initialize_opencog(atomspace)

queue_send = Queue()
queue_recv = Queue()

hass_thread = start_HassCommunicator_in_thread(config['DEFAULT']['uri'], config['DEFAULT']['token'], queue_send, queue_recv)
home_state = HomeState(queue_recv.get(), queue_send, safe_load_entity_ids(config['DEFAULT']['known_entity_ids_file']))
safe_save_entity_ids(home_state.get_all_entity_ids(), config['DEFAULT']['known_entity_ids_file'])

gb_current_event = GroundedObjectNode("current_event", None)

reactive_bls = opencog_reactive_automation_bindlinks(gb_current_event)

while(1):
    msg = queue_recv.get()
    if (msg["type"] != "event"):
        continue
    if (msg["event"]["event_type"] == "state_changed"):
        home_state.state_changed_event_handler(msg["event"])
                                    
    gb_current_event.set_object(Event(msg["event"]))
    
    for bl in reactive_bls:        
        execute_atom(atomspace, bl)
