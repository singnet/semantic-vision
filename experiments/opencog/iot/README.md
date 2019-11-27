## Introduction

Here we demonstrate how opencog can be used in the context of smart home
to manipulate iot devices. In this example we control devices via Home
assistant
([https://www.home-assistant.io/](https://www.home-assistant.io/))
local server. First you need to configure you local home assistant
server and get uri and access token (Long-Lived Access Tokens). You
will need to replace uri and token in config.cfg with your values.

#### Scenarios 
Here we consider very simple scenarios with reactive control on events.
In our setup we have the following devices:

- button which generates special events when it pressed once or
twice ("single" and "double") 
- door sensor. Binary sensor which can be open or closed and which
generate events when state is changed.
- the lamp which we can turn on or off

We would like to make the following toy automation.

- Then button is pressed once and door sensor is in the state "off"
(door is closed) we send "turn on" command to the lamp.
- The button is pressed twice and door sensor is in the state "off" we
send "turn off" command to the lamp.

#### Prototypes

We have two prototypes
- ```proto5_entities``` - without knowledge base
- ```proto6_KB``` - with knowledge base

They are quite similar (the difference only in ```main.py``` and ```opencog_reactive_automation_bindlinks.py```), but prototype with knowledge base have more
complicated opencog code, so you could start with prototype without
knowledge base.


#### Description of modules
 
In ```hass_communicator.py``` we implement ```HassCommunicator``` class for communication with Home
Assistant server. It exchange information with the rest of the code
via ```quest_send``` and ```queue_recv``` (because we run it in separate thread).  When it received a message from the
Home assistant it put it in ```queue_recv``` (and in the main loop in
main.py we wait for this massage). Simillary when someone need to send
a message to Home assistant he put it `queue_send` and
hass_communicator gets it from this queue and send it. It should be
noted that in the current prototype we send messages in entity.py in the
function send_simple_command, and this function is called from
GroundingObjectNode by opencog.

In ```entity.py``` we implement ```Entity``` class which represent some
entity in our smart house (button, sensors or a lamp), this class keep
state of a device and also can send commands to this device. In
opencog each device is represented via GroundedObjectNode of ```Entity```
class. 


In home_state.py we implement HomeState class which manage collection
of Enities.


In event.py we implement Event class which represent an event in
opencog (via GroundedObjectNode).

```In opencog_reactive_automation_bindlinks.py``` in
```opencog_reactive_automation_bindlinks``` function we define bindlinks
which actually define the reactive behavior of opencog on events
(this file is different for ```proto5_entities``` and ```proto6_KB```). 

In ```knowledge_base.py``` (only for ```proto6_KB```) we define the
toy knowledge base. 


