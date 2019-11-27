import asyncio
import websockets
import json
from threading import Thread
from concurrent.futures import ThreadPoolExecutor


class HassCommunicator:
    def __init__(self, uri, token, queue_send, queue_recv):
        self.msg_id    = 1
        self.queue_send = queue_send
        self.queue_recv = queue_recv
        
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.async_hass_communicator(uri, token))
        
    async def async_hass_communicator(self, uri, token):
        self.socket = await websockets.connect(uri)
        msg = json.loads(await self.socket.recv())
        assert msg["type"] == "auth_required", "Error 1"        
        
        await self.socket.send(json.dumps({"type": "auth", "access_token": token}))
        msg = json.loads(await self.socket.recv())
        assert msg["type"] == "auth_ok", "Access denied. Probably token is bad"
        
        await self.send_message_with_id({"type": "get_states"})
        msg = json.loads(await self.socket.recv())
        assert msg["type"] == "result" and msg["success"] == True, "Fail to fetch state"
        self.queue_recv.put(msg)
        
        await self.send_message_with_id({"type": "subscribe_events"})
        msg = json.loads(await self.socket.recv())
        assert msg["type"] == "result" and msg["success"] == True, "Fail to subscribe to events"
        
        await asyncio.gather(asyncio.ensure_future(self.async_wait_messages()), asyncio.ensure_future(self.async_send_messages()))

    async def send_message_with_id(self, msg):
        msg["id"] = str(self.msg_id)                    
        await self.socket.send(json.dumps(msg))                
        self.msg_id += 1

    async def async_wait_messages(self):
        while(1):
            msg = await self.socket.recv()
            # put will not block
            self.queue_recv.put(json.loads(msg))
        
    async def async_send_messages(self):
        pool = ThreadPoolExecutor(max_workers=1)        
        while(1):
            msg = await self.loop.run_in_executor(pool, self.queue_send.get)
            print("send message:", msg)
            await self.send_message_with_id(msg)

def start_HassCommunicator_in_thread(uri, token, queue_send, queue_recv):
    thread = Thread(target = lambda: HassCommunicator(uri, token, queue_send, queue_recv), daemon = True)
    thread.start()
    return thread

