#!/usr/bin/env python
"""Echo server using the asyncio API."""

import argparse
import asyncio
import os
import json

import logging

logger = logging.getLogger(__name__)

from websockets.asyncio.server import broadcast, serve

DEFAULT_HOST = os.environ.get('CCANVAS_WS_HOST', '0.0.0.0')
DEFAULT_PORT = int(os.environ.get('CCANVAS_WS_PORT', 8765))

connected = dict()

async def handler(websocket):
    async for message in websocket:
        if message.startswith('JOIN '):
            joining_user = message[5:]
            connected[joining_user] = websocket
            other_connected = {k:v for k, v in connected.items() if k != joining_user}
            broadcast(other_connected.values(), f'JOINED {joining_user}')
            other_usernames = json.dumps(list(other_connected.keys()), ensure_ascii=False)
            await websocket.send(f'OK JOIN {other_usernames}')
        elif message.startswith('LEAVE '):
            leaving_user = message[6:]
            assert websocket == connected.pop(leaving_user), 'strange, was expecting to find myselfe here...'
            broadcast(connected.values(), f'LEAVED {leaving_user}')
        else:
            await websocket.send(f'UNKNOWN command {message}')

async def main(host=DEFAULT_HOST,port=DEFAULT_PORT):
    async with serve(handler, host, port) as server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog='CCanvas WebSocket Backend',
            description=__doc__,
            epilog='Text at the bottom of help')
    parser.add_argument('-n', '--host', default=DEFAULT_HOST)
    parser.add_argument('-p', '--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    
    # TODO: add logging otions to command line interface
    logging.basicConfig(level=logging.DEBUG)
    logger.info('Started')
    asyncio.run(main(args.host, args.port))