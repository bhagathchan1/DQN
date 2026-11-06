"""Exercise three real Atari matches through HTTP/WebSocket and record model lineage.

Start `python -m dqn_project.pong_server` first. These are scripted controls for
execution verification, not measured human performance or a learned-skill claim.
"""
import argparse
import asyncio
import json
from pathlib import Path

import httpx
import websockets


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--output', type=Path, default=Path('tmp/pong_lab_smoke.json'))
    args = parser.parse_args()
    async with httpx.AsyncClient(base_url=args.url, timeout=90) as client:
        response = await client.post('/api/sessions', json={'config': {
            'match_seconds': 2, 'updates_per_match': 2, 'replay_capacity': 128,
            'min_replay': 8, 'batch_size': 4, 'seed': 1337,
        }})
        response.raise_for_status()
        snapshot = response.json()
        session_id = snapshot['id']
        versions = []
        websocket_url = args.url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/play'
        for index in range(3):
            response = await client.post('/api/matches')
            response.raise_for_status()
            async with websockets.connect(websocket_url, origin=args.url) as connection:
                async def heartbeat():
                    tick = 0
                    while True:
                        # Deliberately exercise both directions and serving.
                        keys = ['fire', 'up' if (tick // 8) % 2 == 0 else 'down']
                        await connection.send(json.dumps({'keys': keys}))
                        tick += 1
                        await asyncio.sleep(0.08)
                task = asyncio.create_task(heartbeat())
                try:
                    for _ in range(600):
                        await asyncio.sleep(0.1)
                        snapshot = (await client.get('/api/session')).json()
                        if snapshot['status'] not in {'playing', 'learning'}:
                            break
                    else:
                        raise RuntimeError('Match/learning phase timed out')
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            assert snapshot['status'] == 'summary', snapshot
            assert snapshot['last_result']['valid'], snapshot
            assert snapshot['training']['updates_completed'] == 2
            assert snapshot['active_version'] == index + 1
            assert snapshot['last_result']['policy_version'] == index
            versions.append(snapshot['active_version'])
            print(f"Game {index+1}: policy v{index} -> active v{index+1}, replay={snapshot['replay_size']}", flush=True)
        response = await client.post('/api/sessions', json={'session_id': session_id})
        response.raise_for_status()
        restored = response.json()
        assert restored['active_version'] == 3
        assert restored['replay_size'] == snapshot['replay_size']
        assert restored['cumulative_updates'] == 6
        assert len(restored['history']) == 3
        frame = await client.get('/api/frame.jpg')
        assert frame.status_code == 200 and frame.headers['content-type'] == 'image/jpeg'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'session_id': session_id, 'versions': versions,
                                         'restored': restored}, indent=2), encoding='utf-8')
        print(f'Restored v3 with six cumulative updates. Evidence: {args.output}')


if __name__ == '__main__':
    asyncio.run(main())
