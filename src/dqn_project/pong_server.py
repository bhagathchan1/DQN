"""Local Pong Learning Lab server, with one isolated, long-lived session process."""
from __future__ import annotations

import argparse
import asyncio
import copy
import io
import json
import multiprocessing as mp
import queue
import re
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse


def _worker(root, commands, updates, replies, cancel):
    # Import torch in the child; neither inference nor learning blocks FastAPI.
    import torch
    from PIL import Image
    from .pong_session import PongSession

    torch.set_num_threads(1)
    session = None
    last_input = 0.0
    last_tick = time.monotonic()
    slow_since = None

    def publish():
        if session is None:
            return
        snapshot = session.snapshot()
        rgb = session.frame()
        frame = None
        inspector = None
        if rgb is not None:
            stream = io.BytesIO()
            Image.fromarray(rgb).save(stream, 'JPEG', quality=85)
            frame = stream.getvalue()
        if session.latest_state is not None:
            import numpy as np
            stream = io.BytesIO()
            Image.fromarray(np.concatenate(session.latest_state, axis=1)).save(stream, 'PNG')
            inspector = stream.getvalue()
        packet = (snapshot, frame, inspector)
        try:
            updates.put_nowait(packet)
        except queue.Full:
            try:
                updates.get_nowait()
            except queue.Empty:
                pass
            try:
                updates.put_nowait(packet)
            except queue.Full:
                pass

    try:
        while True:
            if session is not None and session.status == 'learning':
                # A callback publishes real optimizer metrics, including cancellation.
                session.learn(cancelled=cancel.is_set, on_update=lambda *args, **kwargs: publish())
                publish()
                last_tick = time.monotonic()
            try:
                command = commands.get(timeout=0.01)
            except queue.Empty:
                command = None
            if command:
                action = command['action']
                request_id = command.get('id')
                payload = command.get('payload', {})
                try:
                    if action == 'shutdown':
                        break
                    if action == 'create':
                        if session and session.status == 'playing':
                            raise ValueError('Finish or stop the current match before changing sessions.')
                        if session:
                            session.close()
                            session = None
                        cancel.clear()
                        session = PongSession(Path(root), config=payload.get('config'),
                                              session_id=payload.get('session_id'))
                    elif session is None:
                        raise ValueError('Create a session first.')
                    elif action == 'start':
                        session.start_match()
                        last_input = time.monotonic()
                        last_tick = last_input
                        slow_since = None
                    elif action == 'keys':
                        if session.status == 'playing':
                            session.set_keys(set(payload.get('keys', [])))
                            last_input = time.monotonic()
                    elif action == 'abort':
                        session.abort_match(payload.get('reason', 'aborted'))
                    elif action == 'retry':
                        cancel.clear()
                        session.retry_learning()
                    else:
                        raise ValueError(f'Unknown command: {action}')
                    if request_id:
                        replies.put((request_id, True, session.snapshot()))
                    publish()
                except Exception as exc:
                    traceback.print_exc()
                    if request_id:
                        replies.put((request_id, False, f'{type(exc).__name__}: {exc}'))
                    if session is None:
                        updates.put(({'status': 'no_session', 'error': str(exc)}, None, None))
                    elif session and session.status == 'playing':
                        session.abort_match('control_error')
                        publish()
            now = time.monotonic()
            if session is not None and session.status == 'playing':
                if now - last_input > 0.5:
                    session.abort_match('input_timeout')
                    publish()
                    continue
                if now - last_tick >= 1 / 15:
                    lag = now - last_tick
                    if lag > (1 / 15) / 0.90:
                        slow_since = slow_since or now
                    else:
                        slow_since = None
                    if slow_since is not None and now - slow_since > 5:
                        session.abort_match('invalid_throughput')
                        publish()
                        continue
                    last_tick = now
                    session.tick()
                    if session.status == 'learning':
                        cancel.clear()
                    publish()
    except BaseException as exc:
        traceback.print_exc()
        updates.put(({'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}, None, None))
    finally:
        if session:
            session.close()


class PongController:
    """Own the worker; keep snapshots and frame bytes available without polling it."""
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        context = mp.get_context('spawn')
        self.commands = context.Queue()
        self.updates = context.Queue(maxsize=4)
        self.replies = context.Queue()
        self.cancel = context.Event()
        self.process = None
        self.state = {'status': 'no_session'}
        self.frame_bytes = None
        self.inspector_bytes = None
        self.sequence = 0
        self.lock = threading.Lock()
        self.request_lock = threading.Lock()
        self._closed = False
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        while not self._closed:
            try:
                state, frame, inspector = self.updates.get(timeout=0.2)
            except (queue.Empty, OSError, EOFError):
                continue
            with self.lock:
                self.sequence += 1
                state['events_seq'] = self.sequence
                self.state = state
                self.frame_bytes = frame
                self.inspector_bytes = inspector

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.state)
        if self.process is not None and not self.process.is_alive() and result['status'] in {'playing', 'learning'}:
            result.update(status='interrupted', error='The session process stopped. Restore the last committed session.')
        return result

    def _start_worker(self):
        if self.process is not None and self.process.is_alive():
            return
        if self.process is not None:
            # A dead worker may have left stale commands/replies; use fresh queues.
            raise RuntimeError('Session worker exited. Restart the server to restore its last committed state.')
        self.process = mp.get_context('spawn').Process(
            target=_worker, args=(str(self.root), self.commands, self.updates, self.replies, self.cancel),
            name='pong-session', daemon=True)
        self.process.start()

    def request(self, action, payload=None, timeout=60):
        with self.request_lock:
            self._start_worker()
            request_id = uuid.uuid4().hex
            self.commands.put({'id': request_id, 'action': action, 'payload': payload or {}})
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                if not self.process.is_alive():
                    raise RuntimeError('Session worker exited; inspect server logs and restore the session.')
                try:
                    received, ok, value = self.replies.get(timeout=0.2)
                except queue.Empty:
                    continue
                if received != request_id:
                    continue
                if not ok:
                    raise ValueError(value)
                with self.lock:
                    self.sequence += 1
                    value['events_seq'] = self.sequence
                    self.state = value
                return value
            raise TimeoutError('The worker did not respond. Inspect the server logs before retrying.')

    def keys(self, keys):
        if self.snapshot().get('status') == 'playing':
            self.commands.put({'action': 'keys', 'payload': {'keys': list(keys)}})

    def stop_input(self, reason='disconnected'):
        if self.snapshot().get('status') == 'playing':
            self.commands.put({'action': 'abort', 'payload': {'reason': reason}})

    def cancel_learning(self):
        self.cancel.set()

    def sessions(self):
        result = []
        for path in self.root.glob('*/session.json'):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                result.append({'id': path.parent.name, 'status': data.get('status', 'saved'),
                               'active_version': data.get('active_version', 0),
                               'updated_at': data.get('updated_at', data.get('created_at'))})
            except (OSError, ValueError):
                continue
        return sorted(result, key=lambda row: row['id'], reverse=True)

    def checkpoints(self):
        return [{'path': str(p.resolve()), 'label': str(p.relative_to(self.root))}
                for p in self.root.glob('*/checkpoints/**/learner.pt')]

    def close(self):
        self.cancel.set()
        if self.process and self.process.is_alive():
            self.commands.put({'action': 'shutdown'})
            self.process.join(timeout=8)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)
        self._closed = True


def create_app(root=None, controller=None):
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, Response, StreamingResponse

    # FastAPI resolves annotations against module globals, including local imports.
    globals().update(Request=Request, WebSocket=WebSocket)
    project = Path(__file__).resolve().parents[2]
    owned = controller or PongController(root or project / 'experiments' / 'pong_sessions')

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(owned.close)

    app = FastAPI(title='Pong Learning Lab', lifespan=lifespan)
    app.state.controller = owned

    def local_origin(origin):
        return not origin or urlparse(origin).hostname in {'127.0.0.1', 'localhost', '::1'}

    @app.middleware('http')
    async def origin_guard(request: Request, call_next):
        if request.url.hostname not in {'127.0.0.1', 'localhost', '::1', 'testserver'} or not local_origin(request.headers.get('origin')):
            return Response('This server accepts local requests only.', status_code=403)
        return await call_next(request)

    async def command(action, payload=None):
        try:
            return await asyncio.to_thread(owned.request, action, payload)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(504, str(exc)) from exc

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'worker_alive': bool(owned.process and owned.process.is_alive())}

    @app.get('/api/session')
    def snapshot():
        return owned.snapshot()

    @app.get('/api/sessions')
    def sessions():
        return owned.sessions()

    @app.get('/api/checkpoints')
    def checkpoints():
        return owned.checkpoints()

    @app.post('/api/sessions')
    async def create(request: Request):
        if owned.snapshot().get('status') in {'playing', 'learning'}:
            raise HTTPException(409, 'Finish or cancel the active match/learning phase first.')
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(422, 'Expected a JSON object.') from exc
        if not isinstance(body, dict):
            raise HTTPException(422, 'Expected a JSON object.')
        session_id = body.get('session_id')
        if session_id is not None and (not isinstance(session_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', session_id)):
            raise HTTPException(422, 'Invalid session ID.')
        if session_id and not (owned.root / session_id / 'session.json').is_file():
            raise HTTPException(404, 'Saved session not found.')
        return await command('create', {'config': body.get('config'), 'session_id': session_id})

    @app.post('/api/matches')
    async def start():
        return await command('start')

    @app.post('/api/matches/abort')
    async def abort():
        return await command('abort')

    @app.post('/api/training/retry')
    async def retry():
        return await command('retry')

    @app.post('/api/training/cancel')
    def cancel_learning():
        owned.cancel_learning()
        return {'status': 'cancelling'}

    @app.get('/api/history')
    def history():
        return owned.snapshot().get('history', [])

    @app.get('/api/frame.jpg')
    def frame():
        if owned.frame_bytes is None:
            return Response(status_code=204)
        return Response(owned.frame_bytes, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @app.get('/api/inspector.png')
    def inspector():
        if getattr(owned, 'inspector_bytes', None) is None:
            return Response(status_code=204)
        return Response(owned.inspector_bytes, media_type='image/png', headers={'Cache-Control': 'no-store'})

    @app.get('/api/events')
    async def events(request: Request):
        async def stream():
            previous = None
            while not await request.is_disconnected():
                state = owned.snapshot()
                sequence = state.get('events_seq', 0)
                if sequence != previous:
                    yield f'id: {sequence}\ndata: {json.dumps(state, allow_nan=False)}\n\n'
                    previous = sequence
                else:
                    yield ': heartbeat\n\n'
                await asyncio.sleep(0.1)
        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache'})

    @app.websocket('/ws/play')
    async def play(websocket: WebSocket):
        if not local_origin(websocket.headers.get('origin')):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_json()
                if not isinstance(message, dict):
                    await websocket.close(code=1008)
                    break
                if message.get('blur'):
                    owned.stop_input('focus_lost')
                    continue
                keys = message.get('keys', [])
                if not isinstance(keys, list) or len(keys) > 3 or any(key not in {'up', 'down', 'fire'} for key in keys):
                    await websocket.close(code=1008)
                    break
                owned.keys(keys)
        except (WebSocketDisconnect, ValueError):
            pass
        finally:
            owned.stop_input('disconnected')

    dist = project / 'web' / 'dist'

    @app.get('/{path:path}')
    def static(path: str):
        candidate = (dist / path).resolve()
        if not candidate.is_relative_to(dist.resolve()):
            raise HTTPException(404)
        if candidate.is_file():
            return FileResponse(candidate)
        if path.startswith(('api/', 'ws/')):
            raise HTTPException(404)
        if (dist / 'index.html').is_file():
            return FileResponse(dist / 'index.html')
        return Response('Build the UI first: cd web && npm install && npm run build', status_code=503)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.data_dir), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    mp.freeze_support()
    main()
