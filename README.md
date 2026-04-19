# Snake (Python)

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
python3 -m pip install -r requirements.txt         # inside venv you can use: pip install ...
python3 snake_game.py                               # inside venv: python snake_game.py also works
python3 train_dqn.py --episodes 2000                # headless DQN
python3 train_dqn.py --render --fps 90             # grid + HUD (ESC / window close = stop)
python3 app.py                                   # PyQt6: curves open on Start; View → Training curves anytime
```

No `python` on PATH (common on macOS): use `python3` / `pip3`, or activate the venv (it adds `python`).

### Qt GUI (`app.py`) — `Could not find the Qt platform plugin "cocoa"`

`app.py` sets `QT_PLUGIN_PATH` and `QT_QPA_PLATFORM_PLUGIN_PATH` before loading Qt (required on many macOS venv installs).

If it still aborts:

```bash
pip install --force-reinstall "PyQt6>=6.5" "PyQt6-Qt6>=6.5"
```

**`Library not loaded: @rpath/QtWidgets.framework` / missing `PyQt6/Qt6/lib`:** the Qt runtime wheel (`PyQt6-Qt6`) did not unpack fully or was removed. Same venv:

```bash
pip install --force-reinstall --no-cache-dir PyQt6 PyQt6-Qt6
# worst case:
pip uninstall PyQt6 PyQt6-Qt6 PyQt6-sip -y && pip install PyQt6 PyQt6-Qt6
```

`app.py` checks for `Qt6/lib` before import and prints this hint if broken.

Run `python3 app.py` only after `source .venv/bin/activate` (same venv you used for `pip install`).

## RL (DQN)

- `env/snake_env.py` — vector obs, actions: 0 straight, 1 right, 2 left (relative to heading)
- `agent/dqn.py` — MLP + replay + target net (`DQNConfig`, `save` / `load`)
- Shared rules: `env/snake_core.py`, grid: `env/config.py`
- `app.py` + `gui/` — Qt GUI: hyperparameter sliders, train board, **View → Training curves** (second window), Play tab loads `models/*.pt`, auto-save checkbox after training

## Controls

- Arrows: move
- Space: restart after game over
- Esc: quit

## Learn (short)

| Topic | Level |
|-------|--------|
| Game loop: events → update → draw | know |
| `pygame.time.Clock.tick(FPS)` | know |
| Grid + cell coords vs pixels | know |
| No 180° turn: compare to `OPPOSITE` | surface |
