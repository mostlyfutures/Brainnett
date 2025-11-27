# PRP: Brainnet Blue GUI Launcher

## Goal
Create a "singular word command" CLI tool that launches a simple, idiot-proof, blue-themed GUI window. This GUI will allow users to select a trading symbol (BTC, ETH, ES, NQ) or Exit via clickable buttons.

## Why
- **User Experience**: Non-technical users (or tired quants) need a zero-friction way to launch the trading system without remembering CLI arguments.
- **Simplicity**: A visual menu reduces input errors.
- **Accessibility**: "Idiot-proof" design ensures anyone can operate the brain.

## What
- A new CLI command `dashboard` (or `gui`) added to the `brainnet` entry point.
- When run, it opens a `tkinter` window (standard library, no extra deps).
- **Visual Style**: Heavy use of blue.
- **Interactivity**:
    - Buttons for: `BTC`, `ETH`, `ES`, `NQ`.
    - Button for: `EXIT`.
- **Behavior**: Clicking a symbol button closes the GUI and launches the trading loop/analysis for that symbol in the terminal (or prints the command to run if full integration is complex, but better to run it).

### Success Criteria
- [ ] Running `brainnet gui` (or similar) opens the window.
- [ ] Window background is blue.
- [ ] Buttons are clearly visible and labeled.
- [ ] Clicking a symbol triggers the corresponding action (e.g., runs `gaf_trading_loop` logic).
- [ ] Clicking Exit closes the application cleanly.

## All Needed Context

### Documentation & References
- `tkinter` documentation (standard library).
- `brainnet/services/main.py`: Current CLI entry point.
- `examples/gaf_trading_loop.py`: Logic to be triggered.

### Current Codebase tree
```bash
brainnet/
├── services/
│   ├── main.py          # Entrypoint
│   └── ...
├── agents/              # Core logic
└── ...
```

### Desired Codebase tree
```bash
brainnet/
├── services/
│   ├── main.py          # Modified to include 'gui' command
│   └── gui.py           # NEW: GUI implementation using tkinter
└── ...
```

### Known Gotchas
- `tkinter` must run in the main thread.
- MacOS `tkinter` sometimes has issues with background colors on buttons (use `ttk` or frames if needed, or accept system styling limits, but try to force blue).
- Running the trading loop *after* the GUI closes is safer than threading *inside* the GUI for this simple launcher use case.

## Implementation Blueprint

### Data models and structure
No complex data models needed. Just a mapping of Button Text -> Symbol.

### Task List

```yaml
Task 1:
CREATE brainnet/services/gui.py:
  - Implement `launch_selector_gui()` function.
  - Setup `tkinter.Tk` window.
  - Configure blue background style.
  - Add buttons for BTC, ETH, ES, NQ, EXIT.
  - Return the selected symbol (or None if exit).

Task 2:
MODIFY brainnet/services/main.py:
  - Import `launch_selector_gui`.
  - Add `@cli.command()` named `gui`.
  - In the command handler:
    1. Call `launch_selector_gui()`.
    2. If symbol returned:
       - Call `run_trading_loop(symbol=...)` (import from examples or refactor logic).
       - Note: `examples/gaf_trading_loop.py` logic might need to be moved to `brainnet/services/trading.py` or imported directly if possible.
       - For now, we can import `run_trading_loop` from `examples.gaf_trading_loop` if it's in the python path, or better, move the core loop to `brainnet/core/engine.py` or similar to be proper.
       - *Decision*: To keep it simple as requested, we will dynamically import or assume `examples` is reachable, OR refactor `run_trading_loop` to `brainnet/services/engine.py`. Refactoring is better for a "production" system.
       - *Refined Task 2*: Move `run_trading_loop` logic to `brainnet/services/engine.py` to make it importable by both the example and the GUI.

Task 3:
REFACTOR:
  - Move `run_trading_loop` from `examples/gaf_trading_loop.py` to `brainnet/services/engine.py`.
  - Update `examples/gaf_trading_loop.py` to import it from there.

Task 4:
Validation:
  - Verify `brainnet gui` opens window.
  - Verify selection triggers loop.
```

### Pseudocode

```python
# brainnet/services/gui.py

import tkinter as tk
from typing import Optional

def launch_selector_gui() -> Optional[str]:
    selection = None
    
    def set_selection(symbol):
        nonlocal selection
        selection = symbol
        root.destroy()

    root = tk.Tk()
    root.title("Brainnet Launcher")
    root.configure(bg="blue")
    root.geometry("400x300")

    # Add Header "Select Market"
    
    # Add Buttons
    markets = {"BTC": "BTC-USD", "ETH": "ETH-USD", "ES": "ES=F", "NQ": "NQ=F"}
    for label, symbol in markets.items():
        btn = tk.Button(root, text=label, command=lambda s=symbol: set_selection(s))
        btn.pack(pady=5)
        
    # Exit Button
    
    root.mainloop()
    return selection
```

## Validation Loop

### Level 1: Syntax & Style
```bash
ruff check brainnet/services/gui.py
```

### Level 2: Unit Tests
```python
# Can't easily unit test GUI in headless env, but can test the refactored engine logic.
```

### Level 3: Integration
```bash
python -m brainnet.services.main gui
# Verify window opens and clicks work.
```

## Final validation Checklist
- [ ] `brainnet gui` opens blue window
- [ ] Buttons BTC, ETH, ES, NQ present
- [ ] Selection runs the trading loop
- [ ] Exit button works

