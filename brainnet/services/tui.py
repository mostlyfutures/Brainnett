"""
Brainnet Terminal UI (TUI) - Blue-themed terminal menu
Simple, idiot-proof interface using ANSI colors
"""

import sys
import os
from typing import Optional

# ANSI color codes for BLUE theme
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Blues
    BG_DARK = "\033[48;2;10;22;40m"      # Deep navy
    BG_MEDIUM = "\033[48;2;26;39;68m"    # Medium blue
    BG_HIGHLIGHT = "\033[48;2;42;63;95m" # Highlight blue
    
    # Text colors
    CYAN = "\033[38;2;0;212;255m"        # Accent cyan
    WHITE = "\033[38;2;255;255;255m"
    GRAY = "\033[38;2;136;153;170m"
    RED = "\033[38;2;255;68;102m"
    GREEN = "\033[38;2;0;255;136m"
    YELLOW = "\033[38;2;255;212;0m"


# Market options
MARKETS = [
    ("1", "BTC", "BTC-USD", "Bitcoin"),
    ("2", "ETH", "ETH-USD", "Ethereum"),
    ("3", "ES", "ES=F", "S&P 500 Futures"),
    ("4", "NQ", "NQ=F", "Nasdaq Futures"),
]


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_colored(text: str, end: str = "\n"):
    """Print text with ANSI colors."""
    print(text, end=end)


def draw_box(width: int = 60):
    """Draw a box border."""
    return "═" * width


def launch_tui() -> Optional[str]:
    """
    Launch the terminal UI menu.
    
    Returns:
        Selected symbol string or None if exited
    """
    clear_screen()
    
    c = Colors
    width = 60
    
    # Header
    print()
    print(f"{c.CYAN}{c.BOLD}  ╔{'═' * (width-4)}╗{c.RESET}")
    print(f"{c.CYAN}{c.BOLD}  ║{' ' * ((width-18)//2)}🧠 BRAINNET{' ' * ((width-17)//2)}║{c.RESET}")
    print(f"{c.CYAN}{c.BOLD}  ╚{'═' * (width-4)}╝{c.RESET}")
    print()
    print(f"{c.GRAY}     Adaptive Quant Trading System{c.RESET}")
    print(f"{c.GRAY}     Powered by Phi-3.5-Mini • GAF Pattern Recognition{c.RESET}")
    print()
    print(f"{c.CYAN}  {'─' * (width-4)}{c.RESET}")
    print()
    print(f"{c.WHITE}{c.BOLD}     SELECT A MARKET TO ANALYZE:{c.RESET}")
    print()
    
    # Market options
    for key, short, symbol, name in MARKETS:
        print(f"{c.CYAN}     [{c.WHITE}{c.BOLD}{key}{c.RESET}{c.CYAN}]{c.RESET}  {c.WHITE}{c.BOLD}{short:4}{c.RESET}  {c.GRAY}─  {name}{c.RESET}")
        print()
    
    print(f"{c.CYAN}  {'─' * (width-4)}{c.RESET}")
    print()
    print(f"{c.RED}     [Q]{c.RESET}  {c.RED}EXIT{c.RESET}")
    print()
    print(f"{c.CYAN}  {'─' * (width-4)}{c.RESET}")
    print()
    
    # Input prompt
    print(f"{c.CYAN}{c.BOLD}  ▶ Enter choice (1-4 or Q): {c.RESET}", end="")
    
    try:
        choice = input().strip().upper()
    except (KeyboardInterrupt, EOFError):
        print()
        return None
    
    # Process choice
    if choice == "Q" or choice == "EXIT":
        clear_screen()
        print(f"\n{c.CYAN}  👋 Goodbye!{c.RESET}\n")
        return None
    
    # Find selected market
    for key, short, symbol, name in MARKETS:
        if choice == key or choice == short:
            clear_screen()
            print(f"\n{c.GREEN}  ✓ Selected: {c.WHITE}{c.BOLD}{short}{c.RESET} {c.GRAY}({symbol}){c.RESET}")
            print(f"{c.CYAN}  Starting analysis...{c.RESET}\n")
            return symbol
    
    # Invalid choice - recurse
    print(f"\n{c.RED}  ✗ Invalid choice. Try again.{c.RESET}")
    import time
    time.sleep(1)
    return launch_tui()


def launch_tui_interactive() -> Optional[str]:
    """
    Launch interactive TUI with arrow key navigation.
    Falls back to simple menu if curses unavailable.
    """
    try:
        return _launch_curses_tui()
    except Exception:
        return launch_tui()


def _launch_curses_tui() -> Optional[str]:
    """Curses-based TUI with arrow key navigation."""
    import curses
    
    def main(stdscr):
        # Setup
        curses.start_color()
        curses.use_default_colors()
        curses.curs_set(0)  # Hide cursor
        
        # Define color pairs (blue theme)
        curses.init_pair(1, 51, -1)   # Cyan on default
        curses.init_pair(2, 15, -1)   # White on default
        curses.init_pair(3, 244, -1)  # Gray on default
        curses.init_pair(4, 196, -1)  # Red on default
        curses.init_pair(5, 51, 236)  # Cyan on dark blue (selected)
        curses.init_pair(6, 46, -1)   # Green on default
        
        CYAN = curses.color_pair(1)
        WHITE = curses.color_pair(2)
        GRAY = curses.color_pair(3)
        RED = curses.color_pair(4)
        SELECTED = curses.color_pair(5) | curses.A_BOLD
        GREEN = curses.color_pair(6)
        
        options = MARKETS + [("Q", "EXIT", None, "Quit")]
        current = 0
        
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            # Header
            title = "🧠 BRAINNET"
            subtitle = "Adaptive Quant Trading System"
            
            stdscr.attron(CYAN | curses.A_BOLD)
            stdscr.addstr(2, (width - len(title)) // 2, title)
            stdscr.attroff(CYAN | curses.A_BOLD)
            
            stdscr.attron(GRAY)
            stdscr.addstr(4, (width - len(subtitle)) // 2, subtitle)
            stdscr.attroff(GRAY)
            
            # Instructions
            inst = "SELECT A MARKET (↑↓ to navigate, ENTER to select)"
            stdscr.attron(WHITE)
            stdscr.addstr(7, (width - len(inst)) // 2, inst)
            stdscr.attroff(WHITE)
            
            # Options
            start_y = 10
            for i, (key, short, symbol, name) in enumerate(options):
                y = start_y + i * 2
                
                if i == current:
                    # Selected item
                    line = f"  ▶ [{key}] {short:6} - {name}  "
                    x = (width - len(line)) // 2
                    stdscr.attron(SELECTED)
                    stdscr.addstr(y, x, line)
                    stdscr.attroff(SELECTED)
                else:
                    # Normal item
                    if short == "EXIT":
                        stdscr.attron(RED)
                        line = f"    [{key}] {short:6} - {name}"
                        stdscr.addstr(y, (width - len(line)) // 2, line)
                        stdscr.attroff(RED)
                    else:
                        line = f"    [{key}] {short:6} - {name}"
                        stdscr.attron(WHITE)
                        stdscr.addstr(y, (width - len(line)) // 2, line)
                        stdscr.attroff(WHITE)
            
            # Footer
            footer = "Powered by Phi-3.5-Mini • GAF Pattern Recognition"
            stdscr.attron(GRAY)
            stdscr.addstr(height - 2, (width - len(footer)) // 2, footer)
            stdscr.attroff(GRAY)
            
            stdscr.refresh()
            
            # Handle input
            key = stdscr.getch()
            
            if key == curses.KEY_UP:
                current = (current - 1) % len(options)
            elif key == curses.KEY_DOWN:
                current = (current + 1) % len(options)
            elif key in [curses.KEY_ENTER, 10, 13]:
                _, short, symbol, _ = options[current]
                return symbol
            elif key == ord('q') or key == ord('Q'):
                return None
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                idx = key - ord('1')
                if idx < len(MARKETS):
                    return MARKETS[idx][2]
    
    return curses.wrapper(main)


if __name__ == "__main__":
    selected = launch_tui()
    if selected:
        print(f"Selected: {selected}")
    else:
        print("Exited")

