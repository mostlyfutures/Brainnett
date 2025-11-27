"""
Brainnet Terminal UI (TUI) - Blue-themed terminal interface

Two main user stories:
1. Quick Score View: Select market → Get score 0-100 (0=short, 100=long)
2. Detailed Stats: Press 2 for detailed analysis, 0 to return to menu

Performance optimizations:
- Cached model instances (ConvNeXt, ResearchAgent)
- Fast mode skips LLM calls for quick scoring
- Uses numpy vectorization for GAF features
"""

import sys
import os
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

# ============================================================================
# MODEL CACHE - Load once, use many times
# ============================================================================
_MODEL_CACHE = {
    "research_agent": None,
    "convnext": None,
    "reasoning_agent": None,
    "initialized": False,
}


def _get_cached_models(skip_llm: bool = True):
    """
    Get cached model instances for fast inference.
    
    First call loads models (~3-5s), subsequent calls are instant.
    """
    global _MODEL_CACHE
    
    if not _MODEL_CACHE["initialized"]:
        try:
            from brainnet.agents import ResearchAgent, _CONVNEXT_AVAILABLE
            
            # Cache ResearchAgent (lightweight)
            _MODEL_CACHE["research_agent"] = ResearchAgent()
            
            # Cache ConvNeXt (heavy - 28M params)
            if _CONVNEXT_AVAILABLE:
                from brainnet.agents import ConvNeXtPredictor
                _MODEL_CACHE["convnext"] = ConvNeXtPredictor(device="auto")
            
            # Only load ReasoningAgent if needed (requires LLM)
            if not skip_llm:
                from brainnet.agents import ReasoningAgent
                _MODEL_CACHE["reasoning_agent"] = ReasoningAgent()
            
            _MODEL_CACHE["initialized"] = True
        except Exception as e:
            print(f"Model cache init failed: {e}")
    
    return _MODEL_CACHE


# ANSI color codes for BLUE theme
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Blues
    BG_DARK = "\033[48;2;10;22;40m"
    BG_MEDIUM = "\033[48;2;26;39;68m"
    BG_HIGHLIGHT = "\033[48;2;42;63;95m"
    
    # Text colors
    CYAN = "\033[38;2;0;212;255m"
    WHITE = "\033[38;2;255;255;255m"
    GRAY = "\033[38;2;136;153;170m"
    RED = "\033[38;2;255;68;102m"
    GREEN = "\033[38;2;0;255;136m"
    YELLOW = "\033[38;2;255;212;0m"
    BLUE = "\033[38;2;100;149;237m"
    ORANGE = "\033[38;2;255;165;0m"


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    symbol: str
    score: int  # 0-100 (0=strong short, 50=neutral, 100=strong long)
    price: float
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float
    regime: str
    volatility: str
    trend_score: float
    momentum: float
    features: Dict[str, float]
    analysis_type: str  # "convnext", "llm", "ensemble"


# Market options
MARKETS = [
    ("1", "BTC", "BTC-USD", "Bitcoin"),
    ("2", "ETH", "ETH-USD", "Ethereum"),
    ("3", "ES", "ES=F", "S&P 500 Futures"),
    ("4", "NQ", "NQ=F", "Nasdaq Futures"),
    ("5", "SPY", "SPY", "S&P 500 ETF"),
    ("6", "QQQ", "QQQ", "Nasdaq 100 ETF"),
]


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_score_color(score: int) -> str:
    """Get color based on score value."""
    c = Colors
    if score >= 70:
        return c.GREEN
    elif score >= 55:
        return c.CYAN
    elif score >= 45:
        return c.YELLOW
    elif score >= 30:
        return c.ORANGE
    else:
        return c.RED


def get_score_label(score: int) -> str:
    """Get human-readable label for score."""
    if score >= 80:
        return "STRONG LONG"
    elif score >= 65:
        return "LONG"
    elif score >= 55:
        return "LEAN LONG"
    elif score >= 45:
        return "NEUTRAL"
    elif score >= 35:
        return "LEAN SHORT"
    elif score >= 20:
        return "SHORT"
    else:
        return "STRONG SHORT"


def draw_score_bar(score: int, width: int = 40) -> str:
    """Draw a visual score bar."""
    c = Colors
    filled = int(score / 100 * width)
    
    # Create gradient bar
    bar = ""
    for i in range(width):
        if i < filled:
            # Color based on position (red to green)
            if i < width * 0.3:
                bar += f"{c.RED}█{c.RESET}"
            elif i < width * 0.45:
                bar += f"{c.ORANGE}█{c.RESET}"
            elif i < width * 0.55:
                bar += f"{c.YELLOW}█{c.RESET}"
            elif i < width * 0.7:
                bar += f"{c.CYAN}█{c.RESET}"
            else:
                bar += f"{c.GREEN}█{c.RESET}"
        else:
            bar += f"{c.GRAY}░{c.RESET}"
    
    return bar


def draw_header():
    """Draw the main header."""
    c = Colors
    clear_screen()
    print()
    print(f"{c.CYAN}{c.BOLD}  ╔══════════════════════════════════════════════════════╗{c.RESET}")
    print(f"{c.CYAN}{c.BOLD}  ║                    🧠 BRAINNET                        ║{c.RESET}")
    print(f"{c.CYAN}{c.BOLD}  ║         Adaptive Quant Trading System                 ║{c.RESET}")
    print(f"{c.CYAN}{c.BOLD}  ╚══════════════════════════════════════════════════════╝{c.RESET}")
    print()


def run_analysis(symbol: str, name: str, fast_mode: bool = True) -> AnalysisResult:
    """
    Run market analysis and return score.
    
    Args:
        symbol: Trading symbol
        name: Display name
        fast_mode: Skip LLM calls for faster scoring (default True)
    
    Score interpretation:
    - 0-20: Strong Short signal
    - 20-35: Short signal
    - 35-45: Lean Short
    - 45-55: Neutral
    - 55-65: Lean Long
    - 65-80: Long signal
    - 80-100: Strong Long signal
    """
    c = Colors
    start_time = time.time()
    
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print(f"{c.WHITE}{c.BOLD}     Analyzing {name} ({symbol})...{c.RESET}")
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    
    # Get cached models (fast after first load)
    cache = _get_cached_models(skip_llm=fast_mode)
    research = cache.get("research_agent")
    convnext = cache.get("convnext")
    
    # Import yfinance
    try:
        import yfinance as yf
        from brainnet.agents import _CONVNEXT_AVAILABLE
    except ImportError as e:
        print(f"{c.RED}  ✗ Import error: {e}{c.RESET}")
        return AnalysisResult(
            symbol=symbol, score=50, price=0.0, direction="NEUTRAL",
            confidence=0.0, regime="unknown", volatility="unknown",
            trend_score=0.0, momentum=0.0, features={}, analysis_type="error"
        )
    
    # Re-initialize research agent if not cached
    if research is None:
        from brainnet.agents import ResearchAgent
        research = ResearchAgent()
    
    # Step 1: Fetch data
    step_start = time.time()
    print(f"{c.GRAY}     [1/3] Fetching market data...{c.RESET}", end="", flush=True)
    try:
        data = yf.download(symbol, period="1d", interval="5m", progress=False)
        if data.empty:
            raise ValueError("No data received")
        
        # Get latest price
        try:
            price = float(data['Close'].iloc[-1].iloc[0]) if hasattr(data['Close'].iloc[-1], 'iloc') else float(data['Close'].iloc[-1])
        except:
            price = float(data['Close'].iloc[-1])
        
        elapsed = time.time() - step_start
        print(f"\r{c.GREEN}     ✓ {len(data)} bars | ${price:,.2f} ({elapsed:.1f}s){c.RESET}")
    except Exception as e:
        print(f"\r{c.RED}     ✗ Data fetch failed: {e}{c.RESET}")
        return AnalysisResult(
            symbol=symbol, score=50, price=0.0, direction="NEUTRAL",
            confidence=0.0, regime="unknown", volatility="unknown",
            trend_score=0.0, momentum=0.0, features={}, analysis_type="error"
        )
    
    # Step 2: GAF features (fast - no LLM)
    step_start = time.time()
    print(f"{c.GRAY}     [2/3] Extracting GAF features...{c.RESET}", end="", flush=True)
    try:
        # Extract close prices
        close = data['Close'].values.flatten()[-100:]
        
        # Fast path: extract GAF features directly (no LLM call)
        features = research.extract_gaf_features(close)
        
        trend = features.get('trend_score', 0)
        momentum = features.get('momentum', 0)
        
        elapsed = time.time() - step_start
        print(f"\r{c.GREEN}     ✓ Trend: {trend:+.3f} | Mom: {momentum:+.3f} ({elapsed:.1f}s){c.RESET}")
    except Exception as e:
        print(f"\r{c.YELLOW}     ⚠ GAF warning: {e}{c.RESET}")
        features = {}
        trend = 0
        momentum = 0
    
    # Step 3: ConvNeXt prediction (uses cached model)
    regime = "unknown"
    volatility = "unknown"
    convnext_direction = "neutral"
    convnext_confidence = 0.0
    analysis_type = "gaf_features"
    
    if convnext is not None:
        step_start = time.time()
        print(f"{c.GRAY}     [3/3] ConvNeXt inference...{c.RESET}", end="", flush=True)
        try:
            gaf_rgb = research.generate_gaf_3channel(close)
            prediction = convnext.predict(gaf_rgb)
            
            regime = prediction.regime
            volatility = prediction.volatility
            convnext_direction = prediction.direction
            convnext_confidence = prediction.direction_confidence
            analysis_type = "convnext"
            
            elapsed = time.time() - step_start
            print(f"\r{c.GREEN}     ✓ {convnext_direction.upper()} ({convnext_confidence:.0%}) | {regime} ({elapsed:.1f}s){c.RESET}")
        except Exception as e:
            print(f"\r{c.YELLOW}     ⚠ ConvNeXt: {e}{c.RESET}")
    else:
        print(f"{c.GRAY}     [3/3] ConvNeXt not loaded, using features only{c.RESET}")
    
    # Calculate confidence from features (no LLM call in fast mode)
    if fast_mode:
        # Heuristic confidence based on signal strength
        signal_strength = abs(trend) + abs(momentum) + convnext_confidence
        confidence = min(0.95, 0.4 + signal_strength * 0.3)
    else:
        # Full LLM confidence (slow)
        try:
            reasoning = cache.get("reasoning_agent")
            if reasoning is None:
                from brainnet.agents import ReasoningAgent
                reasoning = ReasoningAgent()
            analysis = research.research(data)
            confidence = reasoning.compute_confidence(analysis.get('analysis', ''))
        except:
            confidence = 0.5
    
    # Calculate final score (0-100)
    # Base score from trend (maps -1..1 to 30..70)
    base_score = 50 + (trend * 20)
    
    # Adjust by momentum (±10 points)
    momentum_adj = momentum * 10
    
    # Adjust by ConvNeXt direction (±15 points)
    direction_adj = 0
    if convnext_direction == "bullish":
        direction_adj = 15 * convnext_confidence
    elif convnext_direction == "bearish":
        direction_adj = -15 * convnext_confidence
    
    # Combine
    score = base_score + momentum_adj + direction_adj
    score = max(0, min(100, int(score)))  # Clamp to 0-100
    
    # Determine final direction
    if score >= 55:
        direction = "LONG"
    elif score <= 45:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"
    
    print()
    
    return AnalysisResult(
        symbol=symbol,
        score=score,
        price=price,
        direction=direction,
        confidence=confidence,
        regime=regime,
        volatility=volatility,
        trend_score=trend,
        momentum=momentum,
        features=features,
        analysis_type=analysis_type,
    )


def display_score(result: AnalysisResult) -> str:
    """
    Display the score screen.
    
    Returns user choice: "0" for menu, "2" for details
    """
    c = Colors
    clear_screen()
    draw_header()
    
    score_color = get_score_color(result.score)
    label = get_score_label(result.score)
    
    # Symbol and price
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print(f"{c.WHITE}{c.BOLD}     {result.symbol}{c.RESET}  {c.GRAY}│{c.RESET}  {c.WHITE}${result.price:,.2f}{c.RESET}")
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    
    # Large score display
    print(f"{c.GRAY}     SHORT ◀────────────────────────────────────▶ LONG{c.RESET}")
    print(f"     {draw_score_bar(result.score, 50)}")
    print()
    
    # Score number
    print(f"     {score_color}{c.BOLD}")
    print(f"                    ┌─────────────┐")
    print(f"                    │    {result.score:3d}      │")
    print(f"                    └─────────────┘")
    print(f"{c.RESET}")
    
    # Label
    print(f"               {score_color}{c.BOLD}{label:^20}{c.RESET}")
    print()
    
    # Quick stats
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    
    # Direction indicator
    if result.direction == "LONG":
        dir_indicator = f"{c.GREEN}▲ LONG{c.RESET}"
    elif result.direction == "SHORT":
        dir_indicator = f"{c.RED}▼ SHORT{c.RESET}"
    else:
        dir_indicator = f"{c.YELLOW}◆ NEUTRAL{c.RESET}"
    
    print(f"     {c.GRAY}Signal:{c.RESET}     {dir_indicator}     {c.GRAY}│{c.RESET}  {c.GRAY}Confidence:{c.RESET} {c.WHITE}{result.confidence:.0%}{c.RESET}")
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    
    # Navigation options
    print(f"     {c.CYAN}[2]{c.RESET}  {c.WHITE}View Detailed Statistics{c.RESET}")
    print()
    print(f"     {c.CYAN}[0]{c.RESET}  {c.GRAY}Back to Menu{c.RESET}")
    print()
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    
    # Input
    print(f"{c.CYAN}{c.BOLD}  ▶ Enter choice (2 or 0): {c.RESET}", end="")
    
    try:
        choice = input().strip()
        return choice
    except (KeyboardInterrupt, EOFError):
        return "0"


def display_detailed_stats(result: AnalysisResult) -> str:
    """
    Display detailed statistics screen.
    
    Returns user choice: "0" for menu, "1" for back to score
    """
    c = Colors
    clear_screen()
    draw_header()
    
    score_color = get_score_color(result.score)
    
    print(f"{c.CYAN}  ═══════════════════════════════════════════════════════{c.RESET}")
    print(f"{c.WHITE}{c.BOLD}     DETAILED ANALYSIS: {result.symbol}{c.RESET}")
    print(f"{c.CYAN}  ═══════════════════════════════════════════════════════{c.RESET}")
    print()
    
    # Price & Score Summary
    print(f"     {c.GRAY}┌─────────────────────────────────────────────────┐{c.RESET}")
    print(f"     {c.GRAY}│{c.RESET}  {c.WHITE}Price:{c.RESET}        {c.CYAN}${result.price:>12,.2f}{c.RESET}                  {c.GRAY}│{c.RESET}")
    print(f"     {c.GRAY}│{c.RESET}  {c.WHITE}Score:{c.RESET}        {score_color}{result.score:>12}{c.RESET} / 100              {c.GRAY}│{c.RESET}")
    print(f"     {c.GRAY}│{c.RESET}  {c.WHITE}Signal:{c.RESET}       {score_color}{result.direction:>12}{c.RESET}                  {c.GRAY}│{c.RESET}")
    print(f"     {c.GRAY}│{c.RESET}  {c.WHITE}Confidence:{c.RESET}   {c.CYAN}{result.confidence:>11.1%}{c.RESET}                  {c.GRAY}│{c.RESET}")
    print(f"     {c.GRAY}└─────────────────────────────────────────────────┘{c.RESET}")
    print()
    
    # Market Regime
    print(f"     {c.WHITE}{c.BOLD}MARKET REGIME{c.RESET}")
    print(f"     {c.CYAN}──────────────────────────────────────────────{c.RESET}")
    
    regime_color = c.WHITE
    if result.regime == "trending":
        regime_color = c.GREEN
    elif result.regime == "volatile":
        regime_color = c.RED
    elif result.regime == "mean_reverting":
        regime_color = c.YELLOW
        
    vol_color = c.WHITE
    if result.volatility in ["calm", "normal"]:
        vol_color = c.GREEN
    elif result.volatility == "elevated":
        vol_color = c.YELLOW
    elif result.volatility == "explosive":
        vol_color = c.RED
    
    print(f"     Regime:      {regime_color}{result.regime:15}{c.RESET}")
    print(f"     Volatility:  {vol_color}{result.volatility:15}{c.RESET}")
    print()
    
    # GAF Features
    print(f"     {c.WHITE}{c.BOLD}GAF PATTERN ANALYSIS{c.RESET}")
    print(f"     {c.CYAN}──────────────────────────────────────────────{c.RESET}")
    
    # Trend score visualization
    trend = result.trend_score
    trend_bar = "█" * int(abs(trend) * 10 + 1)
    if trend > 0:
        print(f"     Trend:       {c.GREEN}{trend:+.4f}  {trend_bar}{c.RESET}")
    else:
        print(f"     Trend:       {c.RED}{trend:+.4f}  {trend_bar}{c.RESET}")
    
    # Momentum visualization
    mom = result.momentum
    mom_bar = "█" * int(abs(mom) * 10 + 1)
    if mom > 0:
        print(f"     Momentum:    {c.GREEN}{mom:+.4f}  {trend_bar}{c.RESET}")
    else:
        print(f"     Momentum:    {c.RED}{mom:+.4f}  {trend_bar}{c.RESET}")
    print()
    
    # Additional features
    if result.features:
        print(f"     {c.WHITE}{c.BOLD}RAW FEATURES{c.RESET}")
        print(f"     {c.CYAN}──────────────────────────────────────────────{c.RESET}")
        
        for key, value in list(result.features.items())[:8]:
            # Color based on value
            if key in ['trend_score', 'momentum', 'recent_strength']:
                val_color = c.GREEN if value > 0 else c.RED
            else:
                val_color = c.WHITE
            print(f"     {c.GRAY}{key:22}{c.RESET} {val_color}{value:+.4f}{c.RESET}")
        print()
    
    # Analysis method
    print(f"     {c.GRAY}Analysis: {result.analysis_type.upper()}{c.RESET}")
    print()
    
    # Navigation
    print(f"     {c.CYAN}[1]{c.RESET}  {c.WHITE}Back to Score View{c.RESET}")
    print()
    print(f"     {c.CYAN}[0]{c.RESET}  {c.GRAY}Back to Menu{c.RESET}")
    print()
    print(f"{c.CYAN}  ═══════════════════════════════════════════════════════{c.RESET}")
    print()
    
    # Input
    print(f"{c.CYAN}{c.BOLD}  ▶ Enter choice (1 or 0): {c.RESET}", end="")
    
    try:
        choice = input().strip()
        return choice
    except (KeyboardInterrupt, EOFError):
        return "0"


def display_menu() -> Optional[tuple]:
    """
    Display main menu and return selected market.
    
    Returns:
        Tuple of (symbol, name) or None if exit
    """
    c = Colors
    clear_screen()
    draw_header()
    
    print(f"{c.GRAY}     Powered by Phi-3.5-Mini • GAF • ConvNeXt{c.RESET}")
    print()
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    print(f"{c.WHITE}{c.BOLD}     SELECT A MARKET TO ANALYZE:{c.RESET}")
    print()
    
    # Market options
    for key, short, symbol, name in MARKETS:
        print(f"     {c.CYAN}[{c.WHITE}{c.BOLD}{key}{c.RESET}{c.CYAN}]{c.RESET}  {c.WHITE}{c.BOLD}{short:4}{c.RESET}  {c.GRAY}─  {name}{c.RESET}")
        print()
    
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    print(f"     {c.RED}[Q]{c.RESET}  {c.RED}EXIT{c.RESET}")
    print()
    print(f"{c.CYAN}  ─────────────────────────────────────────────────────{c.RESET}")
    print()
    
    # Input prompt
    print(f"{c.CYAN}{c.BOLD}  ▶ Enter choice (1-{len(MARKETS)} or Q): {c.RESET}", end="")
    
    try:
        choice = input().strip().upper()
    except (KeyboardInterrupt, EOFError):
        print()
        return None
    
    # Process choice
    if choice == "Q" or choice == "EXIT":
        return None
    
    # Find selected market
    for key, short, symbol, name in MARKETS:
        if choice == key or choice == short:
            return (symbol, name)
    
    # Invalid choice
    print(f"\n{c.RED}  ✗ Invalid choice. Try again.{c.RESET}")
    time.sleep(1)
    return display_menu()


def launch_tui() -> Optional[str]:
    """
    Launch the terminal UI with two user stories:
    1. Quick Score View
    2. Detailed Statistics
    
    Returns:
        Selected symbol string or None if exited
    """
    while True:
        # Story 1: Select market from menu
        selection = display_menu()
        
        if selection is None:
            clear_screen()
            c = Colors
            print(f"\n{c.CYAN}  👋 Goodbye!{c.RESET}\n")
            return None
        
        symbol, name = selection
        
        # Run analysis
        result = run_analysis(symbol, name)
        
        # Score/Details navigation loop
        while True:
            # Display score
            choice = display_score(result)
            
            if choice == "2":
                # Story 2: Detailed statistics
                while True:
                    detail_choice = display_detailed_stats(result)
                    
                    if detail_choice == "1":
                        # Back to score view
                        break
                    elif detail_choice == "0":
                        # Back to menu
                        break
                    else:
                        continue
                
                if detail_choice == "0":
                    break  # Back to menu
            
            elif choice == "0":
                # Back to menu
                break
            
            else:
                # Invalid, show score again
                continue


def launch_tui_interactive() -> Optional[str]:
    """
    Launch interactive TUI.
    Falls back to simple menu if issues occur.
    """
    try:
        return launch_tui()
    except Exception as e:
        c = Colors
        print(f"\n{c.RED}  Error: {e}{c.RESET}\n")
        return None


# Legacy compatibility
def _launch_curses_tui() -> Optional[str]:
    """Deprecated: Use launch_tui() instead."""
    return launch_tui()


if __name__ == "__main__":
    selected = launch_tui()
    if selected:
        print(f"Final selection: {selected}")
