# PRP: ConvNeXt-Tiny GAF Pattern Recognition Integration

**Version**: 1.0  
**Date**: 2025-11-27  
**Status**: Ready for Implementation

---

## Goal

Integrate ConvNeXt-Tiny as a dedicated computer vision backbone for GAF (Gramian Angular Field) pattern recognition, enabling direct neural network-based prediction of market regimes, trend direction, and volatility signals—complementing the existing Phi-3.5-Mini LLM analysis pipeline.

## Why

- **Performance**: ConvNeXt-Tiny provides ~5-8ms inference on A100/4090, enabling real-time high-frequency analysis vs. 50-200ms LLM inference
- **Proven Alpha**: 15%+ Sharpe boost demonstrated in S&P500 backtests vs. vanilla CNNs
- **Fine-grained Features**: Modernized ResNet architecture excels at capturing transitions, lags, and texture patterns in GAF images
- **Transfer Learning**: Pretrained on ImageNet-22K provides robust feature extraction for financial patterns
- **Ensemble Potential**: CNN predictions can be combined with LLM analysis for higher confidence decisions
- **Scalability**: Batch inference allows multi-symbol screening without latency explosion

## What

### User-Visible Behavior
1. New `ConvNeXtPredictor` agent that processes GAF images and outputs:
   - Market regime classification (trending/mean-reverting/volatile/quiet)
   - Directional bias with confidence (bullish/bearish/neutral)
   - Volatility forecast (calm/normal/elevated/explosive)
2. Enhanced 3-channel GAF image generation (224x224):
   - Red channel: GASF (Gramian Angular Summation Field)
   - Green channel: GADF (Gramian Angular Difference Field)
   - Blue channel: Raw price heatmap
3. Optional LSTM head for multi-step predictions
4. Integration with existing `ResearchAgent` and trading loop
5. New CLI flag: `--use-convnext` for ConvNeXt-enhanced analysis

### Technical Requirements
- PyTorch with torchvision for ConvNeXt-Tiny
- 224x224 input images (ImageNet-compatible)
- ~28M parameters, ~150MB model size
- GPU inference preferred, CPU fallback supported

### Success Criteria
- [ ] ConvNeXt-Tiny loads and runs inference on GAF images
- [ ] 3-channel GAF generation produces valid 224x224 RGB images
- [ ] Prediction latency < 50ms on GPU, < 500ms on CPU
- [ ] Integration with existing trading loop works without breaking changes
- [ ] Unit tests pass with >90% coverage on new modules
- [ ] Backtesting shows non-degraded performance vs. LLM-only baseline

---

## All Needed Context

### Documentation & References
```yaml
- url: https://pytorch.org/vision/stable/models/convnext.html
  why: Official ConvNeXt model API, weight loading, and preprocessing requirements
  
- url: https://arxiv.org/abs/2201.03545
  why: "A ConvNet for the 2020s" paper - architecture details and design rationale
  
- url: https://pyts.readthedocs.io/en/stable/modules/image.html
  why: pyts GAF transformation API already used in research.py
  
- file: brainnet/agents/research.py
  why: Existing GAF generation pattern to extend for 3-channel output
  
- file: brainnet/agents/base.py
  why: BaseAgent pattern to follow for new ConvNeXtPredictor agent
  
- file: brainnet/services/engine.py
  why: Integration point for trading loop - where ConvNeXt predictions will be consumed
```

### Current Codebase Tree
```bash
brainnet/
├── agents/
│   ├── __init__.py
│   ├── base.py              # BaseAgent class, Phi35MiniClient
│   ├── coding.py            # CodingAgent
│   ├── reasoning.py         # ReasoningAgent with BSC confidence
│   └── research.py          # ResearchAgent with GAF generation ← MODIFY
├── core/
│   ├── config.py            # BrainnetConfig ← ADD ConvNeXt settings
│   ├── knowledge.py
│   └── memory.py
├── orchestrator/
│   ├── graph.py
│   └── router.py
├── services/
│   ├── engine.py            # Trading loop ← ADD ConvNeXt path
│   ├── main.py              # CLI entrypoint ← ADD --use-convnext flag
│   └── market_data.py
└── tests/
    ├── test_agents.py       # ← ADD ConvNeXt tests
    └── test_convnext.py     # ← NEW
```

### Desired Codebase Tree (additions highlighted)
```bash
brainnet/
├── agents/
│   ├── __init__.py          # ← UPDATE exports
│   ├── base.py
│   ├── coding.py
│   ├── convnext_predictor.py # ← NEW: ConvNeXt model wrapper
│   ├── reasoning.py
│   └── research.py          # ← MODIFY: add 3-channel GAF
├── core/
│   ├── config.py            # ← MODIFY: add ConvNeXt config options
│   ├── knowledge.py
│   └── memory.py
├── models/                   # ← NEW: model weights cache directory
│   └── .gitkeep
├── orchestrator/
│   ├── graph.py
│   └── router.py
├── services/
│   ├── engine.py            # ← MODIFY: add ConvNeXt analysis path
│   ├── main.py              # ← MODIFY: add CLI flags
│   └── market_data.py
└── tests/
    ├── test_agents.py       # ← UPDATE
    └── test_convnext.py     # ← NEW: comprehensive ConvNeXt tests
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: torchvision ConvNeXt requires specific preprocessing
# - Images must be normalized with ImageNet mean/std
# - Input shape: (batch, 3, 224, 224) - channels first!
# - Weight loading requires internet on first run (caches to ~/.cache/torch)

# CRITICAL: pyts GAF output is float64 in range [-1, 1]
# - Must convert to uint8 [0, 255] for image
# - Must resize to 224x224 (pyts default is series_length x series_length)

# GOTCHA: Our ResearchAgent already flattens multi-index columns from yfinance
# - Follow same pattern in ConvNeXtPredictor

# GOTCHA: ConvNeXt pretrained weights are for 1000-class ImageNet
# - Must replace classifier head for our 4-class regime prediction
# - Or use as feature extractor + custom head

# PATTERN: All agents inherit from BaseAgent but ConvNeXt doesn't need LLM
# - Create ConvNeXtPredictor without inheriting BaseAgent (no LLM dependency)
```

---

## Implementation Blueprint

### Data Models and Structures

```python
# brainnet/agents/convnext_predictor.py

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

# Output schema for ConvNeXt predictions
@dataclass
class ConvNeXtPrediction:
    """Prediction output from ConvNeXt GAF analysis."""
    
    # Market regime classification
    regime: Literal["trending", "mean_reverting", "volatile", "quiet"]
    regime_confidence: float  # 0.0 - 1.0
    
    # Directional bias
    direction: Literal["bullish", "bearish", "neutral"]
    direction_confidence: float  # 0.0 - 1.0
    
    # Volatility forecast
    volatility: Literal["calm", "normal", "elevated", "explosive"]
    volatility_confidence: float  # 0.0 - 1.0
    
    # Raw logits for ensemble weighting
    regime_logits: np.ndarray  # (4,)
    direction_logits: np.ndarray  # (3,)
    volatility_logits: np.ndarray  # (4,)
    
    # Latent features for downstream tasks
    features: Optional[np.ndarray] = None  # (768,) ConvNeXt-Tiny feature dim
    
    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "direction": self.direction,
            "direction_confidence": self.direction_confidence,
            "volatility": self.volatility,
            "volatility_confidence": self.volatility_confidence,
        }


# Configuration extension for config.py
@dataclass 
class ConvNeXtConfig:
    """ConvNeXt-specific configuration."""
    enabled: bool = False
    model_name: str = "convnext_tiny"
    weights: str = "IMAGENET1K_V1"  # or path to fine-tuned weights
    device: str = "auto"  # auto, cuda, cpu
    batch_size: int = 1
    cache_dir: str = "models/"
```

### Task List (Implementation Order)

```yaml
Task 1: Add PyTorch dependencies to requirements.txt
  Status: pending
  Files: requirements.txt, pyproject.toml
  Details: Add torch, torchvision with appropriate version constraints
  
Task 2: Extend configuration for ConvNeXt settings
  Status: pending
  Files: brainnet/core/config.py
  Details: Add ConvNeXt config options (enabled, device, weights path)
  
Task 3: Create ConvNeXtPredictor agent module
  Status: pending
  Files: brainnet/agents/convnext_predictor.py
  Details: Core ConvNeXt model wrapper with prediction methods
  
Task 4: Enhance ResearchAgent with 3-channel GAF generation
  Status: pending
  Files: brainnet/agents/research.py
  Details: Add generate_gaf_3channel() method for 224x224 RGB images
  
Task 5: Update agents __init__.py exports
  Status: pending
  Files: brainnet/agents/__init__.py
  Details: Export ConvNeXtPredictor, ConvNeXtPrediction
  
Task 6: Integrate ConvNeXt into trading engine
  Status: pending
  Files: brainnet/services/engine.py
  Details: Add ConvNeXt analysis path alongside LLM analysis
  
Task 7: Add CLI flags for ConvNeXt mode
  Status: pending
  Files: brainnet/services/main.py
  Details: Add --use-convnext, --convnext-device flags
  
Task 8: Create comprehensive test suite
  Status: pending
  Files: tests/test_convnext.py
  Details: Unit tests for ConvNeXtPredictor and 3-channel GAF
  
Task 9: Create models directory with .gitkeep
  Status: pending
  Files: models/.gitkeep
  Details: Cache directory for model weights
```

---

## Task Implementation Details

### Task 1: Add PyTorch Dependencies

```python
# requirements.txt - ADD these lines after matplotlib:
torch>=2.1.0
torchvision>=0.16.0
scipy>=1.11.0  # For image resizing

# pyproject.toml - ADD to dependencies list:
"torch>=2.1.0",
"torchvision>=0.16.0",
"scipy>=1.11.0",
```

### Task 2: Configuration Extension

```python
# brainnet/core/config.py - ADD after existing BrainnetConfig class

@dataclass
class ConvNeXtConfig:
    """ConvNeXt model configuration."""
    enabled: bool = False
    model_name: str = "convnext_tiny"
    weights: str = "IMAGENET1K_V1"
    device: str = "auto"  # auto, cuda, mps, cpu
    fine_tuned_path: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "ConvNeXtConfig":
        load_dotenv()
        return cls(
            enabled=os.getenv("CONVNEXT_ENABLED", "false").lower() == "true",
            model_name=os.getenv("CONVNEXT_MODEL", "convnext_tiny"),
            weights=os.getenv("CONVNEXT_WEIGHTS", "IMAGENET1K_V1"),
            device=os.getenv("CONVNEXT_DEVICE", "auto"),
            fine_tuned_path=os.getenv("CONVNEXT_FINE_TUNED_PATH"),
        )

# UPDATE load_config() to include ConvNeXt settings
```

### Task 3: ConvNeXtPredictor Agent (Core Implementation)

```python
# brainnet/agents/convnext_predictor.py

"""
ConvNeXt-Tiny GAF Pattern Predictor

Modernized CNN backbone optimized for GAF image analysis.
Excels at capturing fine-grained textures like transitions, lags, and 
volatility clusters in financial time series transformed to GAF images.

Research basis:
- "A ConvNet for the 2020s" (Liu et al., 2022)
- 15%+ Sharpe improvement in S&P500 GAF backtests vs vanilla CNNs
"""

import os
from dataclasses import dataclass
from typing import Literal, Optional, Union
import numpy as np

# Lazy imports for torch to avoid import errors when torch not installed
_torch_available = None

def _check_torch():
    global _torch_available
    if _torch_available is None:
        try:
            import torch
            import torchvision
            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


@dataclass
class ConvNeXtPrediction:
    """Structured output from ConvNeXt GAF analysis."""
    regime: Literal["trending", "mean_reverting", "volatile", "quiet"]
    regime_confidence: float
    direction: Literal["bullish", "bearish", "neutral"]
    direction_confidence: float
    volatility: Literal["calm", "normal", "elevated", "explosive"]
    volatility_confidence: float
    regime_logits: np.ndarray
    direction_logits: np.ndarray
    volatility_logits: np.ndarray
    features: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "regime_confidence": round(self.regime_confidence, 4),
            "direction": self.direction,
            "direction_confidence": round(self.direction_confidence, 4),
            "volatility": self.volatility,
            "volatility_confidence": round(self.volatility_confidence, 4),
        }


class ConvNeXtPredictor:
    """
    ConvNeXt-Tiny based predictor for GAF pattern analysis.
    
    Uses pretrained ImageNet weights with custom classification heads
    for regime, direction, and volatility prediction.
    
    Input: 224x224 RGB GAF images (GASF-red, GADF-green, heatmap-blue)
    Output: ConvNeXtPrediction with regime, direction, volatility
    """
    
    REGIME_CLASSES = ["trending", "mean_reverting", "volatile", "quiet"]
    DIRECTION_CLASSES = ["bullish", "bearish", "neutral"]
    VOLATILITY_CLASSES = ["calm", "normal", "elevated", "explosive"]
    
    def __init__(
        self,
        device: str = "auto",
        weights: str = "IMAGENET1K_V1",
        fine_tuned_path: Optional[str] = None,
    ):
        if not _check_torch():
            raise ImportError(
                "PyTorch not installed. Run: pip install torch torchvision"
            )
        
        import torch
        import torchvision.models as models
        from torchvision.models import ConvNeXt_Tiny_Weights
        
        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Load pretrained ConvNeXt-Tiny
        if weights == "IMAGENET1K_V1":
            weight_enum = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        else:
            weight_enum = None
            
        self.backbone = models.convnext_tiny(weights=weight_enum)
        
        # Get feature dimension (768 for ConvNeXt-Tiny)
        self.feature_dim = self.backbone.classifier[2].in_features
        
        # Replace classifier with multi-head output
        # Remove original classifier
        self.backbone.classifier = torch.nn.Identity()
        
        # Create classification heads
        self.regime_head = torch.nn.Linear(self.feature_dim, len(self.REGIME_CLASSES))
        self.direction_head = torch.nn.Linear(self.feature_dim, len(self.DIRECTION_CLASSES))
        self.volatility_head = torch.nn.Linear(self.feature_dim, len(self.VOLATILITY_CLASSES))
        
        # Load fine-tuned weights if provided
        if fine_tuned_path and os.path.exists(fine_tuned_path):
            state_dict = torch.load(fine_tuned_path, map_location=self.device)
            self._load_custom_weights(state_dict)
        
        # Move to device and set eval mode
        self.backbone = self.backbone.to(self.device)
        self.regime_head = self.regime_head.to(self.device)
        self.direction_head = self.direction_head.to(self.device)
        self.volatility_head = self.volatility_head.to(self.device)
        
        self.backbone.eval()
        self.regime_head.eval()
        self.direction_head.eval()
        self.volatility_head.eval()
        
        # Store preprocessing transform
        self.preprocess = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        
    def _load_custom_weights(self, state_dict: dict):
        """Load fine-tuned weights for custom heads."""
        import torch
        if "regime_head" in state_dict:
            self.regime_head.load_state_dict(state_dict["regime_head"])
        if "direction_head" in state_dict:
            self.direction_head.load_state_dict(state_dict["direction_head"])
        if "volatility_head" in state_dict:
            self.volatility_head.load_state_dict(state_dict["volatility_head"])
        if "backbone" in state_dict:
            self.backbone.load_state_dict(state_dict["backbone"])
    
    def preprocess_gaf(self, gaf_image: np.ndarray) -> "torch.Tensor":
        """
        Preprocess GAF image for ConvNeXt input.
        
        Args:
            gaf_image: RGB image as numpy array (H, W, 3) or (3, H, W)
                      Values in [0, 255] uint8 or [0, 1] float
        
        Returns:
            Preprocessed tensor (1, 3, 224, 224)
        """
        import torch
        from PIL import Image
        
        # Ensure HWC format
        if gaf_image.ndim == 3 and gaf_image.shape[0] == 3:
            gaf_image = np.transpose(gaf_image, (1, 2, 0))
        
        # Convert to uint8 if float
        if gaf_image.dtype in [np.float32, np.float64]:
            if gaf_image.max() <= 1.0:
                gaf_image = (gaf_image * 255).astype(np.uint8)
            else:
                gaf_image = gaf_image.astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(gaf_image)
        
        # Apply torchvision transforms (resize, normalize)
        tensor = self.preprocess(pil_image)
        
        # Add batch dimension
        return tensor.unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def predict(
        self, 
        gaf_image: np.ndarray,
        return_features: bool = False,
    ) -> ConvNeXtPrediction:
        """
        Run prediction on a GAF image.
        
        Args:
            gaf_image: 224x224 RGB GAF image (numpy array)
            return_features: Whether to include latent features in output
            
        Returns:
            ConvNeXtPrediction with regime, direction, volatility predictions
        """
        import torch
        import torch.nn.functional as F
        
        # Preprocess
        x = self.preprocess_gaf(gaf_image)
        
        # Extract features
        features = self.backbone(x)  # (1, 768)
        
        # Run classification heads
        regime_logits = self.regime_head(features)  # (1, 4)
        direction_logits = self.direction_head(features)  # (1, 3)
        volatility_logits = self.volatility_head(features)  # (1, 4)
        
        # Softmax for probabilities
        regime_probs = F.softmax(regime_logits, dim=1)
        direction_probs = F.softmax(direction_logits, dim=1)
        volatility_probs = F.softmax(volatility_logits, dim=1)
        
        # Get predictions
        regime_idx = regime_probs.argmax(dim=1).item()
        direction_idx = direction_probs.argmax(dim=1).item()
        volatility_idx = volatility_probs.argmax(dim=1).item()
        
        return ConvNeXtPrediction(
            regime=self.REGIME_CLASSES[regime_idx],
            regime_confidence=regime_probs[0, regime_idx].item(),
            direction=self.DIRECTION_CLASSES[direction_idx],
            direction_confidence=direction_probs[0, direction_idx].item(),
            volatility=self.VOLATILITY_CLASSES[volatility_idx],
            volatility_confidence=volatility_probs[0, volatility_idx].item(),
            regime_logits=regime_logits.cpu().numpy().flatten(),
            direction_logits=direction_logits.cpu().numpy().flatten(),
            volatility_logits=volatility_logits.cpu().numpy().flatten(),
            features=features.cpu().numpy().flatten() if return_features else None,
        )
    
    @torch.no_grad()
    def predict_batch(
        self,
        gaf_images: list[np.ndarray],
    ) -> list[ConvNeXtPrediction]:
        """Batch prediction for multiple GAF images."""
        import torch
        import torch.nn.functional as F
        
        # Stack preprocessed images
        tensors = [self.preprocess_gaf(img) for img in gaf_images]
        x = torch.cat(tensors, dim=0)  # (N, 3, 224, 224)
        
        # Forward pass
        features = self.backbone(x)
        regime_logits = self.regime_head(features)
        direction_logits = self.direction_head(features)
        volatility_logits = self.volatility_head(features)
        
        # Softmax
        regime_probs = F.softmax(regime_logits, dim=1)
        direction_probs = F.softmax(direction_logits, dim=1)
        volatility_probs = F.softmax(volatility_logits, dim=1)
        
        # Build predictions
        predictions = []
        for i in range(len(gaf_images)):
            r_idx = regime_probs[i].argmax().item()
            d_idx = direction_probs[i].argmax().item()
            v_idx = volatility_probs[i].argmax().item()
            
            predictions.append(ConvNeXtPrediction(
                regime=self.REGIME_CLASSES[r_idx],
                regime_confidence=regime_probs[i, r_idx].item(),
                direction=self.DIRECTION_CLASSES[d_idx],
                direction_confidence=direction_probs[i, d_idx].item(),
                volatility=self.VOLATILITY_CLASSES[v_idx],
                volatility_confidence=volatility_probs[i, v_idx].item(),
                regime_logits=regime_logits[i].cpu().numpy(),
                direction_logits=direction_logits[i].cpu().numpy(),
                volatility_logits=volatility_logits[i].cpu().numpy(),
            ))
        
        return predictions
```

### Task 4: Enhanced 3-Channel GAF Generation

```python
# brainnet/agents/research.py - ADD this method to ResearchAgent class

def generate_gaf_3channel(
    self,
    series: np.ndarray,
    image_size: int = 224,
) -> np.ndarray:
    """
    Generate 3-channel GAF image for ConvNeXt analysis.
    
    Channels:
    - Red: GASF (Gramian Angular Summation Field)
    - Green: GADF (Gramian Angular Difference Field) 
    - Blue: Normalized price heatmap
    
    Args:
        series: 1D numpy array of price data
        image_size: Output image size (default 224 for ImageNet)
        
    Returns:
        RGB image as numpy array (H, W, 3) with values [0, 255] uint8
    """
    from scipy.ndimage import zoom
    
    # Ensure 2D for pyts
    if series.ndim == 1:
        series = series.reshape(1, -1)
    
    # Normalize to [-1, 1]
    series_min, series_max = series.min(), series.max()
    if series_max - series_min > 0:
        series_norm = 2 * (series - series_min) / (series_max - series_min) - 1
    else:
        series_norm = np.zeros_like(series)
    
    # Generate GAF matrices
    gaf_sum = self.gaf_summation.fit_transform(series_norm)[0]  # GASF
    gaf_diff = self.gaf_difference.fit_transform(series_norm)[0]  # GADF
    
    # Create price heatmap (outer product of normalized prices)
    price_norm = series_norm.flatten()
    heatmap = np.outer(price_norm, price_norm)
    
    # Normalize all to [0, 1] for image
    def normalize_channel(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min > 0:
            return (arr - arr_min) / (arr_max - arr_min)
        return np.zeros_like(arr)
    
    red = normalize_channel(gaf_sum)
    green = normalize_channel(gaf_diff)
    blue = normalize_channel(heatmap)
    
    # Resize to target size
    current_size = red.shape[0]
    if current_size != image_size:
        scale = image_size / current_size
        red = zoom(red, scale, order=1)
        green = zoom(green, scale, order=1)
        blue = zoom(blue, scale, order=1)
    
    # Stack channels and convert to uint8
    rgb = np.stack([red, green, blue], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb


def generate_gaf_3channel_base64(
    self,
    series: np.ndarray,
    image_size: int = 224,
) -> str:
    """Generate 3-channel GAF and return as base64 PNG."""
    from PIL import Image
    import base64
    from io import BytesIO
    
    rgb = self.generate_gaf_3channel(series, image_size)
    
    # Convert to PIL and encode
    img = Image.fromarray(rgb)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
```

### Task 5: Update Exports

```python
# brainnet/agents/__init__.py - REPLACE with:

from .base import BaseAgent, Phi35MiniClient
from .research import ResearchAgent
from .reasoning import ReasoningAgent
from .coding import CodingAgent

# Conditional ConvNeXt import (requires torch)
try:
    from .convnext_predictor import ConvNeXtPredictor, ConvNeXtPrediction
    _CONVNEXT_AVAILABLE = True
except ImportError:
    ConvNeXtPredictor = None
    ConvNeXtPrediction = None
    _CONVNEXT_AVAILABLE = False

__all__ = [
    "BaseAgent",
    "Phi35MiniClient",
    "ResearchAgent", 
    "ReasoningAgent",
    "CodingAgent",
    "ConvNeXtPredictor",
    "ConvNeXtPrediction",
    "_CONVNEXT_AVAILABLE",
]
```

### Task 6: Trading Engine Integration

```python
# brainnet/services/engine.py - ADD ConvNeXt analysis function

def run_convnext_analysis(
    symbol: str = "ES=F",
    interval: str = "5m",
    combine_with_llm: bool = True,
) -> dict:
    """
    Run ConvNeXt-enhanced GAF analysis.
    
    Args:
        symbol: Trading symbol
        interval: Data interval  
        combine_with_llm: Whether to also run LLM analysis for ensemble
        
    Returns:
        Dictionary with ConvNeXt predictions and optionally LLM analysis
    """
    from brainnet.agents import ResearchAgent, _CONVNEXT_AVAILABLE
    
    if not _CONVNEXT_AVAILABLE:
        raise ImportError("ConvNeXt requires PyTorch. Run: pip install torch torchvision")
    
    from brainnet.agents import ConvNeXtPredictor
    
    print(f"\n{'='*60}")
    print(f"  BRAINNET ConvNeXt ANALYSIS: {symbol}")
    print(f"{'='*60}")
    
    # Initialize
    research = ResearchAgent()
    convnext = ConvNeXtPredictor()
    
    # Fetch data
    print(f"\n[1/4] Fetching {symbol} data...")
    data = yf.download(symbol, period="1d", interval=interval, progress=False)
    
    if data.empty:
        return {"symbol": symbol, "error": "No data"}
    
    # Handle multi-index
    try:
        latest_price = float(data['Close'].iloc[-1].iloc[0]) if hasattr(data['Close'].iloc[-1], 'iloc') else float(data['Close'].iloc[-1])
    except:
        latest_price = float(data['Close'].iloc[-1])
    
    print(f"    {len(data)} bars | Latest: ${latest_price:.2f}")
    
    # Generate 3-channel GAF
    print(f"\n[2/4] Generating 3-channel GAF (224x224)...")
    close = data['Close'].values.flatten()[-100:]
    gaf_rgb = research.generate_gaf_3channel(close, image_size=224)
    print(f"    GAF shape: {gaf_rgb.shape}")
    
    # ConvNeXt prediction
    print(f"\n[3/4] Running ConvNeXt inference...")
    import time
    start = time.perf_counter()
    prediction = convnext.predict(gaf_rgb)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"    Inference time: {elapsed:.1f}ms")
    print(f"    Regime: {prediction.regime} ({prediction.regime_confidence:.1%})")
    print(f"    Direction: {prediction.direction} ({prediction.direction_confidence:.1%})")
    print(f"    Volatility: {prediction.volatility} ({prediction.volatility_confidence:.1%})")
    
    result = {
        "symbol": symbol,
        "price": latest_price,
        "convnext": prediction.to_dict(),
        "inference_ms": elapsed,
    }
    
    # Optionally combine with LLM
    if combine_with_llm:
        print(f"\n[4/4] Running LLM analysis for ensemble...")
        from brainnet.agents import ReasoningAgent
        reasoning = ReasoningAgent()
        llm_analysis = research.research(data)
        confidence = reasoning.compute_confidence(llm_analysis['analysis'])
        
        result["llm"] = {
            "features": llm_analysis['features'],
            "confidence": confidence,
        }
        
        # Ensemble decision
        # Weight ConvNeXt direction by its confidence, LLM by BSC confidence
        convnext_weight = prediction.direction_confidence
        llm_weight = confidence
        
        # Map directions to scores
        dir_scores = {"bullish": 1, "neutral": 0, "bearish": -1}
        trend_sign = 1 if llm_analysis['features']['trend_score'] > 0 else -1
        
        ensemble_score = (
            dir_scores[prediction.direction] * convnext_weight +
            trend_sign * llm_weight
        ) / (convnext_weight + llm_weight + 1e-6)
        
        if ensemble_score > 0.2:
            final_decision = "LONG"
        elif ensemble_score < -0.2:
            final_decision = "SHORT"
        else:
            final_decision = "FLAT"
        
        result["ensemble_decision"] = final_decision
        result["ensemble_score"] = ensemble_score
        
        print(f"\n    Ensemble Decision: {final_decision} (score: {ensemble_score:.3f})")
    
    print(f"\n{'='*60}\n")
    return result
```

### Task 7: CLI Flags

```python
# brainnet/services/main.py - ADD to argument parser

# In the CLI parser, add:
parser.add_argument(
    "--use-convnext", 
    action="store_true",
    help="Use ConvNeXt-Tiny for GAF pattern analysis"
)
parser.add_argument(
    "--convnext-only",
    action="store_true", 
    help="Use only ConvNeXt (skip LLM analysis)"
)
parser.add_argument(
    "--convnext-device",
    type=str,
    default="auto",
    choices=["auto", "cuda", "mps", "cpu"],
    help="Device for ConvNeXt inference"
)
```

### Task 8: Test Suite

```python
# tests/test_convnext.py

"""Tests for ConvNeXt GAF predictor."""

import pytest
import numpy as np

# Skip all tests if torch not available
torch_available = False
try:
    import torch
    torch_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch not installed")


class TestGAF3Channel:
    """Test 3-channel GAF generation."""
    
    def test_generate_gaf_3channel_shape(self):
        from brainnet.agents import ResearchAgent
        agent = ResearchAgent()
        series = np.sin(np.linspace(0, 10, 100))
        
        rgb = agent.generate_gaf_3channel(series, image_size=224)
        
        assert rgb.shape == (224, 224, 3)
        assert rgb.dtype == np.uint8
        
    def test_generate_gaf_3channel_values(self):
        from brainnet.agents import ResearchAgent
        agent = ResearchAgent()
        series = np.sin(np.linspace(0, 10, 100))
        
        rgb = agent.generate_gaf_3channel(series)
        
        assert rgb.min() >= 0
        assert rgb.max() <= 255
        
    def test_generate_gaf_3channel_different_sizes(self):
        from brainnet.agents import ResearchAgent
        agent = ResearchAgent()
        series = np.random.randn(50)
        
        for size in [64, 128, 224, 256]:
            rgb = agent.generate_gaf_3channel(series, image_size=size)
            assert rgb.shape == (size, size, 3)


class TestConvNeXtPredictor:
    """Test ConvNeXt prediction."""
    
    def test_initialization(self):
        from brainnet.agents import ConvNeXtPredictor
        predictor = ConvNeXtPredictor(device="cpu")
        assert predictor is not None
        assert predictor.device.type == "cpu"
        
    def test_predict_returns_valid_output(self):
        from brainnet.agents import ConvNeXtPredictor, ConvNeXtPrediction
        from brainnet.agents import ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        # Generate test GAF
        series = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb)
        
        assert isinstance(result, ConvNeXtPrediction)
        assert result.regime in ["trending", "mean_reverting", "volatile", "quiet"]
        assert result.direction in ["bullish", "bearish", "neutral"]
        assert result.volatility in ["calm", "normal", "elevated", "explosive"]
        assert 0 <= result.regime_confidence <= 1
        assert 0 <= result.direction_confidence <= 1
        assert 0 <= result.volatility_confidence <= 1
        
    def test_predict_with_features(self):
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        series = np.random.randn(100)
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb, return_features=True)
        
        assert result.features is not None
        assert result.features.shape == (768,)  # ConvNeXt-Tiny feature dim
        
    def test_batch_prediction(self):
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        # Generate multiple GAFs
        gaf_images = [
            research.generate_gaf_3channel(np.random.randn(100))
            for _ in range(3)
        ]
        
        results = predictor.predict_batch(gaf_images)
        
        assert len(results) == 3
        for r in results:
            assert r.regime in ["trending", "mean_reverting", "volatile", "quiet"]
            
    def test_to_dict(self):
        from brainnet.agents import ConvNeXtPredictor, ResearchAgent
        
        predictor = ConvNeXtPredictor(device="cpu")
        research = ResearchAgent()
        
        series = np.random.randn(100)
        gaf_rgb = research.generate_gaf_3channel(series)
        
        result = predictor.predict(gaf_rgb)
        d = result.to_dict()
        
        assert "regime" in d
        assert "direction" in d
        assert "volatility" in d
        assert isinstance(d["regime_confidence"], float)


class TestIntegration:
    """Integration tests for ConvNeXt + trading engine."""
    
    def test_engine_convnext_analysis(self):
        """Test full ConvNeXt analysis pipeline."""
        from unittest.mock import patch, MagicMock
        import pandas as pd
        
        # Mock yfinance data
        mock_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100
        })
        
        with patch('yfinance.download', return_value=mock_data):
            from brainnet.services.engine import run_convnext_analysis
            
            result = run_convnext_analysis(
                symbol="TEST",
                interval="5m",
                combine_with_llm=False,
            )
            
            assert "convnext" in result
            assert result["convnext"]["regime"] in ["trending", "mean_reverting", "volatile", "quiet"]
```

---

## Integration Points

```yaml
CONFIG:
  - add to: .env.example
  - values: |
      # ConvNeXt Configuration
      CONVNEXT_ENABLED=false
      CONVNEXT_MODEL=convnext_tiny
      CONVNEXT_DEVICE=auto
      CONVNEXT_FINE_TUNED_PATH=

DEPENDENCIES:
  - add to: requirements.txt
  - values: |
      torch>=2.1.0
      torchvision>=0.16.0
      scipy>=1.11.0

EXPORTS:
  - add to: brainnet/agents/__init__.py
  - pattern: "from .convnext_predictor import ConvNeXtPredictor, ConvNeXtPrediction"
```

---

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check brainnet/agents/convnext_predictor.py --fix
mypy brainnet/agents/convnext_predictor.py --ignore-missing-imports

# Expected: No errors
```

### Level 2: Unit Tests
```bash
# Install torch first if not present
pip install torch torchvision scipy

# Run ConvNeXt-specific tests
pytest tests/test_convnext.py -v

# Run all tests to ensure no regressions
pytest tests/ -v
```

### Level 3: Integration Test
```bash
# Single ConvNeXt analysis
python -c "
from brainnet.services.engine import run_convnext_analysis
result = run_convnext_analysis('SPY', '5m', combine_with_llm=False)
print(result)
"

# Expected output includes convnext predictions with regime, direction, volatility

# Full ensemble analysis
python -c "
from brainnet.services.engine import run_convnext_analysis
result = run_convnext_analysis('SPY', '5m', combine_with_llm=True)
print(f'Ensemble: {result.get(\"ensemble_decision\")}')
"
```

---

## Final Validation Checklist

- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linting errors: `ruff check brainnet/`
- [ ] No type errors: `mypy brainnet/ --ignore-missing-imports`
- [ ] ConvNeXt loads pretrained weights successfully
- [ ] 3-channel GAF generates valid 224x224 RGB images
- [ ] Inference time < 50ms on GPU, < 500ms on CPU
- [ ] Integration with trading loop works
- [ ] CLI flags recognized: `python -m brainnet.services.main --help`
- [ ] Ensemble predictions combine CNN + LLM outputs

---

## Anti-Patterns to Avoid

- ❌ Don't fine-tune on live trading data without walk-forward validation
- ❌ Don't trust ConvNeXt predictions blindly—always ensemble or validate
- ❌ Don't use batch_size > 1 for single-symbol real-time trading (latency)
- ❌ Don't skip the scipy resize—pyts GAF size depends on input length
- ❌ Don't hardcode device—use auto-detection for portability
- ❌ Don't import torch at module level—use lazy imports for graceful degradation
- ❌ Don't mix ConvNeXt preprocessing with custom normalization—use torchvision transforms

---

## Future Enhancements (Out of Scope)

1. **LSTM Head**: Add recurrent head for multi-step forecasting
2. **Fine-Tuning Pipeline**: Training script for domain adaptation
3. **Attention Visualization**: GradCAM for GAF pattern interpretation
4. **Multi-Timeframe Fusion**: Stack GAFs from 1m, 5m, 15m
5. **Quantization**: INT8 for faster edge deployment

