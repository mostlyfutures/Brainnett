"""
Configuration management for Brainnet
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv


@dataclass
class ConvNeXtConfig:
    """ConvNeXt model configuration for GAF pattern recognition."""
    
    enabled: bool = False
    model_name: str = "convnext_tiny"
    weights: str = "IMAGENET1K_V1"
    device: str = "auto"  # auto, cuda, mps, cpu
    fine_tuned_path: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "ConvNeXtConfig":
        """Load ConvNeXt configuration from environment variables."""
        load_dotenv()
        return cls(
            enabled=os.getenv("CONVNEXT_ENABLED", "false").lower() == "true",
            model_name=os.getenv("CONVNEXT_MODEL", "convnext_tiny"),
            weights=os.getenv("CONVNEXT_WEIGHTS", "IMAGENET1K_V1"),
            device=os.getenv("CONVNEXT_DEVICE", "auto"),
            fine_tuned_path=os.getenv("CONVNEXT_FINE_TUNED_PATH"),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "weights": self.weights,
            "device": self.device,
            "fine_tuned_path": self.fine_tuned_path,
        }


@dataclass
class BrainnetConfig:
    """Configuration container for Brainnet system."""

    # LLM settings
    llm_backend: str = "local"
    llm_api_key: str = ""
    llm_base_url: str = "http://localhost:11434/v1"

    # Memory settings
    memory_db: str = "sqlite"
    postgres_url: str = ""
    sqlite_path: str = "brainnet_memory.db"

    # Trading settings
    symbol: str = "ES=F"
    confidence_threshold: float = 0.78
    max_refinements: int = 3

    # System settings
    log_level: str = "INFO"
    debug: bool = False

    @classmethod
    def from_env(cls) -> "BrainnetConfig":
        """Load configuration from environment variables."""
        load_dotenv()

        return cls(
            llm_backend=os.getenv("LLM_BACKEND", "local"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            memory_db=os.getenv("MEMORY_DB", "sqlite"),
            postgres_url=os.getenv("POSTGRES_URL", ""),
            sqlite_path=os.getenv("SQLITE_PATH", "brainnet_memory.db"),
            symbol=os.getenv("YFINANCE_SYMBOL", "ES=F"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.78")),
            max_refinements=int(os.getenv("MAX_REFINEMENTS", "3")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "llm_backend": self.llm_backend,
            "llm_api_key": "***" if self.llm_api_key else "",
            "llm_base_url": self.llm_base_url,
            "memory_db": self.memory_db,
            "postgres_url": "***" if self.postgres_url else "",
            "sqlite_path": self.sqlite_path,
            "symbol": self.symbol,
            "confidence_threshold": self.confidence_threshold,
            "max_refinements": self.max_refinements,
            "log_level": self.log_level,
            "debug": self.debug,
        }


def load_config() -> dict:
    """
    Load configuration from environment variables.

    Returns:
        Dictionary with configuration values
    """
    load_dotenv()

    return {
        "llm_backend": os.getenv("LLM_BACKEND", "local"),
        "llm_api_key": os.getenv("LLM_API_KEY", ""),
        "llm_base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        "memory_db": os.getenv("MEMORY_DB", "sqlite"),
        "postgres_url": os.getenv("POSTGRES_URL", ""),
        "sqlite_path": os.getenv("SQLITE_PATH", "brainnet_memory.db"),
        "symbol": os.getenv("YFINANCE_SYMBOL", "ES=F"),
        "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.78")),
        "max_refinements": int(os.getenv("MAX_REFINEMENTS", "3")),
        # ConvNeXt settings
        "convnext_enabled": os.getenv("CONVNEXT_ENABLED", "false").lower() == "true",
        "convnext_model": os.getenv("CONVNEXT_MODEL", "convnext_tiny"),
        "convnext_device": os.getenv("CONVNEXT_DEVICE", "auto"),
        "convnext_weights": os.getenv("CONVNEXT_WEIGHTS", "IMAGENET1K_V1"),
        "convnext_fine_tuned_path": os.getenv("CONVNEXT_FINE_TUNED_PATH"),
    }


def load_convnext_config() -> ConvNeXtConfig:
    """Load ConvNeXt-specific configuration."""
    return ConvNeXtConfig.from_env()
