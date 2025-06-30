import os
from typing import Optional
from vllm.logger import init_logger

logger = init_logger(__name__)

# Import GBA components - these should be available after proper installation
try:
    from vllm.model_executor.layers.quantization.gba import GBAConfig, GBALinearMethod
    GBA_AVAILABLE = True
except ImportError as e:
    # Fallback if not properly installed
    logger.warning(f"GBA quantization modules not found: {e}. Please ensure proper installation.")
    GBAConfig = None
    GBALinearMethod = None
    GBA_AVAILABLE = False


def integrate_gba_with_vllm():
    """
    Main integration function to set up GBA quantization in vLLM.
    This should be called during vLLM initialization.
    """
    if not GBA_AVAILABLE:
        raise ImportError("GBA quantization modules not available. Please ensure proper installation.")

    # Register GBA quantization using vLLM's official mechanism
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

    if "gba" not in QUANTIZATION_METHODS:
        try:
            # Method 1: Try official registration
            @register_quantization_config("gba")
            class GBAConfigRegistration(GBAConfig):
                """GBA quantization configuration registered with vLLM"""
                pass

            logger.debug("Registered GBA quantization with vLLM using official registration")
        except Exception as e:
            logger.warning(f"Official registration failed: {e}, trying manual registration")

            # Method 2: Manual registration fallback
            try:
                QUANTIZATION_METHODS["gba"] = GBAConfig
                logger.debug("Registered GBA quantization with vLLM using manual registration")
            except Exception as e2:
                logger.error(f"Both registration methods failed: {e2}")
                raise e2
    else:
        logger.debug("GBA quantization already registered")

    logger.debug("GBA integration with vLLM completed")

# Auto-initialize when module is imported
if GBA_AVAILABLE:
    try:
        integrate_gba_with_vllm()
    except Exception as e:
        logger.warning(f"Auto-initialization failed: {e}")