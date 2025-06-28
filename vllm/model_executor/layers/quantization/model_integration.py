import os
from typing import Optional
from vllm.logger import init_logger

logger = init_logger(__name__)

# Import GBA components - these should be available after proper installation
try:
    from vllm.model_executor.layers.quantization.gba import GBAConfig, GBALinearMethod
    from vllm.model_executor.layers.quantization.gba_moe_support import (
        apply_moe_patches, restore_moe_patches
    )
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

            logger.info("Registered GBA quantization with vLLM using official registration")
        except Exception as e:
            logger.warning(f"Official registration failed: {e}, trying manual registration")

            # Method 2: Manual registration fallback
            try:
                QUANTIZATION_METHODS["gba"] = GBAConfig
                logger.info("Registered GBA quantization with vLLM using manual registration")
            except Exception as e2:
                logger.error(f"Both registration methods failed: {e2}")
                raise e2
    else:
        logger.info("GBA quantization already registered")

    logger.info("GBA integration with vLLM completed")


class GBAIntegrationManager:
    """
    Main integration manager for GBA quantization in vLLM.
    Coordinates all aspects of GBA integration.
    """

    def __init__(self):
        self.is_initialized = False
        self._ensure_gba_available()

    def _ensure_gba_available(self):
        if not GBA_AVAILABLE:
            raise ImportError("GBA quantization modules not available. Please ensure proper installation.")

    def register_gba_quantization(self):
        """Register GBA quantization with vLLM's quantization system"""
        integrate_gba_with_vllm()
        self.is_initialized = True

    def detect_and_configure_gba(self, model_config) -> Optional['GBAConfig']:
        """
        Detect if a model uses GBA quantization and configure accordingly.

        Args:
            model_config: vLLM model configuration

        Returns:
            GBA configuration if detected, None otherwise
        """
        if not GBA_AVAILABLE:
            logger.warning("GBA quantization modules not available")
            return None

        from vllm.model_executor.model_loader.weight_utils import detect_gba_quantization, load_gba_strategy_config

        model_path = model_config.model
        hf_config_dict = model_config.hf_config.to_dict() if hasattr(model_config, 'hf_config') else {}

        # Detect GBA quantization
        is_gba, gba_config_dict = detect_gba_quantization(model_path, hf_config_dict)

        if is_gba:
            logger.info(f"Detected GBA quantization for model: {model_path}")

            # Load strategy if available
            if os.path.isdir(model_path):
                strategy_config = load_gba_strategy_config(model_path)
                if strategy_config:
                    gba_config_dict["strategy"] = strategy_config
                    logger.info("Loaded GBA quantization strategy")

            # Create GBA config
            gba_config = GBAConfig.from_config(gba_config_dict)

            # Update model config
            model_config.quantization = "gba"
            model_config.quantization_param_path = model_path

            return gba_config

        return None

    def get_weight_loader_for_param(self, param_name: str, model_config) -> callable:
        """
        Get the appropriate weight loader for a parameter.

        Args:
            param_name: Name of the parameter
            model_config: Model configuration

        Returns:
            Weight loader function
        """
        # Check if this is a GBA model and parameter
        if (hasattr(model_config, 'quantization') and
                model_config.quantization == "gba"):

            from vllm.model_executor.model_loader.weight_utils import should_use_gba_weight_loader, gba_weight_loader

            if should_use_gba_weight_loader(param_name):
                return lambda param, loaded_weight: gba_weight_loader(
                    param, loaded_weight, param_name
                )

        # Fallback to default loader
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        return default_weight_loader


def patch_vllm_for_gba_support(manager: Optional[GBAIntegrationManager] = None):
    """
    Apply patches to vLLM functions to support GBA quantization.

    Args:
        manager: The GBA integration manager (optional)
    """
    if manager is None:
        manager = GBAIntegrationManager()
        manager.register_gba_quantization()

    # Patch the quantization config getter
    try:
        from vllm.model_executor.model_loader.weight_utils import get_quant_config

        # Store original function to avoid infinite recursion
        if not hasattr(get_quant_config, '_original_function'):
            get_quant_config._original_function = get_quant_config

        def patched_get_quant_config(model_config, load_config):
            # Try GBA detection first
            try:
                gba_config = manager.detect_and_configure_gba(model_config)
                if gba_config:
                    logger.info("Using GBA quantization config")
                    return gba_config
            except Exception as e:
                logger.warning(f"GBA detection failed: {e}")

            # Fallback to original implementation
            return get_quant_config._original_function(model_config, load_config)

        # Apply the patch
        import vllm.model_executor.model_loader.weight_utils
        vllm.model_executor.model_loader.weight_utils.get_quant_config = patched_get_quant_config

        logger.info("Applied vLLM patches for GBA support")

    except Exception as e:
        logger.warning(f"Failed to apply some patches: {e}")
        logger.info("GBA will still work but some automatic detection may not function")


# Auto-initialize when module is imported
if GBA_AVAILABLE:
    try:
        integrate_gba_with_vllm()
    except Exception as e:
        logger.warning(f"Auto-initialization failed: {e}")