import importlib
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, HMC, predictive


# module-level storage of module names for the worker processes
_MODEL_MODULE_NAME = None
_ODMRSIM_MODULE_NAME = None


def _universal_model(wrapped):
     """Top-level picklable wrapper used by worker processes.

     Expects `wrapped` to be a tuple: (model_module_name, odmrsim_module_name, data_)
     This avoids relying on module globals which are not set in spawned workers.
     """
     try:
          model_name, odm_name, data = wrapped
     except Exception:
          raise RuntimeError("Worker received unexpected data format; expected (model_name, odm_name, data)")
     model_mod = importlib.import_module(model_name)
     odm_mod = importlib.import_module(odm_name)
     return model_mod.model(odm_mod, data)


def mcmc(model, odmrsimulator, data_, num_samples=5000, warmup_steps=500,
           num_chains=1, jit_compile=True, max_tree_depth=12):
     """
     Run MCMC for a Pyro `model` that takes `(odmrsimulator, data_)`.

     This function registers the model and odmrsimulator module names at
     module scope and uses `_universal_model` (a top-level function) as the
     kernel callable so multiprocessing can pickle it on Windows.
     """
     global _MODEL_MODULE_NAME, _ODMRSIM_MODULE_NAME
     # prefer module names (strings) so child processes can import them
     if isinstance(model, str):
          _MODEL_MODULE_NAME = model
     else:
          _MODEL_MODULE_NAME = getattr(model, "__module__", None)
     if isinstance(odmrsimulator, str):
          _ODMRSIM_MODULE_NAME = odmrsimulator
     else:
          _ODMRSIM_MODULE_NAME = getattr(odmrsimulator, "__name__", None)

     if not _MODEL_MODULE_NAME or not _ODMRSIM_MODULE_NAME:
          raise ValueError("Could not determine module names for model/odmrsimulator")

     kernel = NUTS(_universal_model,
                      jit_compile=jit_compile,
                      ignore_jit_warnings=True,
                      max_tree_depth=max_tree_depth)
     posterior = MCMC(kernel, num_samples=num_samples,
                          warmup_steps=warmup_steps,
                          num_chains=num_chains)
     # wrap the data with module names so spawned workers can import the modules
     wrapped_data = (_MODEL_MODULE_NAME, _ODMRSIM_MODULE_NAME, data_)
     posterior.run(wrapped_data)
     return posterior


