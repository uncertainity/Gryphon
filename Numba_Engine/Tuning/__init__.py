"""Offline calibration tools for the production Gryphon Numba engine.

Nothing in this package implements an alternative game flow.  The tools
construct candidate configurations and measure them by calling the runtime
kernels in :mod:`Numba_Engine.core` and :mod:`Numba_Engine.simulations`.
"""
