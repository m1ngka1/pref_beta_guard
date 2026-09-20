"""Optional factor-only adjustment; requires the optimization extra."""

from .beta_guard import InfeasibleAdjustmentError, SolverFailureError, adjust_factor_covariance

__all__ = ["adjust_factor_covariance", "InfeasibleAdjustmentError", "SolverFailureError"]
