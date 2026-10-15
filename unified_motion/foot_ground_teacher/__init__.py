"""Offline Foot-Ground Dynamics Teacher V1.

This package is deliberately independent from PPO and from the deployable
Physics Student.  Simulator truth enters only the ``privileged_current`` input
and the *next-control-step* labels described by :mod:`.schema`.
"""

from .model import FootGroundTeacher, FootGroundTeacherConfig
from .schema import CONTACT_MODES, FootGroundBatch, RegressionStatistics

__all__ = [
    "CONTACT_MODES",
    "FootGroundBatch",
    "FootGroundTeacher",
    "FootGroundTeacherConfig",
    "RegressionStatistics",
]
