"""WATCH Extensions: 4 architectural improvements.

Modules:
    dgc_layer: Differentiable Geometric Coupling
    dual_stream: Dual-stream transformer backbone with cross-attention
    tto: Test-time optimization with differentiable rendering
    physics_layer: Physics constraints (lightweight + full Brax option)

See record/06_future_directions.md for design rationale.
"""

from .dgc_layer import DGCLayer, DGCWrapper
from .dual_stream import DualStreamEncoder, CrossAttentionBlock
from .tto import TTOOptimizer, DifferentiableSMPLRenderer, project_kp3d_to_kp2d
from .physics_layer import (
    PhysicsConstraintLoss,
    ContactPredictionHead,
    DifferentiablePhysicsLayer,
    add_physics_loss_to_pipeline,
)

__all__ = [
    "DGCLayer", "DGCWrapper",
    "DualStreamEncoder", "CrossAttentionBlock",
    "TTOOptimizer", "DifferentiableSMPLRenderer", "project_kp3d_to_kp2d",
    "PhysicsConstraintLoss", "ContactPredictionHead", "DifferentiablePhysicsLayer",
    "add_physics_loss_to_pipeline",
]
