"""
Differentiable Physics Constraints for WATCH

Key idea: enforce physical plausibility during training/inference:
  1. Foot-ground contact (no foot below floor)
  2. Foot-no-slide when in contact
  3. Velocity smoothness (no impossible accelerations)
  4. Center of mass balance (during contact)

Two implementations:
  - PhysicsConstraintLoss: lightweight, just adds losses (recommended)
  - DifferentiablePhysicsLayer: full Brax-based simulator (advanced, optional)

The lightweight version is what we recommend first. It captures 80% of
the benefit (foot sliding domination) without Brax engineering hell.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# SMPL/SMPLX joint indices
JOINT_LEFT_ANKLE = 7
JOINT_RIGHT_ANKLE = 8
JOINT_LEFT_FOOT = 10
JOINT_RIGHT_FOOT = 11

FOOT_JOINTS = [JOINT_LEFT_ANKLE, JOINT_RIGHT_ANKLE, JOINT_LEFT_FOOT, JOINT_RIGHT_FOOT]


class PhysicsConstraintLoss(nn.Module):
    """Lightweight differentiable physics constraints.

    Computes:
      - Foot-ground contact violation (foot below ground)
      - Foot-sliding (when in contact, foot shouldn't move horizontally)
      - Velocity smoothness (acceleration penalty)
      - Center of mass balance (root close to support polygon)

    Use as additional loss term during training, or as TTO objective.
    """

    def __init__(
        self,
        ground_y=0.0,           # World ground plane y-coordinate
        contact_threshold=0.05,  # foot < 5cm from ground = contact
        weight_ground=1.0,
        weight_sliding=10.0,
        weight_smoothness=0.1,
        weight_balance=0.5,
        gravity_axis=1,          # 0=x, 1=y, 2=z (default y for ay-up)
    ):
        super().__init__()
        self.ground_y = ground_y
        self.contact_threshold = contact_threshold
        self.weight_ground = weight_ground
        self.weight_sliding = weight_sliding
        self.weight_smoothness = weight_smoothness
        self.weight_balance = weight_balance
        self.gravity_axis = gravity_axis

    def forward(self, joints_world, contact_logits=None):
        """
        Args:
            joints_world: (B, L, J, 3) world-frame joint positions
            contact_logits: (B, L, n_feet) optional learned contact predictions

        Returns:
            dict of loss components
        """
        losses = {}

        foot_pos = joints_world[..., FOOT_JOINTS, :]  # (B, L, 4, 3)
        foot_y = foot_pos[..., self.gravity_axis]  # (B, L, 4)

        # 1. Ground penetration (foot should not go below ground)
        ground_violation = F.relu(self.ground_y - foot_y)  # foot below ground
        losses["physics_ground"] = ground_violation.mean() * self.weight_ground

        # 2. Foot-sliding (when contact, horizontal motion penalized)
        if contact_logits is None:
            # Heuristic: foot is in contact if close to ground
            contact_mask = (foot_y - self.ground_y).abs() < self.contact_threshold
            contact_mask = contact_mask.float()
        else:
            contact_mask = torch.sigmoid(contact_logits)  # (B, L, 4)

        # Horizontal axes (not gravity axis)
        horiz_axes = [i for i in range(3) if i != self.gravity_axis]
        foot_horiz = foot_pos[..., horiz_axes]  # (B, L, 4, 2)

        if foot_horiz.shape[1] > 1:
            foot_vel = foot_horiz[:, 1:] - foot_horiz[:, :-1]  # (B, L-1, 4, 2)
            foot_speed = foot_vel.norm(dim=-1)  # (B, L-1, 4)
            sliding = foot_speed * contact_mask[:, :-1]
            losses["physics_sliding"] = sliding.mean() * self.weight_sliding
        else:
            losses["physics_sliding"] = torch.tensor(0.0, device=joints_world.device)

        # 3. Acceleration smoothness (no jerk)
        if joints_world.shape[1] > 2:
            joint_vel = joints_world[:, 1:] - joints_world[:, :-1]
            joint_accel = joint_vel[:, 1:] - joint_vel[:, :-1]
            accel_norm = joint_accel.norm(dim=-1)  # (B, L-2, J)
            losses["physics_smoothness"] = accel_norm.mean() * self.weight_smoothness
        else:
            losses["physics_smoothness"] = torch.tensor(0.0, device=joints_world.device)

        # 4. Center of mass balance (during contact phases)
        # Approximate COM as root joint
        root = joints_world[..., 0, :]  # (B, L, 3)
        # Support polygon = mean of foot positions in contact
        support = foot_pos.mean(dim=-2)  # (B, L, 3)
        # Horizontal distance from COM to support (only during contact)
        any_contact = contact_mask.max(dim=-1).values  # (B, L)
        horiz_offset = (root - support)[..., horiz_axes].norm(dim=-1)  # (B, L)
        balance_violation = (horiz_offset * any_contact).mean()
        losses["physics_balance"] = balance_violation * self.weight_balance

        # Total
        losses["physics_total"] = (
            losses["physics_ground"] + losses["physics_sliding"]
            + losses["physics_smoothness"] + losses["physics_balance"]
        )
        return losses


class ContactPredictionHead(nn.Module):
    """Predicts per-frame foot contact probabilities.

    Add to network as auxiliary head. Use predictions in PhysicsConstraintLoss
    to soften the contact thresholding.
    """

    def __init__(self, latent_dim=512, n_feet=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_feet),
        )

    def forward(self, x):
        """
        Args:
            x: (B, L, latent_dim) backbone features
        Returns:
            contact_logits: (B, L, n_feet)
        """
        return self.head(x)


class DifferentiablePhysicsLayer(nn.Module):
    """[ADVANCED] Full differentiable physics simulator using Brax.

    NOTE: This requires Brax + adapted SMPL humanoid model.
    Recommend starting with PhysicsConstraintLoss above; only use this
    if you need full forward simulation.

    Forward:
        init_state, controls -> simulated trajectory

    Backward: gradients propagate through physics simulator.
    """

    def __init__(self, dt=1/30, ground_height=0.0):
        super().__init__()
        self.dt = dt
        self.ground_height = ground_height

        # Try to load Brax (optional dependency)
        try:
            import brax
            self.brax = brax
            # Build humanoid env (would need SMPL adapter — placeholder)
            self.has_brax = True
        except ImportError:
            print("[Physics] Brax not available — DifferentiablePhysicsLayer disabled")
            self.has_brax = False

    def forward(self, init_state, controls):
        """
        Args:
            init_state: (B, state_dim) — pos, vel, root rotation, joint angles
            controls: (B, L, action_dim) — joint torques per frame

        Returns:
            states: (B, L, state_dim) — simulated trajectory
        """
        if not self.has_brax:
            raise NotImplementedError("Brax not available")

        # Pseudocode (real implementation requires Brax humanoid + SMPL adapter)
        # states = []
        # state = init_state
        # for t in range(controls.shape[1]):
        #     state = self.brax_step(state, controls[:, t])
        #     states.append(state)
        # return torch.stack(states, dim=1)
        raise NotImplementedError("Implement after PhysicsConstraintLoss validates")


def add_physics_loss_to_pipeline(outputs, smpl_model, physics_loss_module):
    """Helper to integrate physics loss into existing pipeline.

    Call this in pipeline.forward after computing pred_smpl_params_global.

    Returns:
        outputs (modified in-place) with physics losses added
    """
    # Compute world-frame joint positions
    smpl_out = smpl_model(**outputs["pred_smpl_params_global"])
    joints_world = smpl_out.joints[:, :, :22]  # SMPL/SMPLX body joints

    # Get contact predictions (if available)
    contact_logits = outputs.get("contact_logits", None)

    # Compute physics losses
    physics_losses = physics_loss_module(joints_world, contact_logits)

    # Add to outputs
    for k, v in physics_losses.items():
        outputs[k] = v

    return outputs
