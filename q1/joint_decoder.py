"""Shim: old `from joint_decoder import ...` → `segpos.segmentation.joint_beam`."""

from segpos.segmentation.joint_beam import BeamHypothesis, JointBeamDecoder

__all__ = ["BeamHypothesis", "JointBeamDecoder"]
