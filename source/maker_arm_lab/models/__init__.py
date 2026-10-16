"""Learned models of the real arm (MS5 control plan task 10): ``residual`` is the torque the
identified dynamics model misses, trained on session logs and run by ``maker_arm_rs``.

Allowed imports: torch, numpy, h5py, ``maker_arm_rs`` and ``maker_arm_lab.core``, plus the
sibling ``maker_arm_lab.ident``, whose free-motion selector the dataset reuses and which holds
to the same list.
"""
