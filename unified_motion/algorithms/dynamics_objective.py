"""Shared system-identification objectives; privileged labels never enter the Actor."""
import math

import torch


def validate_teacher_weights(model, reconstruction, distillation, *, teacher_available=False):
    if any(not math.isfinite(v) or v < 0 for v in (reconstruction, distillation)):
        raise ValueError("Teacher weights must be finite and non-negative")
    if bool(reconstruction) != bool(distillation):
        raise ValueError("enable or disable both Teacher objectives together")
    if reconstruction and not (model.teacher_student or teacher_available):
        raise ValueError("Teacher objectives require a Teacher model")


def identification_loss(model, latent, target, valid, *, reconstruction_weight=0.,
                        distillation_weight=0., teacher_valid=None):
    """Explicit validity, optional parameter-autoencoder auxiliary, shared NLL/MSE."""
    target = target.detach()
    if target.shape != latent.shape[:-1] + (model.system_id_dim,) or valid.shape != latent.shape[:-1]:
        raise ValueError("identification target / validity shapes disagree")
    if not torch.isfinite(target).all().item():
        raise ValueError("identification labels must be finite")
    validate_teacher_weights(model, reconstruction_weight, distillation_weight)
    if model.uncertainty:
        mean, logvar = model.identify_privileged_distribution(latent)
        error = .5*((mean-target).square()*torch.exp(-logvar)+logvar).mean(-1)
    else:
        mean = model.identify_privileged(latent)
        logvar = None
        error = (mean-target).square().mean(-1)
    zero = latent.sum()*0.
    terms = dict(system_identification=torch.where(valid, error, 0.).sum()/valid.sum().clamp_min(1),
                 teacher_reconstruction=zero, teacher_distillation=zero)
    if reconstruction_weight:
        teacher = model.encode_privileged_teacher(target)
        reconstructed = (model.reconstruct_privileged(teacher)-target).square().mean(-1)
        teacher_valid = valid if teacher_valid is None else teacher_valid
        terms["teacher_reconstruction"] = torch.where(teacher_valid, reconstructed, 0.).sum()/teacher_valid.sum().clamp_min(1)
        error = (latent-teacher.detach()).square().mean(-1)
        terms["teacher_distillation"] = torch.where(valid, error, 0.).sum()/valid.sum().clamp_min(1)
    return terms, mean, logvar
