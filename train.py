from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import (
    ARPredictor,
    Embedder,
    MLP,
    SIGReg,
    SubspaceSIGReg,
    TokenARPredictor,
)
from curve_loss import build_curve_loss
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack

def get_model_mode(cfg):
    model_cfg = getattr(cfg, "model", None)
    return (
        str(OmegaConf.select(model_cfg, "mode", default="full")).lower()
        if model_cfg is not None
        else "full"
    )


def get_token_decouple_loss(cfg):
    if get_model_mode(cfg) != "token":
        return False
    return bool(OmegaConf.select(cfg, "model.token.decouple_loss", default=False))


def _token_subspace_slices(model, cfg):
    subspace_dim = int(
        getattr(model, "subspace_dim", None)
        or OmegaConf.select(cfg, "model.token.subspace_dim", default=cfg.wm.embed_dim // 2)
    )
    num_subspaces = int(
        getattr(model, "num_subspaces", None)
        or OmegaConf.select(cfg, "model.token.num_subspaces", default=2)
    )
    if num_subspaces != 2:
        raise RuntimeError(
            f"model.token.decouple_loss requires num_subspaces=2, got {num_subspaces}"
        )
    embed_dim = int(cfg.wm.embed_dim)
    if num_subspaces * subspace_dim != embed_dim:
        raise RuntimeError(
            f"model.token.decouple_loss requires num_subspaces * subspace_dim == embed_dim, "
            f"got {num_subspaces} * {subspace_dim} != {embed_dim}"
        )
    return slice(0, subspace_dim), slice(subspace_dim, embed_dim), subspace_dim


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    model_mode = get_model_mode(cfg)
    sigreg_cfg = getattr(cfg.loss, "sigreg", None)
    sigreg_enabled = bool(sigreg_cfg.enabled) if sigreg_cfg is not None and hasattr(sigreg_cfg, "enabled") else True
    sigreg_weight = float(sigreg_cfg.weight) if sigreg_cfg is not None and hasattr(sigreg_cfg, "weight") else 0.0
    use_subspace_sigreg = model_mode == "subspace"
    decouple_token_loss = get_token_decouple_loss(cfg)

    curve_cfg = getattr(cfg.loss, "curve", None)
    curve_enabled = bool(curve_cfg.enabled) if curve_cfg is not None and hasattr(curve_cfg, "enabled") else False
    curve_weight = float(curve_cfg.weight) if curve_cfg is not None and hasattr(curve_cfg, "weight") else 0.0
    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:]  # label

    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred
    assert pred_emb.shape == tgt_emb.shape, (
        f"pred_emb shape {pred_emb.shape} != tgt_emb shape {tgt_emb.shape}"
    )

    # LeWM base losses
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    total_loss = output["pred_loss"]
    if sigreg_enabled:
        if use_subspace_sigreg:
            if self.subspace_sigreg is None:
                raise RuntimeError(
                    "subspace SIGReg requested but subspace_sigreg was not built; "
                    "check model.mode and loss.sigreg in the training config."
                )
            output["sigreg_loss"] = self.subspace_sigreg(emb)
        elif decouple_token_loss:
            _, sigreg_slice, _ = _token_subspace_slices(self.model, cfg)
            sigreg_emb = emb[..., sigreg_slice]
            output["sigreg_loss"] = self.sigreg(sigreg_emb.transpose(0, 1))
        else:
            output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        total_loss = total_loss + sigreg_weight * output["sigreg_loss"]

    # Optional temporal curvature regularization.
    if curve_enabled and curve_weight > 0.0:
        if decouple_token_loss:
            curve_slice, _, _ = _token_subspace_slices(self.model, cfg)
            curve_features = emb[..., curve_slice]
        else:
            curve_features = emb
        if self.curve_loss is None:
            raise RuntimeError(
                "loss.curve is enabled but no curve_loss module was built; "
                "check loss.curve.type in the training config."
            )
        reg = self.curve_loss(curve_features)
        output[self.curve_loss.log_key] = reg
        total_loss = total_loss + curve_weight * reg

    output["loss"] = total_loss

    log_keys = {k for k in output if "loss" in k}
    log_dict = {f"{stage}/{k}": output[k].detach() for k in log_keys if k in output}
    self.log_dict(log_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    model_mode = get_model_mode(cfg)

    pred_output_dim = embed_dim
    token_predictor = None
    token_num_subspaces = None
    token_subspace_dim = None

    if model_mode == "token":
        token_cfg = OmegaConf.select(cfg, "model.token", default={})
        token_subspace_dim = int(OmegaConf.select(token_cfg, "subspace_dim", default=48))
        token_num_subspaces = OmegaConf.select(
            token_cfg, "num_subspaces", default=4
        )
        token_num_subspaces = (
            None
            if token_num_subspaces in (None, "")
            else int(token_num_subspaces)
        )
        token_residual = bool(OmegaConf.select(token_cfg, "residual", default=True))
        decouple_loss = bool(OmegaConf.select(token_cfg, "decouple_loss", default=False))
        if decouple_loss and token_num_subspaces not in (None, 2):
            raise ValueError(
                f"model.token.decouple_loss requires num_subspaces=2, got {token_num_subspaces}"
            )
        if decouple_loss and token_num_subspaces is None and embed_dim % 2 != 0:
            raise ValueError(
                "model.token.decouple_loss with num_subspaces=None requires embed_dim % 2 == 0"
            )
        predictor_kwargs = OmegaConf.to_container(cfg.predictor, resolve=True)
        for key in ("input_dim", "hidden_dim", "output_dim"):
            predictor_kwargs.pop(key, None)
        token_predictor = TokenARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            action_dim=embed_dim,
            subspace_dim=token_subspace_dim,
            num_subspaces=token_num_subspaces,
            hidden_dim=hidden_dim,
            residual=token_residual,
            **predictor_kwargs,
        )
        token_num_subspaces = token_predictor.num_subspaces
        token_subspace_dim = token_predictor.subspace_dim
        predictor = None
    else:
        predictor = ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=pred_output_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        model_mode=model_mode,
        token_predictor=token_predictor,
        num_subspaces=token_num_subspaces,
        subspace_dim=token_subspace_dim,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    curve_loss = build_curve_loss(getattr(cfg.loss, "curve", None))

    sigreg_cfg = getattr(cfg.loss, "sigreg", None)
    use_subspace_sigreg = model_mode == "subspace"
    sigreg = SIGReg(**OmegaConf.to_container(sigreg_cfg.kwargs, resolve=True))
    subspace_sigreg = None
    if use_subspace_sigreg:
        sub_cfg = OmegaConf.select(sigreg_cfg, "subspace", default={})
        subspace_dim = int(
            OmegaConf.select(sub_cfg, "subspace_dim", default=embed_dim // 4)
        )
        num_subspaces = int(OmegaConf.select(sub_cfg, "num_subspaces", default=4))
        subspace_mode = OmegaConf.select(sub_cfg, "mode", default="row-ortho")
        subspace_mode = (
            "row-ortho"
            if subspace_mode in (None, "")
            else str(subspace_mode).lower()
        )
        subspace_sigreg = SubspaceSIGReg(
            embed_dim=embed_dim,
            subspace_dim=subspace_dim,
            num_subspaces=num_subspaces,
            sigreg=sigreg,
            mode=subspace_mode,
        )

    world_model = spt.Module(
        model=world_model,
        sigreg=sigreg,
        subspace_sigreg=subspace_sigreg,
        curve_loss=curve_loss,
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_dir = Path(cfg.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**OmegaConf.to_container(cfg.wandb.config, resolve=True))
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    dump_epoch_interval = int(OmegaConf.select(cfg, "dump_epoch_interval", default=5))

    object_dump_callback = ModelObjectCallBack(
        dirpath=ckpt_dir,
        filename=cfg.output_model_name,
        epoch_interval=dump_epoch_interval,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        default_root_dir=str(run_dir),
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=False,
    )

    weights_ckpt = (ckpt_dir / f"{cfg.output_model_name}_weights.ckpt").resolve()
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=weights_ckpt if weights_ckpt.is_file() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
