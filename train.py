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
from module import ARPredictor, Embedder, MLP, SIGReg, SubspaceSIGReg
from curve_loss import build_curve_loss
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    sigreg_cfg = getattr(cfg.loss, "sigreg", None)
    sigreg_enabled = bool(sigreg_cfg.enabled) if sigreg_cfg is not None and hasattr(sigreg_cfg, "enabled") else True
    sigreg_weight = float(sigreg_cfg.weight) if sigreg_cfg is not None and hasattr(sigreg_cfg, "weight") else 0.0
    sigreg_type = (
        str(OmegaConf.select(sigreg_cfg, "type", default="full")).lower()
        if sigreg_cfg is not None
        else "full"
    )

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

    # LeWM base losses
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    total_loss = output["pred_loss"]
    if sigreg_enabled:
        if sigreg_type == "subspace":
            if self.subspace_sigreg is None:
                raise RuntimeError(
                    "loss.sigreg.type=subspace but subspace_sigreg was not built; "
                    "check loss.sigreg.subspace in the training config."
                )
            output["sigreg_loss"] = self.subspace_sigreg(emb)
        else:
            output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        total_loss = total_loss + sigreg_weight * output["sigreg_loss"]

    # Optional temporal curvature regularization.
    if curve_enabled and curve_weight > 0.0:
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

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
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
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
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
    sigreg_type = (
        str(OmegaConf.select(sigreg_cfg, "type", default="full")).lower()
        if sigreg_cfg is not None
        else "full"
    )
    sigreg = SIGReg(**OmegaConf.to_container(sigreg_cfg.kwargs, resolve=True))
    subspace_sigreg = None
    if sigreg_type == "subspace":
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

    object_dump_callback = ModelObjectCallBack(
        dirpath=ckpt_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        default_root_dir=str(run_dir),
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=False,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
