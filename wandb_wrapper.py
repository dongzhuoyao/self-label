import wandb
from pytorchgo.utils import logger

def wandb_logging(d, step,use_wandb=True, prefix="training:"):
    if use_wandb:
        wandb.log(d, step=step)
    _str = "{} step={}\t".format(prefix, step)
    for k,v in d.items():
        _str += "{k}={v:.4f}\t".format(k=k,v=v)
    logger.info(_str)
