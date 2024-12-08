import argparse
import wandb

class Config:
    def __init__(self):
        self.debug = False

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--debug', type=bool, default=False)  ## ⚠️ this is the problem

    config = Config()
    for key, value in vars(parser.parse_args()).items():
        setattr(config, key, value)
    
    print(config.__dict__)

    wandb_run = wandb.init(
        ## set the wandb project where this run will be logged
        project="udacity-awsmle-resnet50-dog-breeds",
        allow_val_change=True,
        config=config,
    )
    config.new_arg = 'test'
    try:
        wandb.config.update(config.__dict__, allow_val_change=True)  ## it works
    except Exception as e:
        print(f"⚠️ Updating wandb config failed: {e}")
    wandb.finish()

## $ python misc\test.py