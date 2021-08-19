import wandb


with wandb.init(project="test", name="MyRunName1") as run:
    print("Hello")