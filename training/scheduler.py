## Responsibilities
# Create different learning rate schedulers
# Return the scheduler object
# Support multiple scheduling strategies
# Hide scheduler-specific implementation details

from torch.optim.lr_scheduler import(
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau
)

class SchedulerFactory:

    @staticmethod
    def create_scheduler(
        optimizer,
        scheduler_name="StepLR",
        step_size=10,
        gamma=0.1,
        milestones=None,
        T_max=50,
        patience=5,
        min_lr=1e-6
    ):
        scheduler_name = scheduler_name.lower()

        if scheduler_name == "steplr":

            scheduler = StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma
            )

        elif scheduler_name == "multisteplr":
            if milestones is None:
                milestones = [30, 60, 90]

            scheduler = MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=gamma
            )
        
        elif scheduler_name == "cosineannealinglr":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=min_lr
            )

        elif scheduler_name == "reduceronplateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=gamma,
                patience=patience,
                min_lr=min_lr
            )
        
        else: 
            raise ValueError(
                f"Unsupported scheduler : {scheduler_name}"
            )
        return scheduler
    
    # List Supported Schedulers
    @staticmethod
    def list_scheduler():
        return [
            "StepLR",
            "MultiStepLR",
            "ExponentialLR",
            "CosineAnnealingLR",
            "ReduceLROnPlateau"
        ]