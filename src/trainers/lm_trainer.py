
from trainers.base_trainer import BaseTrainer


class LMTrainer(BaseTrainer):
    
    def train_forward(
        self,
        step,
        input_ids,
    ):
        
        outputs = self.model(
            input_ids=input_ids,
            labels=input_ids,
        )

        loss = outputs.loss

        return loss, {}
    