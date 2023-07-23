import json
import os
from transformers import (
    TrainerCallback,
    TrainingArguments,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from valley.utils import get_logger

logger = get_logger("Trainer")

class LLMCallback(TrainerCallback):
    "A callback that output infomation and do some operators"

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        return super().on_step_end(args, state, control, **kwargs)
    
    def output_log(self, args: TrainingArguments, state: TrainerState):
        def loss_log(data):
            try:
                loss_ = data["loss"]
                learning_rate_ = data["learning_rate"]
                step_ = data["step"]
                loss_log_str = f"step: {step_:<8} || learning_rate: {learning_rate_:<25} || loss: {loss_:<10}"
            except:
                loss_log_str = json.dumps(data)
            return loss_log_str

        output_file = os.path.join(args.output_dir, "trainer.log")
        log_history = map(loss_log, state.log_history)
        with open(output_file, "w") as f:
            for line in log_history:
                f.write(line + "\n")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # TODO: support deepspeed zero3 save extra weights not all llm weights
        if args.save_strategy == 'steps' and state.global_step%args.save_steps == 0:
            self.output_log(args, state)
            peft_str = "peft"
            model_ = kwargs["model"]
            save_number = str(state.global_step)
            if (
                args.lora
                and peft_str in str(type(model_))
                and not is_deepspeed_zero3_enabled()
            ):
                # if model is peft-based, save the extra weights, zero3 not supprot
                epoch = "steps_" + save_number
                output_dir = os.path.join(args.output_dir, epoch)
                os.makedirs(output_dir, exist_ok=True)
                model_.save_pretrained(output_dir)
            if (
                args.lora
                and peft_str in str(type(model_))
                and args.tune_mm_mlp_adapter
                and not is_deepspeed_zero3_enabled()
            ):
                epoch = "steps_" + save_number
                output_dir = os.path.join(args.output_dir, epoch)
                os.makedirs(output_dir, exist_ok=True)
                model_.basemodel.model.save_pretrained(output_dir)
                
        return super().on_step_end(args, state, control, **kwargs)
    
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # TODO: support deepspeed zero3 save extra weights not all llm weights
        if args.save_strategy == 'epoch':
            self.output_log(args, state)
            peft_str = "peft"
            model_ = kwargs["model"]
            save_number = str(state.global_step)
            if (
                args.lora
                and peft_str in str(type(model_))
                and not is_deepspeed_zero3_enabled()
            ):
                # if model is peft-based, save the extra weights, zero3 not supprot
                epoch = "epoch_" + save_number
                output_dir = os.path.join(args.output_dir, epoch)
                os.makedirs(output_dir, exist_ok=True)
                model_.save_pretrained(output_dir)
            if (
                args.lora
                and peft_str in str(type(model_))
                and args.tune_mm_mlp_adapter
                and not is_deepspeed_zero3_enabled()
            ):
                epoch = "epoch_" + save_number
                output_dir = os.path.join(args.output_dir, epoch)
                os.makedirs(output_dir, exist_ok=True)
                model_.basemodel.model.save_pretrained(output_dir)
        return super().on_epoch_end(args, state, control, **kwargs)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.merge_files(args.prediction_file_name)

        return super().on_evaluate(args, state, control, **kwargs)

    def merge_files(self, prediction_file_name):
        import glob

        re_prediction_file_name = prediction_file_name.replace(".jsonl", "_*")
        old_files = glob.glob(re_prediction_file_name)
        with open(prediction_file_name, "w") as writer:
            for file_name in old_files:
                with open(file_name, "r") as reader:
                    for line in reader:
                        writer.write(f"{line}")
                # os.remove(file_name)

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ):
        self.merge_files(args.prediction_file_name)
        return super().on_predict(args, state, control, metrics, **kwargs)
