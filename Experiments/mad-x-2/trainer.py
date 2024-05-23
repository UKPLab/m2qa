import os
import random
import time
from typing import List, Optional, Dict

import datasets
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AdapterTrainer
from transformers.adapters.composition import Stack
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSampler,
    DistributedSamplerWithLoop,
    LengthGroupedSampler,
)
from transformers.trainer_utils import TrainOutput, seed_worker, speed_metrics
from transformers.training_args import ParallelMode
from transformers.utils import (
    is_datasets_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

logger = logging.get_logger(__name__)


class MadX2TrainerState(TrainerState):
    # global_step is already defined in TrainerState
    track_steps_per_dataset: Dict[
        str, Dict[str, int]
    ]  # how many steps have been trained for each dataset, so that we can resume training
    dataset_step_list: List[str]  # List of tuples (language, domain)


class DomainLanguageTrainer(AdapterTrainer):
    def __init__(
        self,
        train_datasets: List[Dataset],
        eval_datasets: List[Dataset],
        steps_per_dataset_dict: Dict[str, Dict[str, int]],
        mlm_head_name: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_datasets: List[Dataset] = train_datasets
        self.eval_datasets: List[Dataset] = eval_datasets
        self.steps_per_dataset_dict: Dict[str, Dict[str, int]] = steps_per_dataset_dict
        self.mlm_head_name: str = mlm_head_name
        self.model.freeze_model()
        self.model.base_model.model_frozen = True
        self.train_adapter_fusion = False

    def _get_train_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                print(f"train_dataset: {train_dataset}")
                return RandomSampler(train_dataset, generator=generator)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )

    def get_train_dataloaders(self) -> Dict[str, Dict[str, Dict[str, any]]]:
        if self.train_datasets is None:
            raise ValueError("Trainer: training requires a train_datasets (instead of" " train_dataset).")

        data_collator = self.data_collator

        train_dataloaders = {}

        for train_dataset in self.train_datasets:
            train_sampler = self._get_train_sampler(train_dataset["dataset"])

            if train_dataset["language"] not in train_dataloaders:
                train_dataloaders[train_dataset["language"]] = {}

            train_dataloaders[train_dataset["language"]][train_dataset["domain"]] = {
                "dataloader": DataLoader(
                    train_dataset["dataset"],
                    batch_size=self._train_batch_size,
                    sampler=train_sampler,
                    collate_fn=data_collator,
                    drop_last=self.args.dataloader_drop_last,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=self.args.dataloader_pin_memory,
                    worker_init_fn=seed_worker,
                )
            }

        return train_dataloaders

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self._train_batch_size = batch_size

        train_dataloaders: Dict[str, Dict[str, Dict[str, any]]] = self.get_train_dataloaders()

        # Setting up training control variables:
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        total_steps = sum(
            [sum([steps for steps in domain.values()]) for domain in self.steps_per_dataset_dict.values()]
        )

        self.state = MadX2TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # Use self.steps_per_dataset for the scheduler
        self.create_optimizer_and_scheduler(num_training_steps=total_steps)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Instantaneous batch size per device =" f" {args.per_device_train_batch_size}")
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")  # fmt: skip
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {total_steps}")

        self.state.epoch = 0
        start_time = time.time()

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            # Problems: State of model, optimizer, learning rate and state of the dataloaders
            # in the Huggin Face code, there is a lot more going on here, but we don't need these options
            # self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            raise NotImplementedError("resume_from_checkpoint is not yet supported.")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.state.trial_params = None

        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = total_steps
        self.state.num_train_epochs = 1
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        ############## Training Epochs #####################
        if args.gradient_accumulation_steps > 1:
            raise NotImplementedError("gradient_accumulation_steps > 1 is not yet supported.")

        # Create iterator for each dataloader
        for language, domains in train_dataloaders.items():
            for domain, dataloader in domains.items():
                dataloader["iterator"] = iter(dataloader["dataloader"])

        if resume_from_checkpoint is not None:
            raise NotImplementedError("resume_from_checkpoint is not yet supported.")
            # track_steps_per_dataset = self.state.track_steps_per_dataset
            # dataset_step_list = self.state.dataset_step_list

            # # Skip past any already trained steps on the dataloaders.
            # for language, domains in train_dataloaders.items():
            #     for domain, dataloader in domains.items():
            #         if track_steps_per_dataset[language][domain] > 0:
            #             for _ in range(track_steps_per_dataset[language][domain]):
            #                 next(dataloader["iterator"])

            # remaining_dataset_step_list = dataset_step_list[dataset_step_list.index((language, domain)) + 1 :]
            # self._load_rng_state(resume_from_checkpoint) # set the random seeds if we are resuming from a checkpoint

        else:
            # track_steps_per_dataset is only needed if we want to resume training
            track_steps_per_dataset = {
                lang: {domain: 0 for domain in domains} for lang, domains in self.steps_per_dataset_dict.items()
            }
            # Create list of (language, domain) pairs as the order in which we want to train the adapters
            dataset_step_list = []
            for language, domains in self.steps_per_dataset_dict.items():
                for domain, steps in domains.items():
                    dataset_step_list.extend([(language, domain)] * steps)

            random.shuffle(dataset_step_list)

            self.state.track_steps_per_dataset = track_steps_per_dataset
            self.state.dataset_step_list = dataset_step_list
            remaining_dataset_step_list = dataset_step_list

        logging.set_verbosity_warning()
        for step, (language, domain) in enumerate(remaining_dataset_step_list):
            train_dataloader = train_dataloaders[language][domain]

            model.active_adapters = Stack(
                language,
                domain,
            )
            model.active_head = self.mlm_head_name

            def train_adapters():
                for name, param in model.named_parameters():
                    if language in name or domain in name or self.mlm_head_name in name:
                        param.requires_grad = True
                    else:
                        param.requires_gard = False

            train_adapters()

            try:
                inputs = next(train_dataloader["iterator"])
            except StopIteration:
                # If iterator has reached the end of the dataset, reset it
                train_dataloader["iterator"] = iter(train_dataloader["dataloader"])
                inputs = next(train_dataloader["iterator"])
                # set counter to 0
                track_steps_per_dataset[language][domain] = 0

            tr_loss_step = self.training_step(model, inputs)

            if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                # if loss is nan or inf simply add the average of previous logged losses
                tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
            else:
                tr_loss += tr_loss_step

            self.current_flos += float(self.floating_point_ops(inputs))

            # Gradient clipping
            if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                if hasattr(self.optimizer, "clip_grad_norm"):
                    # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                    self.optimizer.clip_grad_norm(args.max_grad_norm)
                elif hasattr(model, "clip_grad_norm_"):
                    # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                    model.clip_grad_norm_(args.max_grad_norm)
                else:
                    # Revert to normal clipping otherwise, handling Apex or full precision
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()
            track_steps_per_dataset[language][domain] += 1

            model.zero_grad()
            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)

            self._maybe_log_save_evaluate(
                tr_loss=tr_loss,
                model=model,
                trial=trial,
                epoch=0,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

        ############## End of Training Epochs #####################

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed.\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        num_train_samples = total_train_batch_size * total_steps
        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
