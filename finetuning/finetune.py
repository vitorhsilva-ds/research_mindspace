#!/usr/bin/env python3
"""
Name: lora_finetuning_pipeline
Input: JSONL training and validation files containing conversation records
Output: LoRA adapter directory, optional merged model directory, and training_summary.json
Usage: python3 finetune_enterprise_complexity.py [--dry-run] [--no-merge] [--lora-r N] [--epochs N] [--lr VALUE]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


MODEL_ID = "openai/gpt-oss-20b"
TRAIN_FILE = Path("./finetune_data/dataset_train.jsonl")
VAL_FILE = Path("./finetune_data/dataset_val.jsonl")
OUTPUT_DIR = Path("./model_lora")
MERGED_DIR = Path("./model_merged")

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
WEIGHT_DECAY = 0.01

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_ENCODING = "utf-8"
DEFAULT_RANDOM_STATE = 42
DEFAULT_CHAT_TEMPLATE_NAME = "chatml"
DEFAULT_MERGED_SAVE_METHOD = "merged_16bit"
TRAINING_SUMMARY_FILE_NAME = "training_summary.json"

JsonRecord = dict[str, Any]
JsonObject = dict[str, Any]


class FineTuningExecutionError(Exception):
    """Base exception for fine-tuning execution failures."""


class DatasetMaterializationError(FineTuningExecutionError):
    """Raised when dataset materialization cannot be completed."""


class TrainerAssemblyError(FineTuningExecutionError):
    """Raised when the supervised fine-tuning trainer cannot be assembled."""


class ArtifactPersistenceError(FineTuningExecutionError):
    """Raised when a required training artifact cannot be persisted."""


@dataclass(frozen=True)
class DatasetContract:
    """Defines the dataset input contract."""

    training_file_path: Path
    validation_file_path: Path
    record_conversation_key: str = "conversations"
    encoding: str = DEFAULT_ENCODING


@dataclass(frozen=True)
class ModelLoadingContract:
    """Defines the model materialization contract."""

    model_id: str
    max_sequence_length: int
    dtype: Any | None = None
    load_in_4bit: bool = False


@dataclass(frozen=True)
class LoRAAdaptationContract:
    """Defines the LoRA adapter configuration contract."""

    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    bias: str = "none"
    gradient_checkpointing_mode: str = "unsloth"
    random_state: int = DEFAULT_RANDOM_STATE
    use_rslora: bool = False
    loftq_config: Any | None = None


@dataclass(frozen=True)
class TrainingHyperparameterContract:
    """Defines the supervised fine-tuning hyperparameter contract."""

    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    warmup_ratio: float
    lr_scheduler_type: str
    weight_decay: float
    max_sequence_length: int
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str = "none"
    dataloader_num_workers: int = 0
    seed: int = DEFAULT_RANDOM_STATE
    optim: str = "adamw_8bit"
    packing: bool = True
    dataset_num_proc: int = 1
    dataset_text_field: str = "text"

    @property
    def effective_batch_size(self) -> int:
        """Return the effective batch size used by gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class ArtifactPersistenceContract:
    """Defines the model artifact persistence contract."""

    lora_output_dir: Path
    merged_output_dir: Path
    training_summary_file_name: str = TRAINING_SUMMARY_FILE_NAME
    merged_save_method: str = DEFAULT_MERGED_SAVE_METHOD

    @property
    def training_summary_file_path(self) -> Path:
        """Return the training summary file path."""
        return self.lora_output_dir / self.training_summary_file_name


@dataclass(frozen=True)
class RuntimeExecutionDirective:
    """Defines runtime execution switches."""

    dry_run: bool = False
    no_merge: bool = False


@dataclass(frozen=True)
class FineTuningExecutionContext:
    """Aggregates all contracts required by the fine-tuning execution lifecycle."""

    dataset_contract: DatasetContract
    model_loading_contract: ModelLoadingContract
    lora_adaptation_contract: LoRAAdaptationContract
    training_hyperparameter_contract: TrainingHyperparameterContract
    artifact_persistence_contract: ArtifactPersistenceContract
    runtime_directive: RuntimeExecutionDirective


@dataclass(frozen=True)
class TrainableParameterEnvelope:
    """Contains trainable and total parameter counts."""

    trainable_parameters: int
    total_parameters: int

    @property
    def trainable_percentage(self) -> float:
        """Return the trainable parameter percentage."""
        return 100 * self.trainable_parameters / self.total_parameters


@dataclass(frozen=True)
class GpuMemoryEnvelope:
    """Contains a CUDA memory snapshot."""

    device_name: str
    total_memory_gb: float
    reserved_memory_gb: float


@dataclass(frozen=True)
class TrainingMetricsEnvelope:
    """Contains runtime metrics required by the training summary contract."""

    train_runtime_s: float
    train_loss: float
    vram_used_gb: float
    max_memory_gb: float
    training_sample_count: int
    validation_sample_count: int


class JsonLinesRepositoryProtocol(Protocol):
    def load_records(self, file_path: Path, encoding: str) -> list[JsonRecord]:
        """Load JSONL records from the provided file path."""


class ConversationDatasetMaterializerProtocol(Protocol):
    def materialize_dataset(
        self,
        records: list[JsonRecord],
        conversation_key: str,
    ) -> Any:
        """Materialize records into a trainer-compatible dataset."""


class ModelMaterializationGatewayProtocol(Protocol):
    def materialize_model_and_tokenizer(
        self,
        model_loading_contract: ModelLoadingContract,
    ) -> tuple[Any, Any]:
        """Materialize the model and tokenizer."""


class LoRAAdaptationGatewayProtocol(Protocol):
    def apply_lora_adapter(
        self,
        model: Any,
        lora_contract: LoRAAdaptationContract,
    ) -> Any:
        """Apply the LoRA adapter contract to a model."""


class JsonLinesRecordRepository:
    """Loads JSONL records without altering record values."""

    def load_records(self, file_path: Path, encoding: str) -> list[JsonRecord]:
        records: list[JsonRecord] = []

        with open(file_path, encoding=encoding) as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))

        return records


class HuggingFaceConversationDatasetMaterializer:
    """Materializes conversation records as a HuggingFace Dataset."""

    def materialize_dataset(
        self,
        records: list[JsonRecord],
        conversation_key: str,
    ) -> Any:
        try:
            from datasets import Dataset

            rows = []
            for record in records:
                rows.append({conversation_key: record[conversation_key]})

            return Dataset.from_list(rows)
        except Exception as error:
            raise DatasetMaterializationError(
                "Conversation dataset materialization failed."
            ) from error


class UnslothModelMaterializationGateway:
    """Materializes the configured language model through Unsloth."""

    def materialize_model_and_tokenizer(
        self,
        model_loading_contract: ModelLoadingContract,
    ) -> tuple[Any, Any]:
        from unsloth import FastLanguageModel

        return FastLanguageModel.from_pretrained(
            model_name=model_loading_contract.model_id,
            max_seq_length=model_loading_contract.max_sequence_length,
            dtype=model_loading_contract.dtype,
            load_in_4bit=model_loading_contract.load_in_4bit,
        )


class UnslothLoRAAdaptationGateway:
    """Applies the configured LoRA adapter through Unsloth."""

    def apply_lora_adapter(
        self,
        model: Any,
        lora_contract: LoRAAdaptationContract,
    ) -> Any:
        from unsloth import FastLanguageModel

        return FastLanguageModel.get_peft_model(
            model,
            r=lora_contract.rank,
            target_modules=list(lora_contract.target_modules),
            lora_alpha=lora_contract.alpha,
            lora_dropout=lora_contract.dropout,
            bias=lora_contract.bias,
            use_gradient_checkpointing=(
                lora_contract.gradient_checkpointing_mode
            ),
            random_state=lora_contract.random_state,
            use_rslora=lora_contract.use_rslora,
            loftq_config=lora_contract.loftq_config,
        )


class TrainableParameterInspectionService:
    """Inspects trainable and total model parameter counts."""

    def inspect(self, model: Any) -> TrainableParameterEnvelope:
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())

        return TrainableParameterEnvelope(
            trainable_parameters=trainable_parameters,
            total_parameters=total_parameters,
        )


class ChatTemplateConfigurationService:
    """Applies and executes the configured chat template contract."""

    def apply_template_to_tokenizer(self, tokenizer: Any) -> Any:
        from unsloth.chat_templates import get_chat_template

        return get_chat_template(
            tokenizer,
            chat_template=DEFAULT_CHAT_TEMPLATE_NAME,
        )

    def apply_template_to_dataset(self, dataset: Any, tokenizer: Any) -> Any:
        def format_conversations(examples: JsonObject) -> JsonObject:
            texts = tokenizer.apply_chat_template(
                examples["conversations"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": texts}

        return dataset.map(format_conversations, batched=True)


class TrainerAssemblyService:
    """Builds the supervised fine-tuning trainer from execution contracts."""

    def assemble_trainer(
        self,
        model: Any,
        tokenizer: Any,
        training_dataset: Any,
        validation_dataset: Any,
        execution_context: FineTuningExecutionContext,
    ) -> Any:
        try:
            from trl import SFTConfig, SFTTrainer

            hyperparameters = (
                execution_context.training_hyperparameter_contract
            )
            artifacts = execution_context.artifact_persistence_contract
            artifacts.lora_output_dir.mkdir(parents=True, exist_ok=True)

            return SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=training_dataset,
                eval_dataset=validation_dataset,
                args=SFTConfig(
                    dataset_text_field=hyperparameters.dataset_text_field,
                    max_seq_length=hyperparameters.max_sequence_length,
                    per_device_train_batch_size=hyperparameters.batch_size,
                    per_device_eval_batch_size=hyperparameters.batch_size,
                    gradient_accumulation_steps=(
                        hyperparameters.gradient_accumulation_steps
                    ),
                    num_train_epochs=hyperparameters.epochs,
                    learning_rate=hyperparameters.learning_rate,
                    warmup_ratio=hyperparameters.warmup_ratio,
                    lr_scheduler_type=hyperparameters.lr_scheduler_type,
                    weight_decay=hyperparameters.weight_decay,
                    fp16=hyperparameters.fp16,
                    bf16=hyperparameters.bf16,
                    logging_steps=hyperparameters.logging_steps,
                    eval_strategy=hyperparameters.eval_strategy,
                    eval_steps=hyperparameters.eval_steps,
                    save_strategy=hyperparameters.save_strategy,
                    save_steps=hyperparameters.save_steps,
                    save_total_limit=hyperparameters.save_total_limit,
                    load_best_model_at_end=(
                        hyperparameters.load_best_model_at_end
                    ),
                    metric_for_best_model=hyperparameters.metric_for_best_model,
                    greater_is_better=hyperparameters.greater_is_better,
                    output_dir=str(artifacts.lora_output_dir),
                    report_to=hyperparameters.report_to,
                    dataloader_num_workers=(
                        hyperparameters.dataloader_num_workers
                    ),
                    seed=hyperparameters.seed,
                    optim=hyperparameters.optim,
                    packing=hyperparameters.packing,
                    dataset_num_proc=hyperparameters.dataset_num_proc,
                ),
            )
        except Exception as error:
            raise TrainerAssemblyError(
                "Supervised fine-tuning trainer assembly failed."
            ) from error


class CudaRuntimeInspectionGateway:
    """Inspects CUDA runtime memory without altering training behavior."""

    def capture_pre_training_memory(self) -> GpuMemoryEnvelope:
        import torch

        gpu_stats = torch.cuda.get_device_properties(0)
        reserved_memory_gb = round(
            torch.cuda.max_memory_reserved() / 1024**3,
            1,
        )
        total_memory_gb = round(gpu_stats.total_memory / 1024**3, 1)

        return GpuMemoryEnvelope(
            device_name=gpu_stats.name,
            total_memory_gb=total_memory_gb,
            reserved_memory_gb=reserved_memory_gb,
        )

    def capture_reserved_memory_gb(self) -> float:
        import torch

        return round(torch.cuda.max_memory_reserved() / 1024**3, 1)


class ModelArtifactPersistenceService:
    """Persists LoRA and merged model artifacts."""

    def save_lora_adapter(
        self,
        model: Any,
        tokenizer: Any,
        artifact_contract: ArtifactPersistenceContract,
    ) -> None:
        artifact_contract.lora_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(artifact_contract.lora_output_dir))
        tokenizer.save_pretrained(str(artifact_contract.lora_output_dir))

    def save_merged_model(
        self,
        model: Any,
        tokenizer: Any,
        artifact_contract: ArtifactPersistenceContract,
    ) -> None:
        artifact_contract.merged_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(
            str(artifact_contract.merged_output_dir),
            tokenizer,
            save_method=artifact_contract.merged_save_method,
        )


class TrainingSummaryPersistenceService:
    """Persists the training summary JSON artifact."""

    def build_summary_payload(
        self,
        execution_context: FineTuningExecutionContext,
        metrics: TrainingMetricsEnvelope,
    ) -> JsonObject:
        model_contract = execution_context.model_loading_contract
        lora_contract = execution_context.lora_adaptation_contract
        hyperparameters = execution_context.training_hyperparameter_contract
        artifacts = execution_context.artifact_persistence_contract

        return {
            "model_id": model_contract.model_id,
            "lora_r": lora_contract.rank,
            "lora_alpha": lora_contract.alpha,
            "lora_dropout": lora_contract.dropout,
            "lora_targets": list(lora_contract.target_modules),
            "epochs": hyperparameters.epochs,
            "learning_rate": hyperparameters.learning_rate,
            "batch_effective": hyperparameters.effective_batch_size,
            "max_seq_length": model_contract.max_sequence_length,
            "train_samples": metrics.training_sample_count,
            "val_samples": metrics.validation_sample_count,
            "train_runtime_s": metrics.train_runtime_s,
            "train_loss": metrics.train_loss,
            "vram_used_gb": metrics.vram_used_gb,
            "output_lora": str(artifacts.lora_output_dir.resolve()),
            "output_merged": str(artifacts.merged_output_dir.resolve()),
        }

    def persist_summary(
        self,
        summary_payload: JsonObject,
        artifact_contract: ArtifactPersistenceContract,
    ) -> Path:
        summary_file_path = artifact_contract.training_summary_file_path

        try:
            with open(summary_file_path, "w", encoding=DEFAULT_ENCODING) as file:
                json.dump(summary_payload, file, ensure_ascii=False, indent=2)
        except Exception as error:
            raise ArtifactPersistenceError(
                "Training summary persistence failed."
            ) from error

        return summary_file_path


class ExecutionTelemetryPresenter:
    """Presents neutral operational execution messages."""

    def present_execution_contract(
        self,
        execution_context: FineTuningExecutionContext,
    ) -> None:
        model_contract = execution_context.model_loading_contract
        dataset_contract = execution_context.dataset_contract
        lora_contract = execution_context.lora_adaptation_contract
        hyperparameters = execution_context.training_hyperparameter_contract
        artifacts = execution_context.artifact_persistence_contract

        print(f"\n{'=' * 60}")
        print("LoRA supervised fine-tuning pipeline")
        print(f"{'=' * 60}")
        print(f"  Base model       : {model_contract.model_id}")
        print(f"  Train file       : {dataset_contract.training_file_path}")
        print(f"  Validation file  : {dataset_contract.validation_file_path}")
        print(
            f"  LoRA rank        : {lora_contract.rank}  "
            f"alpha: {lora_contract.alpha}"
        )
        print(
            f"  Effective batch  : "
            f"{hyperparameters.effective_batch_size}"
        )
        print(f"  Epochs           : {hyperparameters.epochs}")
        print(f"  Learning rate    : {hyperparameters.learning_rate}")
        print(f"  Max seq length   : {model_contract.max_sequence_length}")
        print(f"  LoRA output      : {artifacts.lora_output_dir}")
        print(f"  Merged output    : {artifacts.merged_output_dir}")

    def present_model_loading_started(self, model_id: str) -> None:
        print(f"\n[1/5] Loading model {model_id}...")

    def present_lora_application_started(
        self,
        lora_contract: LoRAAdaptationContract,
    ) -> None:
        print(
            f"\n[2/5] Applying LoRA "
            f"(r={lora_contract.rank}, alpha={lora_contract.alpha})..."
        )

    def present_trainable_parameters(
        self,
        parameter_envelope: TrainableParameterEnvelope,
    ) -> None:
        print(
            f"  Trainable parameters: "
            f"{parameter_envelope.trainable_parameters:,} / "
            f"{parameter_envelope.total_parameters:,} "
            f"({parameter_envelope.trainable_percentage:.2f}%)"
        )

    def present_dry_run_completed(self) -> None:
        print(
            "\n[DRY-RUN] Execution configuration validated. "
            "Training was not started."
        )

    def present_dataset_loading_started(self) -> None:
        print("\n[3/5] Loading datasets...")

    def present_dataset_counts(
        self,
        training_sample_count: int,
        validation_sample_count: int,
    ) -> None:
        print(f"  Train: {training_sample_count} records")
        print(f"  Val  : {validation_sample_count} records")

    def present_training_started(self) -> None:
        print("\n[4/5] Starting training...")

    def present_cuda_memory(self, memory_envelope: GpuMemoryEnvelope) -> None:
        print(
            f"  GPU: {memory_envelope.device_name}  "
            f"Total VRAM: {memory_envelope.total_memory_gb}GB  "
            f"Reserved: {memory_envelope.reserved_memory_gb}GB"
        )

    def present_training_completed(
        self,
        metrics: TrainingMetricsEnvelope,
    ) -> None:
        print("\n  Training completed:")
        print(
            f"  Total time    : {metrics.train_runtime_s:.0f}s "
            f"({metrics.train_runtime_s / 60:.1f} min)"
        )
        print(f"  Final loss    : {metrics.train_loss:.4f}")
        print(
            f"  VRAM used     : {metrics.vram_used_gb}GB / "
            f"{metrics.max_memory_gb}GB"
        )

    def present_lora_persistence_started(self, output_dir: Path) -> None:
        print(f"\n  Saving LoRA adapter to {output_dir}...")

    def present_no_merge_completed(self) -> None:
        print("\n[--no-merge] LoRA adapter saved. Merge was not executed.")

    def present_merge_started(self, merged_output_dir: Path) -> None:
        print("\n[5/5] Merging LoRA adapter with base model...")
        print(f"  Output: {merged_output_dir}")

    def present_pipeline_completed(
        self,
        artifact_contract: ArtifactPersistenceContract,
    ) -> None:
        print(f"\n{'=' * 60}")
        print("Pipeline execution completed.")
        print(f"  LoRA adapter : {artifact_contract.lora_output_dir.resolve()}")
        print(f"  Merged model : {artifact_contract.merged_output_dir.resolve()}")
        print(f"{'=' * 60}")

    def present_summary_persisted(self, summary_file_path: Path) -> None:
        print(f"\n  Summary saved to: {summary_file_path}")


class FineTuningLifecycleOrchestrator:
    """Coordinates the LoRA fine-tuning lifecycle."""

    def __init__(
        self,
        record_repository: JsonLinesRepositoryProtocol,
        dataset_materializer: ConversationDatasetMaterializerProtocol,
        model_gateway: ModelMaterializationGatewayProtocol,
        lora_gateway: LoRAAdaptationGatewayProtocol,
        parameter_inspector: TrainableParameterInspectionService,
        chat_template_service: ChatTemplateConfigurationService,
        trainer_assembly_service: TrainerAssemblyService,
        cuda_gateway: CudaRuntimeInspectionGateway,
        artifact_service: ModelArtifactPersistenceService,
        summary_service: TrainingSummaryPersistenceService,
        telemetry_presenter: ExecutionTelemetryPresenter,
    ) -> None:
        self._record_repository = record_repository
        self._dataset_materializer = dataset_materializer
        self._model_gateway = model_gateway
        self._lora_gateway = lora_gateway
        self._parameter_inspector = parameter_inspector
        self._chat_template_service = chat_template_service
        self._trainer_assembly_service = trainer_assembly_service
        self._cuda_gateway = cuda_gateway
        self._artifact_service = artifact_service
        self._summary_service = summary_service
        self._telemetry_presenter = telemetry_presenter

    def execute(
        self,
        execution_context: FineTuningExecutionContext,
    ) -> None:
        self._telemetry_presenter.present_execution_contract(execution_context)

        model, tokenizer = self._materialize_model_and_tokenizer(
            execution_context,
        )
        model = self._apply_lora_adaptation(model, execution_context)
        self._present_trainable_parameter_contract(model)

        if execution_context.runtime_directive.dry_run:
            self._telemetry_presenter.present_dry_run_completed()
            return

        training_records, validation_records = self._load_training_records(
            execution_context,
        )
        training_dataset, validation_dataset = self._materialize_datasets(
            training_records=training_records,
            validation_records=validation_records,
            execution_context=execution_context,
        )
        tokenizer, training_dataset, validation_dataset = (
            self._apply_chat_template_contract(
                tokenizer=tokenizer,
                training_dataset=training_dataset,
                validation_dataset=validation_dataset,
            )
        )
        trainer = self._assemble_training_lifecycle(
            model=model,
            tokenizer=tokenizer,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            execution_context=execution_context,
        )
        metrics = self._execute_training_lifecycle(
            trainer=trainer,
            model=model,
            training_sample_count=len(training_records),
            validation_sample_count=len(validation_records),
        )
        self._persist_lora_adapter(
            model=model,
            tokenizer=tokenizer,
            execution_context=execution_context,
        )

        if execution_context.runtime_directive.no_merge:
            self._telemetry_presenter.present_no_merge_completed()
            return

        self._persist_merged_model(
            model=model,
            tokenizer=tokenizer,
            execution_context=execution_context,
        )
        self._persist_training_summary(
            metrics=metrics,
            execution_context=execution_context,
        )

    def _materialize_model_and_tokenizer(
        self,
        execution_context: FineTuningExecutionContext,
    ) -> tuple[Any, Any]:
        self._telemetry_presenter.present_model_loading_started(
            execution_context.model_loading_contract.model_id,
        )
        return self._model_gateway.materialize_model_and_tokenizer(
            execution_context.model_loading_contract,
        )

    def _apply_lora_adaptation(
        self,
        model: Any,
        execution_context: FineTuningExecutionContext,
    ) -> Any:
        self._telemetry_presenter.present_lora_application_started(
            execution_context.lora_adaptation_contract,
        )
        return self._lora_gateway.apply_lora_adapter(
            model=model,
            lora_contract=execution_context.lora_adaptation_contract,
        )

    def _present_trainable_parameter_contract(self, model: Any) -> None:
        parameter_envelope = self._parameter_inspector.inspect(model)
        self._telemetry_presenter.present_trainable_parameters(
            parameter_envelope,
        )

    def _load_training_records(
        self,
        execution_context: FineTuningExecutionContext,
    ) -> tuple[list[JsonRecord], list[JsonRecord]]:
        dataset_contract = execution_context.dataset_contract
        self._telemetry_presenter.present_dataset_loading_started()
        training_records = self._record_repository.load_records(
            dataset_contract.training_file_path,
            dataset_contract.encoding,
        )
        validation_records = self._record_repository.load_records(
            dataset_contract.validation_file_path,
            dataset_contract.encoding,
        )
        self._telemetry_presenter.present_dataset_counts(
            training_sample_count=len(training_records),
            validation_sample_count=len(validation_records),
        )
        return training_records, validation_records

    def _materialize_datasets(
        self,
        training_records: list[JsonRecord],
        validation_records: list[JsonRecord],
        execution_context: FineTuningExecutionContext,
    ) -> tuple[Any, Any]:
        conversation_key = execution_context.dataset_contract.record_conversation_key
        training_dataset = self._dataset_materializer.materialize_dataset(
            records=training_records,
            conversation_key=conversation_key,
        )
        validation_dataset = self._dataset_materializer.materialize_dataset(
            records=validation_records,
            conversation_key=conversation_key,
        )
        return training_dataset, validation_dataset

    def _apply_chat_template_contract(
        self,
        tokenizer: Any,
        training_dataset: Any,
        validation_dataset: Any,
    ) -> tuple[Any, Any, Any]:
        tokenizer = self._chat_template_service.apply_template_to_tokenizer(
            tokenizer,
        )
        training_dataset = self._chat_template_service.apply_template_to_dataset(
            training_dataset,
            tokenizer,
        )
        validation_dataset = self._chat_template_service.apply_template_to_dataset(
            validation_dataset,
            tokenizer,
        )
        return tokenizer, training_dataset, validation_dataset

    def _assemble_training_lifecycle(
        self,
        model: Any,
        tokenizer: Any,
        training_dataset: Any,
        validation_dataset: Any,
        execution_context: FineTuningExecutionContext,
    ) -> Any:
        self._telemetry_presenter.present_training_started()
        return self._trainer_assembly_service.assemble_trainer(
            model=model,
            tokenizer=tokenizer,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            execution_context=execution_context,
        )

    def _execute_training_lifecycle(
        self,
        trainer: Any,
        model: Any,
        training_sample_count: int,
        validation_sample_count: int,
    ) -> TrainingMetricsEnvelope:
        model.config.use_cache = False
        memory_envelope = self._cuda_gateway.capture_pre_training_memory()
        self._telemetry_presenter.present_cuda_memory(memory_envelope)

        trainer_stats = trainer.train()
        used_memory_gb = self._cuda_gateway.capture_reserved_memory_gb()

        metrics = TrainingMetricsEnvelope(
            train_runtime_s=trainer_stats.metrics["train_runtime"],
            train_loss=trainer_stats.metrics["train_loss"],
            vram_used_gb=used_memory_gb,
            max_memory_gb=memory_envelope.total_memory_gb,
            training_sample_count=training_sample_count,
            validation_sample_count=validation_sample_count,
        )
        self._telemetry_presenter.present_training_completed(metrics)
        return metrics

    def _persist_lora_adapter(
        self,
        model: Any,
        tokenizer: Any,
        execution_context: FineTuningExecutionContext,
    ) -> None:
        artifact_contract = execution_context.artifact_persistence_contract
        self._telemetry_presenter.present_lora_persistence_started(
            artifact_contract.lora_output_dir,
        )
        self._artifact_service.save_lora_adapter(
            model=model,
            tokenizer=tokenizer,
            artifact_contract=artifact_contract,
        )

    def _persist_merged_model(
        self,
        model: Any,
        tokenizer: Any,
        execution_context: FineTuningExecutionContext,
    ) -> None:
        artifact_contract = execution_context.artifact_persistence_contract
        self._telemetry_presenter.present_merge_started(
            artifact_contract.merged_output_dir,
        )
        self._artifact_service.save_merged_model(
            model=model,
            tokenizer=tokenizer,
            artifact_contract=artifact_contract,
        )
        self._telemetry_presenter.present_pipeline_completed(
            artifact_contract,
        )

    def _persist_training_summary(
        self,
        metrics: TrainingMetricsEnvelope,
        execution_context: FineTuningExecutionContext,
    ) -> None:
        artifact_contract = execution_context.artifact_persistence_contract
        summary_payload = self._summary_service.build_summary_payload(
            execution_context=execution_context,
            metrics=metrics,
        )
        summary_file_path = self._summary_service.persist_summary(
            summary_payload=summary_payload,
            artifact_contract=artifact_contract,
        )
        self._telemetry_presenter.present_summary_persisted(summary_file_path)


class FineTuningExecutionContextFactory:
    """Builds the fine-tuning execution context from module-level contracts."""

    def build(
        self,
        dry_run: bool,
        no_merge: bool,
    ) -> FineTuningExecutionContext:
        return FineTuningExecutionContext(
            dataset_contract=DatasetContract(
                training_file_path=TRAIN_FILE,
                validation_file_path=VAL_FILE,
            ),
            model_loading_contract=ModelLoadingContract(
                model_id=MODEL_ID,
                max_sequence_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=False,
            ),
            lora_adaptation_contract=LoRAAdaptationContract(
                rank=LORA_R,
                alpha=LORA_ALPHA,
                dropout=LORA_DROPOUT,
                target_modules=tuple(LORA_TARGETS),
            ),
            training_hyperparameter_contract=TrainingHyperparameterContract(
                batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                warmup_ratio=WARMUP_RATIO,
                lr_scheduler_type=LR_SCHEDULER,
                weight_decay=WEIGHT_DECAY,
                max_sequence_length=MAX_SEQ_LENGTH,
            ),
            artifact_persistence_contract=ArtifactPersistenceContract(
                lora_output_dir=OUTPUT_DIR,
                merged_output_dir=MERGED_DIR,
            ),
            runtime_directive=RuntimeExecutionDirective(
                dry_run=dry_run,
                no_merge=no_merge,
            ),
        )


class FineTuningLifecycleOrchestratorFactory:
    """Builds the fine-tuning orchestrator."""

    def build(self) -> FineTuningLifecycleOrchestrator:
        return FineTuningLifecycleOrchestrator(
            record_repository=JsonLinesRecordRepository(),
            dataset_materializer=HuggingFaceConversationDatasetMaterializer(),
            model_gateway=UnslothModelMaterializationGateway(),
            lora_gateway=UnslothLoRAAdaptationGateway(),
            parameter_inspector=TrainableParameterInspectionService(),
            chat_template_service=ChatTemplateConfigurationService(),
            trainer_assembly_service=TrainerAssemblyService(),
            cuda_gateway=CudaRuntimeInspectionGateway(),
            artifact_service=ModelArtifactPersistenceService(),
            summary_service=TrainingSummaryPersistenceService(),
            telemetry_presenter=ExecutionTelemetryPresenter(),
        )


def load_jsonl(path: Path) -> list[JsonRecord]:
    """Load JSONL records using the preserved module-level encoding."""
    return JsonLinesRecordRepository().load_records(path, DEFAULT_ENCODING)


def to_hf_dataset(records: list[JsonRecord]) -> Any:
    """Materialize records into a HuggingFace Dataset."""
    return HuggingFaceConversationDatasetMaterializer().materialize_dataset(
        records=records,
        conversation_key="conversations",
    )


def run_finetune(dry_run: bool = False, no_merge: bool = False) -> None:
    """Run the LoRA fine-tuning lifecycle."""
    execution_context = FineTuningExecutionContextFactory().build(
        dry_run=dry_run,
        no_merge=no_merge,
    )
    orchestrator = FineTuningLifecycleOrchestratorFactory().build()
    orchestrator.execute(execution_context)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning with Unsloth and optional model merge"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without starting training",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Train and save the LoRA adapter without merging the model",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=LORA_R,
        help=f"LoRA rank (default: {LORA_R})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    return parser


def main() -> None:
    """Execute the command-line fine-tuning lifecycle."""
    global LORA_R, LORA_ALPHA, EPOCHS, LEARNING_RATE

    parser = build_argument_parser()
    args = parser.parse_args()

    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_r
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr

    run_finetune(dry_run=args.dry_run, no_merge=args.no_merge)


if __name__ == "__main__":
    main()
