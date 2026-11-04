from enum import StrEnum


class ModelKind(StrEnum):
    FULL = "full"
    FROZEN_CORE = "frozen_core"
    ACTIVE = "active"


class MappingKind(StrEnum):
    JORDAN_WIGNER = "jordan_wigner"
    PARITY = "parity"


class AnsatzKind(StrEnum):
    UCCSD = "uccsd"
    HEA = "hea"


class Optimizer(StrEnum):
    SLSQP = "SLSQP"
    COBYLA = "COBYLA"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
