"""PC build definition and compatibility validation schemas."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from src.models.hardware import CPU, Motherboard, GPU, RAM, Storage, PSU, Cooler, Case


class PCBuild(BaseModel):
    cpu: CPU
    motherboard: Motherboard
    gpu: Optional[GPU] = None  # None indicates usage of CPU integrated graphics
    ram: RAM
    storage: Storage
    psu: PSU
    case: Case
    cooler: Optional[Cooler] = None

    @property
    def total_price(self) -> int:
        price = (
            self.cpu.price_usd
            + self.motherboard.price_usd
            + self.ram.price_usd
            + self.storage.price_usd
            + self.psu.price_usd
            + self.case.price_usd
        )
        if self.gpu:
            price += self.gpu.price_usd
        if self.cooler:
            price += self.cooler.price_usd
        return price

    @property
    def estimated_peak_power_w(self) -> int:
        """Estimated maximum system power draw under heavy load."""
        base_system_draw = 50  # fans, motherboard, drives, peripherals
        gpu_draw = self.gpu.tdp_w if self.gpu else 0
        return self.cpu.tdp_w + gpu_draw + base_system_draw


class CompatibilityIssue(BaseModel):
    severity: Literal["error", "warning"]
    rule_name: str
    component_a: str
    component_b: str
    message: str
    suggested_fix: Optional[str] = None


class ValidationResult(BaseModel):
    is_compatible: bool
    issues: List[CompatibilityIssue] = Field(default_factory=list)
    warnings: List[CompatibilityIssue] = Field(default_factory=list)
