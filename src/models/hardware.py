"""Hardware component schemas for PC building."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

SocketType = Literal["AM4", "AM5", "LGA1700", "LGA1851"]
RAMType = Literal["DDR4", "DDR5"]
MotherboardFormFactor = Literal["ATX", "Micro-ATX", "Mini-ITX"]
StorageType = Literal["NVMe M.2 SSD", "SATA SSD", "HDD"]


class HardwareItem(BaseModel):
    id: str
    name: str
    price_usd: int
    brand: str


class CPU(HardwareItem):
    socket: SocketType
    cores: int
    threads: int
    tdp_w: int
    supports_ram: List[RAMType]
    has_integrated_gpu: bool = False
    benchmark_score: int = Field(
        ..., description="Relative CPU performance score (higher is better)"
    )


class Motherboard(HardwareItem):
    socket: SocketType
    ram_type: RAMType
    ram_slots: int
    form_factor: MotherboardFormFactor
    pcie_version: str = "4.0"
    m2_slots_count: int = 2


class GPU(HardwareItem):
    chip: str
    vram_gb: int
    tdp_w: int
    recommended_psu_w: int
    length_mm: int
    benchmark_score: int = Field(
        ..., description="Relative GPU performance score (e.g. 3DMark/TimeSpy)"
    )


class RAM(HardwareItem):
    ram_type: RAMType
    capacity_gb: int
    speed_mhz: int
    modules_count: int


class Storage(HardwareItem):
    storage_type: StorageType
    capacity_gb: int
    read_speed_mb_s: int


class PSU(HardwareItem):
    wattage_w: int
    certification: Literal["Standard", "Bronze", "Gold", "Platinum", "Titanium"]
    is_modular: bool = False


class Cooler(HardwareItem):
    supported_sockets: List[SocketType]
    max_tdp_w: int
    height_mm: int


class Case(HardwareItem):
    supported_form_factors: List[MotherboardFormFactor]
    max_gpu_length_mm: int
    max_cooler_height_mm: int
