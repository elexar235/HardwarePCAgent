import pytest
from src.models import (
    CPU,
    GPU,
    RAM,
    Motherboard,
    PSU,
    Storage,
    Case,
    Cooler,
    PCBuild,
)
from src.rules.compatibility import validate_pc_build


@pytest.fixture
def valid_am5_build():
    return PCBuild(
        cpu=CPU(
            id="cpu-r5-7500f",
            name="AMD Ryzen 5 7500F",
            brand="AMD",
            socket="AM5",
            cores=6,
            threads=12,
            tdp_w=65,
            supports_ram=["DDR5"],
            has_integrated_gpu=False,
            price_usd=145,
            benchmark_score=16000,
        ),
        motherboard=Motherboard(
            id="mb-b650m-hdv",
            name="ASRock B650M-HDV/M.2",
            brand="ASRock",
            socket="AM5",
            ram_type="DDR5",
            ram_slots=2,
            form_factor="Micro-ATX",
            pcie_version="4.0",
            m2_slots_count=2,
            price_usd=119,
        ),
        gpu=GPU(
            id="gpu-rx9060xt-16g",
            name="Sapphire PULSE AMD Radeon RX 9060 XT 16GB",
            brand="Sapphire",
            chip="RX 9060 XT",
            vram_gb=16,
            tdp_w=160,
            recommended_psu_w=550,
            length_mm=250,
            price_usd=369,
            benchmark_score=16500,
        ),
        ram=RAM(
            id="ram-32g-d5-6000",
            name="Kingston FURY Beast 32GB DDR5 6000MHz",
            brand="Kingston",
            ram_type="DDR5",
            capacity_gb=32,
            speed_mhz=6000,
            modules_count=2,
            price_usd=279,
        ),
        storage=Storage(
            id="ssd-1tb-nv2",
            name="Kingston NV2 1TB",
            brand="Kingston",
            storage_type="NVMe M.2 SSD",
            capacity_gb=1000,
            read_speed_mb_s=3500,
            price_usd=55,
        ),
        psu=PSU(
            id="psu-650w-bronze",
            name="DeepCool PK650D 650W Bronze",
            brand="DeepCool",
            wattage_w=650,
            certification="Bronze",
            is_modular=False,
            price_usd=55,
        ),
        case=Case(
            id="case-montech-air100",
            name="Montech AIR 100 LITE",
            brand="Montech",
            supported_form_factors=["Micro-ATX", "Mini-ITX"],
            max_gpu_length_mm=330,
            max_cooler_height_mm=161,
            price_usd=45,
        ),
        cooler=Cooler(
            id="cooler-ag400",
            name="DeepCool AG400",
            brand="DeepCool",
            supported_sockets=["AM4", "AM5", "LGA1700", "LGA1851"],
            max_tdp_w=220,
            height_mm=150,
            price_usd=22,
        ),
    )


def test_valid_build_passes(valid_am5_build: PCBuild):
    result = validate_pc_build(valid_am5_build)
    assert result.is_compatible is True
    assert len(result.issues) == 0


def test_igpu_office_build_passes(valid_am5_build: PCBuild):
    # System with NO discrete GPU, but CPU has integrated graphics
    office_build = valid_am5_build.model_copy(deep=True)
    office_build.gpu = None
    office_build.cpu.has_integrated_gpu = True
    office_build.cpu.name = "AMD Ryzen 5 8500G (Radeon 740M iGPU)"

    result = validate_pc_build(office_build)
    assert result.is_compatible is True
    assert len(result.issues) == 0


def test_missing_video_output_fails(valid_am5_build: PCBuild):
    # System with NO discrete GPU and CPU has NO integrated graphics
    headless_build = valid_am5_build.model_copy(deep=True)
    headless_build.gpu = None
    headless_build.cpu.has_integrated_gpu = False

    result = validate_pc_build(headless_build)
    assert result.is_compatible is False
    assert any("No Video Output" in issue.rule_name for issue in result.issues)


def test_socket_mismatch_fails(valid_am5_build: PCBuild):
    invalid_build = valid_am5_build.model_copy(deep=True)
    invalid_build.motherboard.socket = "LGA1851"
    invalid_build.motherboard.name = "MSI Z890 Motherboard"

    result = validate_pc_build(invalid_build)
    assert result.is_compatible is False
    assert any("Socket Mismatch" in issue.rule_name for issue in result.issues)


def test_ram_type_mismatch_fails(valid_am5_build: PCBuild):
    invalid_build = valid_am5_build.model_copy(deep=True)
    invalid_build.ram.ram_type = "DDR4"
    invalid_build.ram.name = "DDR4 RAM Kit"

    result = validate_pc_build(invalid_build)
    assert result.is_compatible is False
    assert any("RAM Type Mismatch" in issue.rule_name for issue in result.issues)


def test_gpu_too_long_fails(valid_am5_build: PCBuild):
    invalid_build = valid_am5_build.model_copy(deep=True)
    invalid_build.gpu.length_mm = 350

    result = validate_pc_build(invalid_build)
    assert result.is_compatible is False
    assert any("GPU Length Exceeded" in issue.rule_name for issue in result.issues)


def test_psu_underpowered_fails(valid_am5_build: PCBuild):
    invalid_build = valid_am5_build.model_copy(deep=True)
    invalid_build.psu.wattage_w = 200

    result = validate_pc_build(invalid_build)
    assert result.is_compatible is False
    assert any("Insufficient Power" in issue.rule_name for issue in result.issues)
