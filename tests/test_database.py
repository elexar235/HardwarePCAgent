import pytest
from pathlib import Path
from src.db.database import HardwareDatabase


@pytest.fixture
def temp_db(tmp_path: Path):
    db_file = tmp_path / "test_parts.db"
    return HardwareDatabase(db_path=db_file)


def test_database_seeding(temp_db: HardwareDatabase):
    cpus = temp_db.list_cpus()
    assert len(cpus) >= 5
    gpus = temp_db.list_gpus()
    assert len(gpus) >= 5


def test_query_filter_socket(temp_db: HardwareDatabase):
    am5_cpus = temp_db.list_cpus(socket="AM5")
    assert all(cpu.socket == "AM5" for cpu in am5_cpus)
    assert any("7800X3D" in cpu.name for cpu in am5_cpus)


def test_query_filter_gpu_price(temp_db: HardwareDatabase):
    budget_gpus = temp_db.list_gpus(max_price=400)
    assert all(gpu.price_usd <= 400 for gpu in budget_gpus)
