"""SQLite Database manager and catalog query engine."""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional
from src.models.hardware import (
    CPU,
    GPU,
    RAM,
    Case,
    Cooler,
    Motherboard,
    PSU,
    Storage,
    MotherboardFormFactor,
    RAMType,
    SocketType,
)
from src.db.seed_data import (
    CASES_SEED,
    COOLERS_SEED,
    CPUS_SEED,
    GPUS_SEED,
    MOTHERBOARDS_SEED,
    PSUS_SEED,
    RAMS_SEED,
    STORAGES_SEED,
)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "pc_parts.db"


class HardwareDatabase:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables and seed if empty."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cpus (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    brand TEXT,
                    socket TEXT,
                    cores INTEGER,
                    threads INTEGER,
                    tdp_w INTEGER,
                    supports_ram TEXT,
                    price_usd INTEGER,
                    benchmark_score INTEGER
                )
            """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS motherboards (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    brand TEXT,
                    socket TEXT,
                    ram_type TEXT,
                    ram_slots INTEGER,
                    form_factor TEXT,
                    pcie_version TEXT,
                    m2_slots_count INTEGER,
                    price_usd INTEGER
                )
            """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS gpus (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    brand TEXT,
                    chip TEXT,
                    vram_gb INTEGER,
                    tdp_w INTEGER,
                    recommended_psu_w INTEGER,
                    length_mm INTEGER,
                    price_usd INTEGER,
                    benchmark_score INTEGER
                )
            """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rams (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    brand TEXT,
                    ram_type TEXT,
                    capacity_gb INTEGER,
                    speed_mhz INTEGER,
                    modules_count INTEGER,
                    price_usd INTEGER
                )
            """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS storages (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    brand TEXT,
                    storage_type TEXT,
                    capacity_gb INTEGER,
                    read_speed_mb_s INTEGER,
                    price_usd INTEGER
                )
            """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS psus (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    brand TEXT,
                    wattage_w INTEGER,
                    certification TEXT,
                    is_modular BOOLEAN,
                    price_usd INTEGER
                )
            """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    brand TEXT,
                    supported_form_factors TEXT,
                    max_gpu_length_mm INTEGER,
                    max_cooler_height_mm INTEGER,
                    price_usd INTEGER
                )
            """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS coolers (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    brand TEXT,
                    supported_sockets TEXT,
                    max_tdp_w INTEGER,
                    height_mm INTEGER,
                    price_usd INTEGER
                )
            """
            )
            conn.commit()

            # Seed if empty
            cur.execute("SELECT COUNT(*) as count FROM cpus")
            if cur.fetchone()["count"] == 0:
                self._seed_database(conn)

    def _seed_database(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        for item in CPUS_SEED:
            cur.execute(
                """
                INSERT INTO cpus VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    item["id"],
                    item["name"],
                    item["brand"],
                    item["socket"],
                    item["cores"],
                    item["threads"],
                    item["tdp_w"],
                    json.dumps(item["supports_ram"]),
                    item["price_usd"],
                    item["benchmark_score"],
                ),
            )
        for item in MOTHERBOARDS_SEED:
            cur.execute(
                """
                INSERT INTO motherboards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    item["id"],
                    item["name"],
                    item["brand"],
                    item["socket"],
                    item["ram_type"],
                    item["ram_slots"],
                    item["form_factor"],
                    item["pcie_version"],
                    item["m2_slots_count"],
                    item["price_usd"],
                ),
            )
        for item in GPUS_SEED:
            cur.execute(
                """
                INSERT INTO gpus VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    item["id"],
                    item["name"],
                    item["brand"],
                    item["chip"],
                    item["vram_gb"],
                    item["tdp_w"],
                    item["recommended_psu_w"],
                    item["length_mm"],
                    item["price_usd"],
                    item["benchmark_score"],
                ),
            )
        for item in RAMS_SEED:
            cur.execute(
                """
                INSERT INTO rams VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    item["id"],
                    item["name"],
                    item["brand"],
                    item["ram_type"],
                    item["capacity_gb"],
                    item["speed_mhz"],
                    item["modules_count"],
                    item["price_usd"],
                ),
            )
        for item in STORAGES_SEED:
            cur.execute(
                """
                INSERT INTO storages VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    item["id"],
                    item["name"],
                    item["brand"],
                    item["storage_type"],
                    item["capacity_gb"],
                    item["read_speed_mb_s"],
                    item["price_usd"],
                ),
            )
        for item in PSUS_SEED:
            cur.execute(
                """
                INSERT INTO psus VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    item["id"],
                    item["name"],
                    item["brand"],
                    item["wattage_w"],
                    item["certification"],
                    item["is_modular"],
                    item["price_usd"],
                ),
            )
        for item in CASES_SEED:
            cur.execute(
                """
                INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    item["id"],
                    item["name"],
                    item["brand"],
                    json.dumps(item["supported_form_factors"]),
                    item["max_gpu_length_mm"],
                    item["max_cooler_height_mm"],
                    item["price_usd"],
                ),
            )
        for item in COOLERS_SEED:
            cur.execute(
                """
                INSERT INTO coolers VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    item["id"],
                    item["name"],
                    item["brand"],
                    json.dumps(item["supported_sockets"]),
                    item["max_tdp_w"],
                    item["height_mm"],
                    item["price_usd"],
                ),
            )
        conn.commit()

    # Query methods
    def list_cpus(
        self, socket: Optional[SocketType] = None, max_price: Optional[int] = None
    ) -> List[CPU]:
        query = "SELECT * FROM cpus WHERE 1=1"
        params = []
        if socket:
            query += " AND socket = ?"
            params.append(socket)
        if max_price:
            query += " AND price_usd <= ?"
            params.append(max_price)
        query += " ORDER BY benchmark_score DESC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                CPU(
                    id=row["id"],
                    name=row["name"],
                    brand=row["brand"],
                    socket=row["socket"],
                    cores=row["cores"],
                    threads=row["threads"],
                    tdp_w=row["tdp_w"],
                    supports_ram=json.loads(row["supports_ram"]),
                    price_usd=row["price_usd"],
                    benchmark_score=row["benchmark_score"],
                )
                for row in rows
            ]

    def list_motherboards(
        self,
        socket: Optional[SocketType] = None,
        ram_type: Optional[RAMType] = None,
        max_price: Optional[int] = None,
    ) -> List[Motherboard]:
        query = "SELECT * FROM motherboards WHERE 1=1"
        params = []
        if socket:
            query += " AND socket = ?"
            params.append(socket)
        if ram_type:
            query += " AND ram_type = ?"
            params.append(ram_type)
        if max_price:
            query += " AND price_usd <= ?"
            params.append(max_price)
        query += " ORDER BY price_usd ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                Motherboard(
                    id=row["id"],
                    name=row["name"],
                    brand=row["brand"],
                    socket=row["socket"],
                    ram_type=row["ram_type"],
                    ram_slots=row["ram_slots"],
                    form_factor=row["form_factor"],
                    pcie_version=row["pcie_version"],
                    m2_slots_count=row["m2_slots_count"],
                    price_usd=row["price_usd"],
                )
                for row in rows
            ]

    def list_gpus(
        self, max_price: Optional[int] = None, min_vram_gb: Optional[int] = None
    ) -> List[GPU]:
        query = "SELECT * FROM gpus WHERE 1=1"
        params = []
        if max_price:
            query += " AND price_usd <= ?"
            params.append(max_price)
        if min_vram_gb:
            query += " AND vram_gb >= ?"
            params.append(min_vram_gb)
        query += " ORDER BY benchmark_score DESC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                GPU(
                    id=row["id"],
                    name=row["name"],
                    brand=row["brand"],
                    chip=row["chip"],
                    vram_gb=row["vram_gb"],
                    tdp_w=row["tdp_w"],
                    recommended_psu_w=row["recommended_psu_w"],
                    length_mm=row["length_mm"],
                    price_usd=row["price_usd"],
                    benchmark_score=row["benchmark_score"],
                )
                for row in rows
            ]

    def list_rams(
        self,
        ram_type: Optional[RAMType] = None,
        min_capacity_gb: Optional[int] = None,
        max_price: Optional[int] = None,
    ) -> List[RAM]:
        query = "SELECT * FROM rams WHERE 1=1"
        params = []
        if ram_type:
            query += " AND ram_type = ?"
            params.append(ram_type)
        if min_capacity_gb:
            query += " AND capacity_gb >= ?"
            params.append(min_capacity_gb)
        if max_price:
            query += " AND price_usd <= ?"
            params.append(max_price)
        query += " ORDER BY price_usd ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                RAM(
                    id=row["id"],
                    name=row["name"],
                    brand=row["brand"],
                    ram_type=row["ram_type"],
                    capacity_gb=row["capacity_gb"],
                    speed_mhz=row["speed_mhz"],
                    modules_count=row["modules_count"],
                    price_usd=row["price_usd"],
                )
                for row in rows
            ]

    def list_storages(self, min_capacity_gb: Optional[int] = None) -> List[Storage]:
        query = "SELECT * FROM storages WHERE 1=1"
        params = []
        if min_capacity_gb:
            query += " AND capacity_gb >= ?"
            params.append(min_capacity_gb)
        query += " ORDER BY price_usd ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                Storage(
                    id=row["id"],
                    name=row["name"],
                    brand=row["brand"],
                    storage_type=row["storage_type"],
                    capacity_gb=row["capacity_gb"],
                    read_speed_mb_s=row["read_speed_mb_s"],
                    price_usd=row["price_usd"],
                )
                for row in rows
            ]

    def list_psus(self, min_wattage_w: Optional[int] = None) -> List[PSU]:
        query = "SELECT * FROM psus WHERE 1=1"
        params = []
        if min_wattage_w:
            query += " AND wattage_w >= ?"
            params.append(min_wattage_w)
        query += " ORDER BY price_usd ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                PSU(
                    id=row["id"],
                    name=row["name"],
                    brand=row["brand"],
                    wattage_w=row["wattage_w"],
                    certification=row["certification"],
                    is_modular=bool(row["is_modular"]),
                    price_usd=row["price_usd"],
                )
                for row in rows
            ]

    def list_cases(
        self, form_factor: Optional[MotherboardFormFactor] = None
    ) -> List[Case]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM cases ORDER BY price_usd ASC").fetchall()
            cases = [
                Case(
                    id=row["id"],
                    name=row["name"],
                    brand=row["brand"],
                    supported_form_factors=json.loads(row["supported_form_factors"]),
                    max_gpu_length_mm=row["max_gpu_length_mm"],
                    max_cooler_height_mm=row["max_cooler_height_mm"],
                    price_usd=row["price_usd"],
                )
                for row in rows
            ]
            if form_factor:
                cases = [c for c in cases if form_factor in c.supported_form_factors]
            return cases

    def list_coolers(
        self, socket: Optional[SocketType] = None, min_tdp_w: Optional[int] = None
    ) -> List[Cooler]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM coolers ORDER BY price_usd ASC"
            ).fetchall()
            coolers = [
                Cooler(
                    id=row["id"],
                    name=row["name"],
                    brand=row["brand"],
                    supported_sockets=json.loads(row["supported_sockets"]),
                    max_tdp_w=row["max_tdp_w"],
                    height_mm=row["height_mm"],
                    price_usd=row["price_usd"],
                )
                for row in rows
            ]
            if socket:
                coolers = [c for c in coolers if socket in c.supported_sockets]
            if min_tdp_w:
                coolers = [c for c in coolers if c.max_tdp_w >= min_tdp_w]
            return coolers
