"""Comprehensive logical and regression tests for PC Builder Agent.

Validates heuristic parsing, workload classification, dynamic platform combo
evaluation, GPU utility scoring, and end-to-end LangGraph pipeline execution.
"""

import pytest
from src.agent.graph import app
from src.agent.nodes import (
    _determine_workload_profile,
    _heuristic_parse_requirements,
    _select_gpu,
    _select_platform_combo,
)
from src.config import DB_PATH
from src.db.database import HardwareDatabase
from src.models import GPU, CPU, UserRequirements

db = HardwareDatabase(db_path=DB_PATH)


@pytest.fixture(autouse=True)
def disable_external_llm(monkeypatch):
    """Ensure automated tests run deterministically against the heuristic engine without network/model lag."""
    monkeypatch.setattr("src.agent.nodes._get_llm", lambda: None)


# =========================================================================
# 1. Heuristic Requirements Parser Tests
# =========================================================================

def test_heuristic_parser_budget_formats():
    """Verify various natural language currency formats are accurately parsed."""
    test_cases = [
        ("$1200", 1200),
        ("1500$", 1500),
        ("800 usd", 800),
        ("950 bucks", 950),
        ("1.5k", 1500),
        ("2k", 2000),
        ("budget 1100", 1100),
        ("budget: 650", 650),
        ("budget is $450", 450),
    ]
    for query, expected_budget in test_cases:
        req = _heuristic_parse_requirements(query)
        assert req.budget_usd == expected_budget, f"Failed on query: '{query}'"


def test_heuristic_parser_cyberpunk_not_budget():
    """Ensure game title year in 'Cyberpunk 2077' is never mistaken for a $2077 budget."""
    req = _heuristic_parse_requirements("Build a PC for Cyberpunk 2077, budget $1300")
    assert req.budget_usd == 1300
    assert "cyberpunk" in req.target_games or "cyberpunk 2077" in req.target_games

    # When no budget is given at all with Cyberpunk 2077, it should fall back to default, not 2077
    req_no_budget = _heuristic_parse_requirements("Build a PC for Cyberpunk 2077 at 1440p")
    assert req_no_budget.budget_usd != 2077
    assert req_no_budget.budget_usd == 1200


def test_heuristic_parser_resolutions():
    """Verify explicit resolutions are captured and omitted resolutions remain None."""
    assert _heuristic_parse_requirements("Gaming in 4K resolution").resolution == "4k"
    assert _heuristic_parse_requirements("Need 3840x2160 UHD gaming").resolution == "4k"
    assert _heuristic_parse_requirements("1440p gaming build").resolution == "1440p"
    assert _heuristic_parse_requirements("2K display setup").resolution == "1440p"
    assert _heuristic_parse_requirements("1080p esports machine").resolution == "1080p"
    assert _heuristic_parse_requirements("Full HD gaming").resolution == "1080p"

    # When resolution is NOT specified, it must remain None (not forced to 1440p)
    assert _heuristic_parse_requirements("gaming PC, budget $1000").resolution is None
    assert _heuristic_parse_requirements("office PC for web browsing").resolution is None
    assert _heuristic_parse_requirements("ai models and machine learning").resolution is None


def test_heuristic_parser_gaming_keywords():
    """Verify general gaming intent is captured even when specific game titles are absent."""
    queries = [
        "PC for playing games, budget is $300",
        "cheap gaming computer",
        "PC to play games online",
        "need a gamer PC for $1000",
        "esports machine",
    ]
    for q in queries:
        req = _heuristic_parse_requirements(q)
        assert len(req.target_games) > 0, f"Failed to identify gaming intent in: '{q}'"


# =========================================================================
# 2. Profile Classification Tests
# =========================================================================

def test_profile_classification_office():
    """Office and browsing tasks must map to office profile."""
    for q in [
        "office PC for paperwork",
        "budget PC for web surfing and office, budget $450",
        "browser and typing machine",
    ]:
        req = _heuristic_parse_requirements(q)
        assert _determine_workload_profile(req) == "office"


def test_profile_classification_ai():
    """AI, ML, and 3D tasks must map to ai_3d profile."""
    for q in [
        "ai models and machine learning",
        "local llm workstation, budget $1500",
        "blender and 3d rendering setup",
    ]:
        req = _heuristic_parse_requirements(q)
        assert _determine_workload_profile(req) == "ai_3d"


def test_profile_classification_productivity():
    """Development, audio, and video editing tasks without games map to productivity."""
    for q in [
        "programming and software development PC",
        "music production in FL Studio with audio DAW",
        "video editing workstation, budget $1400",
    ]:
        req = _heuristic_parse_requirements(q)
        assert _determine_workload_profile(req) == "productivity"


def test_profile_gaming_priority_over_low_budget():
    """A gaming request with a low budget must remain gaming, NOT get hijacked to office."""
    req = _heuristic_parse_requirements("PC for playing games, budget is $300")
    profile = _determine_workload_profile(req)
    assert profile == "gaming", f"Expected 'gaming' profile, but got '{profile}'"


# =========================================================================
# 3. Dynamic Platform Combo Selection Tests
# =========================================================================

def test_platform_combo_office_always_has_igpu():
    """Office platform selection must strictly guarantee an integrated GPU."""
    all_cpus = db.list_cpus()
    cpu, mobo, ram = _select_platform_combo(
        all_cpus=all_cpus,
        target_platform_budget=300,
        target_ram_gb=16,
        require_igpu=True,
    )
    assert cpu.has_integrated_gpu is True
    assert cpu.socket == mobo.socket
    assert mobo.ram_type == ram.ram_type
    assert ram.ram_type in cpu.supports_ram


def test_platform_combo_hardware_compatibility():
    """Selected platform combo must be 100% physically and electrically compatible."""
    all_cpus = db.list_cpus()
    for budget in [250, 400, 600, 900]:
        cpu, mobo, ram = _select_platform_combo(
            all_cpus=all_cpus,
            target_platform_budget=budget,
            target_ram_gb=16,
            require_igpu=False,
        )
        assert cpu.socket == mobo.socket, f"Socket mismatch: {cpu.socket} != {mobo.socket}"
        assert mobo.ram_type == ram.ram_type, f"RAM type mismatch: {mobo.ram_type} != {ram.ram_type}"
        assert ram.ram_type in cpu.supports_ram, f"CPU does not support {ram.ram_type}"


def test_platform_combo_stays_in_budget_when_possible():
    """Platform combo should respect the platform budget ceiling when valid options exist."""
    all_cpus = db.list_cpus()
    platform_budget = 400
    cpu, mobo, ram = _select_platform_combo(
        all_cpus=all_cpus,
        target_platform_budget=platform_budget,
        target_ram_gb=16,
        require_igpu=False,
    )
    total_combo_cost = cpu.price_usd + mobo.price_usd + ram.price_usd
    assert total_combo_cost <= platform_budget


# =========================================================================
# 4. GPU Scoring & Selection Tests
# =========================================================================

def test_gpu_selection_ai_prefers_nvidia_vram():
    """For AI/ML, utility scoring must prioritize NVIDIA CUDA and VRAM capacity."""
    all_gpus = db.list_gpus()
    # At a budget of $500, RTX 4060 Ti 16GB ($449) should be selected for AI
    selected_gpu = _select_gpu(
        all_gpus=all_gpus,
        target_gpu_budget=500,
        profile="ai_3d",
    )
    assert selected_gpu is not None
    assert selected_gpu.vram_gb >= 12
    assert "nvidia" in selected_gpu.brand.lower() or "rtx" in selected_gpu.name.lower()


def test_gpu_selection_office_returns_none():
    """Office profile must never select a discrete GPU."""
    all_gpus = db.list_gpus()
    gpu = _select_gpu(all_gpus=all_gpus, target_gpu_budget=500, profile="office")
    assert gpu is None


def test_gpu_selection_gaming_maximizes_raster_score():
    """Gaming GPU selection must pick the highest benchmark score within budget."""
    all_gpus = db.list_gpus()
    target_budget = 400
    selected_gpu = _select_gpu(
        all_gpus=all_gpus,
        target_gpu_budget=target_budget,
        profile="gaming",
    )
    assert selected_gpu is not None
    assert selected_gpu.price_usd <= target_budget
    # Verify no cheaper card in budget had a higher benchmark score
    affordable = [g for g in all_gpus if g.price_usd <= target_budget]
    max_bench = max(g.benchmark_score for g in affordable)
    assert selected_gpu.benchmark_score == max_bench


# =========================================================================
# 5. End-to-End Pipeline & Integration Tests
# =========================================================================

def _run_pipeline(query: str):
    initial_state = {
        "user_query": query,
        "requirements": None,
        "budget_allocation": None,
        "current_build": None,
        "validation_result": None,
        "iteration_count": 0,
        "feedback_history": [],
        "final_response": None,
    }
    return app.invoke(initial_state)


def test_e2e_office_build_under_budget():
    """Office build must stay strictly under budget, use iGPU, and omit game benchmarks."""
    state = _run_pipeline("Build a budget PC for web surfing and office, budget $450")
    build = state["current_build"]
    val = state["validation_result"]
    resp = state["final_response"]

    assert build is not None
    assert build.gpu is None, "Office build must not have a discrete GPU"
    assert build.cpu.has_integrated_gpu is True, "Office CPU must have integrated graphics"
    assert build.total_price <= 450, f"Build price ${build.total_price} exceeds $450"
    assert val.is_compatible is True, "Build must be 100% compatible"
    assert "Cyberpunk" not in resp, "Office report must not contain gaming benchmarks"
    assert "FPS" not in resp, "Office report must not contain FPS"


def test_e2e_ai_build_within_budget():
    """AI build at $1200 must have high VRAM, NVIDIA GPU, and 32GB RAM within budget."""
    state = _run_pipeline("ai models, budget 1200$")
    build = state["current_build"]
    val = state["validation_result"]
    resp = state["final_response"]

    assert build is not None
    assert build.gpu is not None, "AI build must have dedicated GPU"
    assert build.gpu.vram_gb >= 12, "AI build requires at least 12GB VRAM"
    assert build.ram.capacity_gb >= 32, "AI build requires at least 32GB RAM"
    assert build.total_price <= 1200, f"Build price ${build.total_price} exceeds budget $1200"
    assert val.is_compatible is True
    assert "AI & Local LLM Capability" in resp, "Report must explain AI capabilities"
    assert "Cyberpunk" not in resp, "AI report must not contain game benchmarks"


def test_e2e_gaming_build_1080p_default():
    """Gaming build without explicit resolution must default to 1080P and show FPS."""
    state = _run_pipeline("gaming PC, budget 1000$")
    build = state["current_build"]
    val = state["validation_result"]
    resp = state["final_response"]

    assert build is not None
    assert build.gpu is not None
    assert build.total_price <= 1000
    assert val.is_compatible is True
    assert "1080P" in resp, "Default resolution must be 1080P"
    assert "Estimated Gaming Benchmarks (1080P" in resp
    assert "Cyberpunk 2077" in resp


def test_e2e_gaming_build_1440p_explicit():
    """Gaming build with explicit 1440p request must target 1440P."""
    state = _run_pipeline("Build a PC for Cyberpunk 2077 at 1440p, budget $1600")
    build = state["current_build"]
    val = state["validation_result"]
    resp = state["final_response"]

    assert build is not None
    assert build.gpu is not None
    assert val.is_compatible is True
    assert "1440P" in resp
    assert "Estimated Gaming Benchmarks (1440P" in resp


def test_e2e_gaming_impossible_budget_graceful():
    """Gaming build at $300 must report budget limit gracefully while still showing FPS."""
    state = _run_pipeline("PC for playing games, budget is $300")
    build = state["current_build"]
    val = state["validation_result"]
    resp = state["final_response"]

    assert build is not None
    assert val.is_compatible is False, "A $300 gaming build should exceed budget"
    assert any("Budget Limit Exceeded" in issue.rule_name for issue in val.issues)
    assert "High-Performance Gaming" in resp
    assert "1080P" in resp
    assert "Estimated Gaming Benchmarks" in resp, "FPS should still be estimated"


def test_e2e_power_headroom_always_guaranteed():
    """Verify that every generated build provides >= 25% PSU headroom over peak consumption."""
    queries = [
        "office PC, budget $500",
        "gaming PC, budget $1000",
        "workstation for video editing, budget $1500",
        "ai models, budget $1200",
    ]
    for q in queries:
        state = _run_pipeline(q)
        build = state["current_build"]
        assert build is not None
        min_required_psu = int(build.estimated_peak_power_w * 1.25)
        assert build.psu.wattage_w >= min_required_psu, (
            f"PSU wattage {build.psu.wattage_w}W is lower than required 125% ({min_required_psu}W) for query: '{q}'"
        )


def test_e2e_physical_clearance_always_guaranteed():
    """Verify physical GPU length, cooler height, and motherboard mounting across builds."""
    queries = [
        "gaming PC, budget $800",
        "gaming PC, budget $1200",
        "ai models, budget $1500",
    ]
    for q in queries:
        state = _run_pipeline(q)
        build = state["current_build"]
        assert build is not None
        if build.gpu:
            assert build.gpu.length_mm <= build.case.max_gpu_length_mm
        if build.cooler:
            assert build.cooler.height_mm <= build.case.max_cooler_height_mm
        assert build.motherboard.form_factor in build.case.supported_form_factors


# =========================================================================
# 6. Advanced Edge-Case & Regression Tests
# =========================================================================

def test_heuristic_parser_comma_currencies():
    """Verify natural comma formatting in numbers ($1,500, 2,000$) is parsed without truncation."""
    test_cases = [
        ("$1,500", 1500),
        ("1,200$", 1200),
        ("budget: $2,500", 2500),
        ("3,000 usd", 3000),
        ("build pc for 1,800 bucks", 1800),
    ]
    for query, expected_budget in test_cases:
        req = _heuristic_parse_requirements(query)
        assert req.budget_usd == expected_budget, f"Failed on query: '{query}', got {req.budget_usd}"


def test_heuristic_parser_hardware_models_not_confused_as_budget():
    """Verify GPU/CPU model numbers (RTX 4070, RX 7600, Ryzen 5600) are never confused for budgets."""
    # When no budget is specified, must fall back to default (1200), not the model number
    queries_default_budget = [
        "Gaming PC with RTX 4070",
        "PC build with RX 7600",
        "Ryzen 5600 system",
        "Arc A580 gaming build",
        "Gaming rig with RTX 4060",
    ]
    for q in queries_default_budget:
        req = _heuristic_parse_requirements(q)
        assert req.budget_usd == 1200, f"Query '{q}' misparsed model number as budget: {req.budget_usd}"

    # When explicit budget is present alongside hardware model, the explicit budget must win
    req_explicit = _heuristic_parse_requirements("Gaming PC with RTX 4070, budget $1500")
    assert req_explicit.budget_usd == 1500


def test_heuristic_parser_ram_and_storage_priorities():
    """Verify explicit RAM and storage capacity requests are recorded in priorities."""
    req1 = _heuristic_parse_requirements("Gaming PC with 32GB RAM, budget $1500")
    assert "ram_32gb" in req1.priorities

    req2 = _heuristic_parse_requirements("Workstation with 64GB RAM and 2TB SSD, budget $2500")
    assert "ram_64gb" in req2.priorities
    assert "storage_2000gb" in req2.priorities

    req3 = _heuristic_parse_requirements("Cheap APU esports machine for $400")
    assert "apu" in req3.priorities


def test_e2e_apu_gaming_build_succeeds_under_budget():
    """APU gaming build at $400 must configure Ryzen 5600G with Vega 7 iGPU under budget."""
    state = _run_pipeline("APU gaming build, budget $400")
    build = state["current_build"]
    val = state["validation_result"]
    resp = state["final_response"]

    assert build is not None
    assert build.gpu is None, "APU build must not include a discrete GPU"
    assert build.cpu.has_integrated_gpu is True, "APU build must have integrated GPU"
    assert build.total_price <= 400, f"APU build ${build.total_price} exceeded $400"
    assert val.is_compatible is True
    assert "Integrated Graphics" in resp
    assert "Counter-Strike 2" in resp
    assert "Valorant" in resp


def test_e2e_explicit_ram_and_storage_honored():
    """Verify that user-requested 32GB RAM and 2TB SSD are selected in the final build."""
    state = _run_pipeline("Gaming PC with 32GB RAM and 2TB SSD, budget $2000")
    build = state["current_build"]
    val = state["validation_result"]

    assert build is not None
    assert build.ram.capacity_gb >= 32, f"Expected >= 32GB RAM, got {build.ram.capacity_gb}GB"
    assert build.storage.capacity_gb >= 2000, f"Expected >= 2000GB SSD, got {build.storage.capacity_gb}GB"
    assert build.total_price <= 2000
    assert val.is_compatible is True


def test_e2e_high_budget_picks_top_tier_components():
    """High-budget build ($3500) must pick top-tier GPU, CPU, and PSU without errors."""
    state = _run_pipeline("Ultimate 4K gaming PC, budget $3500")
    build = state["current_build"]
    val = state["validation_result"]
    resp = state["final_response"]

    assert build is not None
    assert build.gpu is not None
    assert build.gpu.vram_gb >= 16, "High-end 4K build must have high VRAM"
    assert build.psu.wattage_w >= 850
    assert build.total_price <= 3500
    assert val.is_compatible is True
    assert "4K High/Ultra" in resp or "4K" in resp


def test_e2e_ultra_low_budget_graceful_handling():
    """Ultra low budget ($150) must not crash and gracefully explain the budget deficit."""
    state = _run_pipeline("Gaming PC, budget $150")
    build = state["current_build"]
    val = state["validation_result"]
    resp = state["final_response"]

    assert build is not None
    assert val.is_compatible is False
    assert "Budget Constraint Notice" in resp
    assert "exceeds your budget of **$150**" in resp


def test_e2e_various_workload_profiles_produce_safe_reports():
    """Diverse real-world requests across all profiles produce valid reports without errors."""
    queries = [
        "3D Blender workstation, budget $1800",
        "Music production with FL Studio, budget $1100",
        "Office computer with 500GB SSD, budget $400",
        "Esports gaming at 1080p, budget $750",
    ]
    for q in queries:
        state = _run_pipeline(q)
        assert state["final_response"] is not None
        assert len(state["final_response"]) > 100
        assert "Recommended PC Configuration" in state["final_response"]
        assert "Total Price" in state["final_response"]


def test_should_continue_conditional_routing():
    """Verify state routing: 'explain' when compatible or max iterations reached, 'retry' otherwise."""
    from src.agent.nodes import should_continue
    from src.models.build import ValidationResult

    # 1. Compatible -> must proceed to 'explain'
    state_compatible = {
        "validation_result": ValidationResult(is_compatible=True),
        "iteration_count": 1,
    }
    assert should_continue(state_compatible) == "explain"

    # 2. Incompatible at iteration 1 -> must 'retry'
    state_retry_1 = {
        "validation_result": ValidationResult(is_compatible=False),
        "iteration_count": 1,
    }
    assert should_continue(state_retry_1) == "retry"

    # 3. Incompatible at iteration 2 -> must 'retry'
    state_retry_2 = {
        "validation_result": ValidationResult(is_compatible=False),
        "iteration_count": 2,
    }
    assert should_continue(state_retry_2) == "retry"

    # 4. Incompatible at iteration 3 -> max iterations reached, must proceed to 'explain'
    state_max_iter = {
        "validation_result": ValidationResult(is_compatible=False),
        "iteration_count": 3,
    }
    assert should_continue(state_max_iter) == "explain"


def test_plan_and_select_feedback_penalty():
    """Verify that feedback with budget overrun increases over_budget_penalty on subsequent iteration."""
    from src.agent.nodes import plan_and_select_node

    req = _heuristic_parse_requirements("Gaming PC, budget $1000")
    # Simulate iteration 1 with feedback indicating an overrun of $150
    state = {
        "user_query": "Gaming PC, budget $1000",
        "requirements": req,
        "iteration_count": 1,
        "feedback_history": ["[Budget Limit Exceeded] Total price $1,150 exceeds budget by $150."],
    }
    result = plan_and_select_node(state)
    assert result["current_build"] is not None
    assert result["iteration_count"] == 2
    # Second iteration must produce a build that adjusts for the overrun
    assert result["current_build"].total_price <= 1000


def test_cli_main_execution():
    """Verify that src.main.run_agent executes cleanly without raising unhandled exceptions."""
    from src.main import run_agent

    # Test that run_agent runs end-to-end without unhandled exception
    try:
        run_agent("Office PC, budget $500")
    except Exception as exc:
        pytest.fail(f"run_agent raised an unexpected exception: {exc}")


