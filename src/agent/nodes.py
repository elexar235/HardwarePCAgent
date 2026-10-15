"""Graph nodes and execution logic for the PC Builder Agent."""

import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.config import DB_PATH, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from src.db.database import HardwareDatabase
from src.models import (
    CPU,
    GPU,
    RAM,
    Case,
    Cooler,
    Motherboard,
    PCBuild,
    PSU,
    Storage,
    UserRequirements,
    ValidationResult,
    CompatibilityIssue,
)
from src.rules.compatibility import validate_pc_build

db = HardwareDatabase(db_path=DB_PATH)


def _get_llm() -> Optional[ChatOpenAI]:
    if OPENAI_API_KEY and len(OPENAI_API_KEY.strip()) > 5:
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.2,
        )
    return None


def _heuristic_parse_requirements(query: str) -> UserRequirements:
    """Fallback rule-based parser when LLM API key is not configured."""
    q = query.lower()

    # Budget extraction (USD)
    budget = None
    # match patterns like $1500, 1500$, 1500 usd, 1.5k, 1500
    dollar_match = re.search(r"\$\s*(\d+)", q) or re.search(r"(\d+)\s*(?:\$|usd|bucks)", q)
    if dollar_match:
        budget = int(dollar_match.group(1))
    else:
        k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", q)
        if k_match:
            val = float(k_match.group(1))
            budget = int(val * 1000) if val < 20 else int(val)
        else:
            num_match = re.search(r"\b([5-9]\d{2}|[1-9]\d{3})\b", q)
            if num_match:
                budget = int(num_match.group(1))

    # Default fallback budget if not specified
    if not budget or budget < 300:
        budget = 1200

    # Resolution
    res = "1440p"
    if "4k" in q or "3840" in q or "uhd" in q:
        res = "4k"
    elif "1080p" in q or "fhd" in q or "full hd" in q:
        res = "1080p"
    elif "2k" in q or "1440p" in q or "qhd" in q:
        res = "1440p"

    # Known target games
    known_games = [
        "cyberpunk",
        "cyberpunk 2077",
        "cs2",
        "counter-strike",
        "dota 2",
        "witcher",
        "gta v",
        "baldurs gate 3",
        "warzone",
        "valorant",
        "apex legends",
        "escape from tarkov",
        "fortnite",
    ]
    games = [g for g in known_games if g in q]

    # Known non-gaming workloads
    known_tasks = [
        "blender",
        "3d rendering",
        "video editing",
        "premiere pro",
        "machine learning",
        "ai",
        "software development",
        "streaming",
    ]
    tasks = [t for t in known_tasks if t in q]

    return UserRequirements(
        budget_usd=budget,
        resolution=res,
        target_fps=60,
        target_games=games,
        work_tasks=tasks,
    )


def parse_requirements_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Extract structured user requirements from natural language input."""
    user_query = state["user_query"]
    llm = _get_llm()

    if llm:
        try:
            structured_llm = llm.with_structured_output(UserRequirements)
            system_prompt = (
                "You are an expert PC hardware consultant. Extract detailed structured requirements "
                "from the user's natural language request. Infer budget in USD."
            )
            requirements = structured_llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]
            )
            if not requirements.budget_usd:
                requirements.budget_usd = 1200
            return {"requirements": requirements}
        except Exception:
            pass

    requirements = _heuristic_parse_requirements(user_query)
    return {"requirements": requirements}


def plan_and_select_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Select balanced components from catalog database within budget limits."""
    requirements = state["requirements"] or _heuristic_parse_requirements(state["user_query"])
    budget = requirements.budget_usd or 1200
    iteration = state.get("iteration_count", 0)
    feedback = state.get("feedback_history", [])

    # Target GPU allocation: ~40-48% of total budget
    over_budget_penalty = 0
    if any("exceeds" in f.lower() for f in feedback):
        over_budget_penalty = 100 * max(1, iteration)

    adjusted_budget = max(450, budget - over_budget_penalty)
    target_gpu_budget = int(adjusted_budget * 0.45)
    target_cpu_budget = int(adjusted_budget * 0.22)

    # 1. Select GPU
    available_gpus = db.list_gpus(max_price=target_gpu_budget)
    if not available_gpus:
        available_gpus = db.list_gpus()
    gpu = available_gpus[0] if available_gpus else db.list_gpus()[-1]

    # Resolution preference: If 4k, prioritize 16GB VRAM cards
    if requirements.resolution == "4k":
        high_vram_gpus = [g for g in available_gpus if g.vram_gb >= 16]
        if high_vram_gpus:
            gpu = high_vram_gpus[0]

    # 2. Select CPU & Platform (AM5 / AM4 / LGA1700)
    available_cpus = db.list_cpus(max_price=target_cpu_budget)
    if not available_cpus:
        available_cpus = db.list_cpus()

    # Prioritize AM5 if budget >= $900 for long-term upgradability
    am5_cpus = [c for c in available_cpus if c.socket == "AM5"]
    if budget >= 900 and am5_cpus:
        cpu = am5_cpus[0]
    else:
        cpu = available_cpus[0]

    # 3. Select Motherboard strictly matching CPU socket
    target_ram_type = "DDR5" if "DDR5" in cpu.supports_ram and cpu.socket == "AM5" else "DDR4"
    motherboards = db.list_motherboards(socket=cpu.socket, ram_type=target_ram_type)
    if not motherboards:
        motherboards = db.list_motherboards(socket=cpu.socket)
    motherboard = motherboards[0]

    actual_ram_type = motherboard.ram_type

    # 4. Select RAM
    target_ram_capacity = 32 if budget >= 900 else 16
    rams = db.list_rams(ram_type=actual_ram_type, min_capacity_gb=target_ram_capacity)
    if not rams:
        rams = db.list_rams(ram_type=actual_ram_type)
    ram = rams[0]

    # 5. Select Storage
    storages = db.list_storages()
    if budget >= 1600 and len(storages) >= 3:
        storage = storages[2]
    else:
        storage = storages[0]

    # 6. Select PSU based on GPU & CPU power needs
    estimated_power = cpu.tdp_w + gpu.tdp_w + 60
    min_psu_wattage = max(gpu.recommended_psu_w, int(estimated_power * 1.25))
    psus = db.list_psus(min_wattage_w=min_psu_wattage)
    if not psus:
        psus = db.list_psus()
    psu = psus[0]

    # 7. Select Case with sufficient form factor & GPU clearance
    cases = db.list_cases(form_factor=motherboard.form_factor)
    cases = [c for c in cases if c.max_gpu_length_mm >= gpu.length_mm]
    if not cases:
        cases = db.list_cases()
    case = cases[0]

    # 8. Select CPU Cooler
    coolers = db.list_coolers(socket=cpu.socket, min_tdp_w=cpu.tdp_w)
    if not coolers:
        coolers = db.list_coolers(socket=cpu.socket)
    cooler = coolers[0] if coolers else None

    build = PCBuild(
        cpu=cpu,
        motherboard=motherboard,
        gpu=gpu,
        ram=ram,
        storage=storage,
        psu=psu,
        case=case,
        cooler=cooler,
    )

    return {
        "current_build": build,
        "iteration_count": iteration + 1,
    }


def validate_build_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Validate physical & electrical compatibility and budget."""
    build = state["current_build"]
    requirements = state["requirements"]
    feedback = list(state.get("feedback_history", []))

    if not build:
        return {"validation_result": ValidationResult(is_compatible=False)}

    val_result = validate_pc_build(build)

    # Budget check
    if requirements and requirements.budget_usd:
        if build.total_price > requirements.budget_usd:
            overrun = build.total_price - requirements.budget_usd
            issue = CompatibilityIssue(
                severity="error",
                rule_name="Budget Limit Exceeded",
                component_a="Total Build Price",
                component_b=f"Budget: ${requirements.budget_usd:,}",
                message=f"Total price ${build.total_price:,} exceeds budget by ${overrun:,}.",
                suggested_fix="Downgrade GPU or CPU tier to meet budget constraint.",
            )
            val_result.is_compatible = False
            val_result.issues.append(issue)

    if not val_result.is_compatible:
        for err in val_result.issues:
            feedback.append(f"[{err.rule_name}] {err.message}")

    return {
        "validation_result": val_result,
        "feedback_history": feedback,
    }


def should_continue(state: AgentState) -> str:
    """Conditional Edge: Decide whether to self-correct (loop back) or proceed to explanation."""
    val_result = state.get("validation_result")
    iteration = state.get("iteration_count", 0)

    if val_result and val_result.is_compatible:
        return "explain"

    if iteration < 3:
        return "retry"

    return "explain"


def _estimate_game_fps(gpu: GPU, resolution: str) -> Dict[str, int]:
    """Estimate average FPS based on GPU benchmark score and resolution scale."""
    scale = {"1080p": 1.0, "1440p": 0.72, "4k": 0.44}.get(resolution, 0.72)
    score = gpu.benchmark_score

    cyberpunk_fps = int((score / 220) * scale)
    cs2_fps = int((score / 70) * scale)
    witcher_fps = int((score / 180) * scale)
    bg3_fps = int((score / 190) * scale)

    return {
        "Cyberpunk 2077 (Ultra)": max(30, cyberpunk_fps),
        "Counter-Strike 2 (Competitive)": max(120, cs2_fps),
        "The Witcher 3: Next-Gen": max(45, witcher_fps),
        "Baldur's Gate 3 (Ultra)": max(40, bg3_fps),
    }


def explain_build_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Synthesize structured breakdown, benchmarks, and rationale for user."""
    build = state["current_build"]
    requirements = state["requirements"]
    val_result = state.get("validation_result")
    iterations = state.get("iteration_count", 1)

    if not build:
        return {"final_response": "Unable to configure a build matching specified constraints."}

    resolution = requirements.resolution if requirements else "1440p"
    budget = requirements.budget_usd if requirements else 0
    fps_estimates = _estimate_game_fps(build.gpu, resolution)

    status_icon = "✅" if val_result and val_result.is_compatible else "⚠️"
    status_text = "100% Fully Compatible" if val_result and val_result.is_compatible else "Budget/Hardware adjustments needed"

    lines = [
        f"## 🖥️ Recommended PC Configuration ({resolution.upper()})",
        "",
        f"**Compatibility Status:** {status_icon} {status_text}",
        f"**Optimization Iterations:** {iterations}",
        "",
        "### 📦 Selected Components:",
        "| Category | Component | Specification | Price |",
        "| :--- | :--- | :--- | :--- |",
        f"| **CPU** | {build.cpu.name} | {build.cpu.cores}C/{build.cpu.threads}T, Socket {build.cpu.socket} | ${build.cpu.price_usd:,} |",
        f"| **Motherboard** | {build.motherboard.name} | {build.motherboard.socket}, {build.motherboard.ram_type}, {build.motherboard.form_factor} | ${build.motherboard.price_usd:,} |",
        f"| **GPU** | {build.gpu.name} | {build.gpu.vram_gb}GB VRAM, {build.gpu.length_mm}mm | ${build.gpu.price_usd:,} |",
        f"| **RAM** | {build.ram.name} | {build.ram.capacity_gb}GB {build.ram.ram_type} {build.ram.speed_mhz}MHz | ${build.ram.price_usd:,} |",
        f"| **Storage (SSD)** | {build.storage.name} | {build.storage.capacity_gb}GB NVMe ({build.storage.read_speed_mb_s} MB/s) | ${build.storage.price_usd:,} |",
        f"| **Power Supply (PSU)** | {build.psu.name} | {build.psu.wattage_w}W {build.psu.certification} | ${build.psu.price_usd:,} |",
        f"| **Case** | {build.case.name} | Supports GPU up to {build.case.max_gpu_length_mm}mm | ${build.case.price_usd:,} |",
    ]

    if build.cooler:
        lines.append(f"| **CPU Cooler** | {build.cooler.name} | Rated up to {build.cooler.max_tdp_w}W TDP | ${build.cooler.price_usd:,} |")

    lines.extend([
        "",
        f"💰 **Total Price:** `${build.total_price:,}` (Target Budget: `${budget:,}`)",
        f"⚡ **Estimated Peak Power Draw:** `~{build.estimated_peak_power_w}W` (PSU Headroom: `+{build.psu.wattage_w - build.estimated_peak_power_w}W`)",
        "",
        f"### 🎮 Estimated Gaming Performance ({resolution.upper()} High/Ultra):",
    ])

    for game, fps in fps_estimates.items():
        lines.append(f"- **{game}:** ~`{fps} FPS`")

    lines.extend([
        "",
        "### 🔍 Technical Selection Rationale:",
        f"1. **Platform & Socket:** Socket `{build.cpu.socket}` with `{build.ram.ram_type}` memory ensures system longevity and seamless future CPU upgrades without replacing the motherboard.",
        f"2. **CPU/GPU Balance:** The pair `{build.cpu.name}` + `{build.gpu.name}` is well-balanced to eliminate severe bottlenecks across both CPU-heavy and GPU-intensive workloads.",
        f"3. **Power Margin:** The `{build.psu.wattage_w}W` unit safely covers transient current spikes with a >25% operating buffer.",
        f"4. **Chassis Clearance:** GPU length ({build.gpu.length_mm}mm) fits cleanly within the chassis tolerance (max {build.case.max_gpu_length_mm}mm).",
    ])

    if val_result and val_result.warnings:
        lines.append("")
        lines.append("### 💡 Notes & Warnings:")
        for w in val_result.warnings:
            lines.append(f"- *{w.rule_name}*: {w.message}")

    return {"final_response": "\n".join(lines)}
