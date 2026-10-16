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
    dollar_match = re.search(r"\$\s*(\d+)", q) or re.search(r"(\d+)\s*(?:\$|usd|bucks)", q)
    if dollar_match:
        budget = int(dollar_match.group(1))
    else:
        k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", q)
        if k_match:
            val = float(k_match.group(1))
            budget = int(val * 1000) if val < 20 else int(val)
        else:
            num_match = re.search(r"\b([3-9]\d{2}|[1-9]\d{3})\b", q)
            if num_match:
                budget = int(num_match.group(1))

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
        "local llm",
        "software development",
        "programming",
        "coding",
        "audio",
        "music",
        "fl studio",
        "daw",
        "streaming",
        "office",
        "surfing",
        "browser",
        "web surfing",
    ]
    tasks = [t for t in known_tasks if t in q]

    # Fallback default budget
    if not budget:
        budget = 450 if any(t in ["office", "surfing", "browser", "web surfing"] for t in tasks) else 1200

    return UserRequirements(
        budget_usd=budget,
        resolution=res,
        target_fps=60,
        target_games=games,
        work_tasks=tasks,
    )


def _determine_workload_profile(req: UserRequirements) -> str:
    """Classify user intent into a specific workload profile."""
    tasks = [t.lower() for t in req.work_tasks]
    games = [g.lower() for g in req.target_games]

    has_heavy_gpu_tasks = any(
        k in tasks for k in ["3d rendering", "blender", "machine learning", "ai", "local llm"]
    )
    has_productivity = any(
        k in tasks
        for k in [
            "programming",
            "coding",
            "software development",
            "video editing",
            "audio",
            "music",
            "fl studio",
            "daw",
        ]
    )
    is_gaming = len(games) > 0 or req.resolution in ["1440p", "4k"]
    is_office = any(k in tasks for k in ["office", "surfing", "browser", "web surfing"])

    if has_heavy_gpu_tasks:
        return "ai_3d"
    if has_productivity and not is_gaming:
        return "productivity"
    if is_office or (not is_gaming and not has_productivity and (req.budget_usd or 0) <= 600):
        return "office"
    return "gaming"


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
    """Node 2: Select balanced components dynamically based on workload profile & budget."""
    requirements = state["requirements"] or _heuristic_parse_requirements(state["user_query"])
    budget = requirements.budget_usd or 1200
    iteration = state.get("iteration_count", 0)
    feedback = state.get("feedback_history", [])

    profile = _determine_workload_profile(requirements)

    over_budget_penalty = 0
    if any("exceeds" in f.lower() for f in feedback):
        over_budget_penalty = 80 * max(1, iteration)

    adjusted_budget = max(350, budget - over_budget_penalty)

    # 1. Determine CPU & GPU Allocation based on Profile
    gpu: Optional[GPU] = None

    if profile == "office":
        # Office / Web Surfing: prioritize CPU with integrated GPU (0$ discrete GPU!)
        target_cpu_budget = int(adjusted_budget * 0.40)
        target_ram_gb = 16
        # Find CPUs with integrated graphics
        igpu_cpus = [c for c in db.list_cpus(max_price=target_cpu_budget) if c.has_integrated_gpu]
        if not igpu_cpus:
            igpu_cpus = [c for c in db.list_cpus() if c.has_integrated_gpu]
        cpu = igpu_cpus[0] if igpu_cpus else db.list_cpus()[-1]
        gpu = None  # No discrete GPU needed!

    elif profile == "productivity":
        target_cpu_budget = int(adjusted_budget * 0.40)
        target_gpu_budget = int(adjusted_budget * 0.18)
        target_ram_gb = 64 if budget >= 1600 else 32

        # CPU first
        available_cpus = db.list_cpus(max_price=target_cpu_budget)
        cpu = available_cpus[0] if available_cpus else db.list_cpus()[0]

        # Discrete GPU (entry-to-mid for multi-monitor / video decoding)
        available_gpus = db.list_gpus(max_price=target_gpu_budget)
        gpu = available_gpus[0] if available_gpus else db.list_gpus()[-1]

    elif profile == "ai_3d":
        target_gpu_budget = int(adjusted_budget * 0.52)
        target_cpu_budget = int(adjusted_budget * 0.20)
        target_ram_gb = 64 if budget >= 2000 else 32

        available_gpus = db.list_gpus(max_price=target_gpu_budget)
        high_vram_gpus = [g for g in available_gpus if g.vram_gb >= 16]
        gpu = high_vram_gpus[0] if high_vram_gpus else (available_gpus[0] if available_gpus else db.list_gpus()[0])

        available_cpus = db.list_cpus(max_price=target_cpu_budget)
        cpu = available_cpus[0] if available_cpus else db.list_cpus()[0]

    else:  # gaming
        target_gpu_budget = int(adjusted_budget * 0.46)
        target_cpu_budget = int(adjusted_budget * 0.22)
        target_ram_gb = 32 if budget >= 1200 else 16

        available_gpus = db.list_gpus(max_price=target_gpu_budget)
        gpu = available_gpus[0] if available_gpus else db.list_gpus()[-1]

        if requirements.resolution == "4k":
            high_vram_gpus = [g for g in available_gpus if g.vram_gb >= 16]
            if high_vram_gpus:
                gpu = high_vram_gpus[0]

        available_cpus = db.list_cpus(max_price=target_cpu_budget)
        modern_cpus = [c for c in available_cpus if c.socket in ["AM5", "LGA1851"]]
        if budget >= 900 and modern_cpus:
            x3d_cpus = [c for c in modern_cpus if "X3D" in c.name]
            cpu = x3d_cpus[0] if x3d_cpus else modern_cpus[0]
        else:
            cpu = available_cpus[0] if available_cpus else db.list_cpus()[0]

    # 2. Select Motherboard strictly matching CPU socket
    target_ram_type = "DDR5" if "DDR5" in cpu.supports_ram and cpu.socket in ["AM5", "LGA1851"] else "DDR4"
    motherboards = db.list_motherboards(socket=cpu.socket, ram_type=target_ram_type)
    if not motherboards:
        motherboards = db.list_motherboards(socket=cpu.socket)
    motherboard = motherboards[0]

    actual_ram_type = motherboard.ram_type

    # 3. Select RAM
    rams = db.list_rams(ram_type=actual_ram_type, min_capacity_gb=target_ram_gb)
    if not rams:
        rams = db.list_rams(ram_type=actual_ram_type)
    ram = rams[0]

    # 4. Select Storage
    storages = db.list_storages()
    if profile == "office" or budget <= 600:
        storage = storages[0]  # The most affordable NVMe drive ($38 for 500GB)
    elif budget >= 2000:
        high_cap = [s for s in storages if s.capacity_gb >= 2000]
        storage = high_cap[0] if high_cap else storages[-1]
    elif budget >= 1200:
        one_tb = [s for s in storages if s.capacity_gb == 1000]
        storage = one_tb[-1] if one_tb else storages[0]
    else:
        storage = storages[0]

    # 5. Select PSU based on system load
    gpu_power = gpu.tdp_w if gpu else 0
    gpu_rec_psu = gpu.recommended_psu_w if gpu else 400
    estimated_power = cpu.tdp_w + gpu_power + 50
    min_psu_wattage = max(gpu_rec_psu, int(estimated_power * 1.25))

    psus = db.list_psus(min_wattage_w=min_psu_wattage)
    if not psus:
        psus = db.list_psus()
    psu = psus[0]

    # 6. Select Case
    cases = db.list_cases(form_factor=motherboard.form_factor)
    if gpu:
        cases = [c for c in cases if c.max_gpu_length_mm >= gpu.length_mm]
    if not cases:
        cases = db.list_cases()
    case = cases[0]

    # 7. Select Cooler
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


def _estimate_game_fps(gpu: Optional[GPU], resolution: str) -> Dict[str, int]:
    """Estimate average FPS based on GPU benchmark score and resolution scale."""
    if not gpu:
        return {}

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
    profile = _determine_workload_profile(requirements) if requirements else "gaming"
    fps_estimates = _estimate_game_fps(build.gpu, resolution)

    status_icon = "✅" if val_result and val_result.is_compatible else "⚠️"
    status_text = "100% Fully Compatible" if val_result and val_result.is_compatible else "Budget/Hardware adjustments needed"

    profile_names = {
        "gaming": "High-Performance Gaming",
        "productivity": "Workstation & Development",
        "ai_3d": "AI / ML & 3D Rendering",
        "office": "General Office & Web Surfing",
    }
    profile_label = profile_names.get(profile, "Custom Balanced")

    gpu_display_name = build.gpu.name if build.gpu else f"Integrated Graphics (Built into {build.cpu.name})"
    gpu_spec = f"{build.gpu.vram_gb}GB VRAM, {build.gpu.length_mm}mm" if build.gpu else "Shared System Memory"
    gpu_price = f"${build.gpu.price_usd:,}" if build.gpu else "$0 (Included)"

    lines = [
        f"## 🖥️ Recommended PC Configuration ({profile_label})",
        "",
        f"**Target Resolution:** `{resolution.upper()}` | **Profile:** `{profile_label}`",
        f"**Compatibility Status:** {status_icon} {status_text} (Optimized in {iterations} iteration{'s' if iterations > 1 else ''})",
        "",
        "### 📦 Selected Components:",
        "| Category | Component | Key Specification | Price |",
        "| :--- | :--- | :--- | :--- |",
        f"| **CPU** | {build.cpu.name} | {build.cpu.cores}C/{build.cpu.threads}T, Socket {build.cpu.socket} | ${build.cpu.price_usd:,} |",
        f"| **Motherboard** | {build.motherboard.name} | Socket {build.motherboard.socket}, {build.motherboard.ram_type}, {build.motherboard.form_factor} | ${build.motherboard.price_usd:,} |",
        f"| **GPU** | {gpu_display_name} | {gpu_spec} | {gpu_price} |",
        f"| **RAM** | {build.ram.name} | {build.ram.capacity_gb}GB {build.ram.ram_type} {build.ram.speed_mhz}MHz | ${build.ram.price_usd:,} |",
        f"| **Storage (SSD)** | {build.storage.name} | {build.storage.capacity_gb}GB NVMe ({build.storage.read_speed_mb_s} MB/s) | ${build.storage.price_usd:,} |",
        f"| **Power Supply (PSU)** | {build.psu.name} | {build.psu.wattage_w}W {build.psu.certification} | ${build.psu.price_usd:,} |",
        f"| **Case** | {build.case.name} | Form factors: {', '.join(build.case.supported_form_factors)} | ${build.case.price_usd:,} |",
    ]

    if build.cooler:
        lines.append(f"| **CPU Cooler** | {build.cooler.name} | Up to {build.cooler.max_tdp_w}W TDP | ${build.cooler.price_usd:,} |")

    lines.extend([
        "",
        f"💰 **Total Price:** `${build.total_price:,}` (Target Budget: `${budget:,}`)",
        f"⚡ **Estimated Peak Power Draw:** `~{build.estimated_peak_power_w}W` (Headroom: `+{build.psu.wattage_w - build.estimated_peak_power_w}W`)",
    ])

    if fps_estimates:
        lines.extend([
            "",
            f"### 🎮 Estimated Gaming Benchmarks ({resolution.upper()} High/Ultra):",
        ])
        for game, fps in fps_estimates.items():
            lines.append(f"- **{game}:** ~`{fps} FPS`")

    lines.extend([
        "",
        "### 🔍 Engineering Rationale:",
        f"1. **Workload Alignment:** Tailored for `{profile_label}`. "
        + ("Utilizing cost-effective CPU integrated graphics ($0) to minimize cost and power draw for web surfing and office work." if not build.gpu else f"Dedicated graphics ({build.gpu.name}) chosen to meet visual fidelity targets."),
        f"2. **Platform & Socket:** Socket `{build.cpu.socket}` with `{build.ram.ram_type}` delivers optimal efficiency.",
        f"3. **Power Safety:** `{build.psu.wattage_w}W` power supply provides robust headroom over the peak draw of `{build.estimated_peak_power_w}W`.",
    ])

    if val_result and val_result.warnings:
        lines.append("")
        lines.append("### 💡 Notes & Warnings:")
        for w in val_result.warnings:
            lines.append(f"- *{w.rule_name}*: {w.message}")

    return {"final_response": "\n".join(lines)}
