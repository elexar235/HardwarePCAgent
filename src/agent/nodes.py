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
    budget_kw_match = re.search(r"budget\s*(?:is|of|around|:)?\s*\$?(\d+)", q)

    if dollar_match:
        budget = int(dollar_match.group(1))
    elif budget_kw_match:
        budget = int(budget_kw_match.group(1))
    else:
        k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", q)
        if k_match:
            val = float(k_match.group(1))
            budget = int(val * 1000) if val < 20 else int(val)
        else:
            # Match 3-4 digit numbers, excluding common game years (e.g. 2077)
            num_matches = re.findall(r"\b([3-9]\d{2}|[1-9]\d{3})\b", q)
            valid_nums = [int(n) for n in num_matches if n not in ["2077", "1440", "1080", "2160", "3840"]]
            if valid_nums:
                budget = valid_nums[0]

    # Resolution (None unless explicitly specified by the user)
    res = None
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
    is_office = any(k in tasks for k in ["office", "surfing", "browser", "web surfing"])
    is_gaming = len(games) > 0 or (req.resolution is not None)

    if has_heavy_gpu_tasks:
        return "ai_3d"
    if has_productivity and not is_gaming:
        return "productivity"
    if is_office:
        return "office"
    if is_gaming:
        return "gaming"
    if (req.budget_usd or 0) <= 600 and not has_productivity and not has_heavy_gpu_tasks:
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


def _select_platform_combo(
    all_cpus: List[CPU],
    target_platform_budget: int,
    target_ram_gb: int,
    require_igpu: bool = False,
) -> tuple[CPU, Motherboard, RAM]:
    """Select compatible (CPU, Motherboard, RAM) combo maximizing performance within budget."""
    candidates = []

    for cpu in all_cpus:
        if require_igpu and not cpu.has_integrated_gpu:
            continue

        mobos = db.list_motherboards(socket=cpu.socket)
        if not mobos:
            continue

        # Evaluate each supported RAM type for this CPU
        for ram_type in cpu.supports_ram:
            compat_mobos = [m for m in mobos if m.ram_type == ram_type]
            if not compat_mobos:
                continue
            best_mobo = compat_mobos[0]

            matching_rams = db.list_rams(ram_type=ram_type, min_capacity_gb=target_ram_gb)
            if not matching_rams:
                matching_rams = db.list_rams(ram_type=ram_type)
            if not matching_rams:
                continue
            best_ram = matching_rams[0]

            combo_cost = cpu.price_usd + best_mobo.price_usd + best_ram.price_usd
            combo_score = cpu.benchmark_score + (best_ram.capacity_gb * 50) + (best_ram.speed_mhz * 0.05)

            candidates.append({
                "cpu": cpu,
                "motherboard": best_mobo,
                "ram": best_ram,
                "cost": combo_cost,
                "score": combo_score,
            })

    if not candidates:
        cheapest_cpu = min(all_cpus, key=lambda c: c.price_usd)
        mobo = db.list_motherboards(socket=cheapest_cpu.socket)[0]
        ram = db.list_rams(ram_type=mobo.ram_type)[0]
        return cheapest_cpu, mobo, ram

    within_budget = [c for c in candidates if c["cost"] <= target_platform_budget]
    if within_budget:
        best = max(within_budget, key=lambda c: c["score"])
    else:
        best = min(candidates, key=lambda c: c["cost"])

    return best["cpu"], best["motherboard"], best["ram"]


def _select_gpu(
    all_gpus: List[GPU],
    target_gpu_budget: int,
    profile: str,
    target_resolution: Optional[str] = None,
) -> Optional[GPU]:
    """Select optimal GPU dynamically using a multi-criteria scoring function."""
    if profile == "office":
        return None

    cheapest_gpu = min(all_gpus, key=lambda g: g.price_usd)
    available_gpus = [g for g in all_gpus if g.price_usd <= target_gpu_budget]
    if not available_gpus:
        available_gpus = [cheapest_gpu]

    def _gpu_score(gpu: GPU) -> float:
        if profile == "ai_3d":
            # AI workloads value VRAM capacity and CUDA / Tensor Core acceleration
            is_nvidia = (
                gpu.brand.lower() == "nvidia"
                or "rtx" in gpu.name.lower()
                or "geforce" in gpu.name.lower()
            )
            cuda_weight = 6000.0 if is_nvidia else 0.0
            return (gpu.vram_gb * 1000.0) + cuda_weight + (gpu.benchmark_score * 0.1)
        elif profile == "productivity":
            return float(gpu.benchmark_score) + (gpu.vram_gb * 200.0)
        else:  # gaming
            vram_bonus = 3000.0 if (target_resolution == "4k" and gpu.vram_gb >= 16) else 0.0
            return float(gpu.benchmark_score) + vram_bonus

    return max(available_gpus, key=_gpu_score)


def plan_and_select_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Select balanced components dynamically based on workload profile & budget."""
    requirements = state["requirements"] or _heuristic_parse_requirements(state["user_query"])
    budget = requirements.budget_usd or 1200
    iteration = state.get("iteration_count", 0)
    feedback = state.get("feedback_history", [])

    profile = _determine_workload_profile(requirements)

    over_budget_penalty = 0
    overrun_amount = 0
    for f in feedback:
        m = re.search(r"exceeds budget by \$([0-9,]+)", f)
        if m:
            overrun_amount = max(overrun_amount, int(m.group(1).replace(",", "")))

    if overrun_amount > 0:
        over_budget_penalty = overrun_amount + 30 * iteration
    elif any("exceeds" in f.lower() for f in feedback):
        over_budget_penalty = 60 * max(1, iteration)

    adjusted_budget = max(350, budget - over_budget_penalty)
    all_cpus = db.list_cpus()
    all_gpus = db.list_gpus()

    # Determine dynamic budget fractions by profile
    if profile == "office":
        target_platform_budget = int(adjusted_budget * 0.68)
        target_ram_gb = 16
        gpu = None
        cpu, motherboard, ram = _select_platform_combo(
            all_cpus=all_cpus,
            target_platform_budget=target_platform_budget,
            target_ram_gb=target_ram_gb,
            require_igpu=True,
        )
    elif profile == "ai_3d":
        target_gpu_budget = int(adjusted_budget * 0.52)
        target_platform_budget = int(adjusted_budget * 0.36)
        target_ram_gb = 32
        gpu = _select_gpu(all_gpus, target_gpu_budget, profile, requirements.resolution)
        cpu, motherboard, ram = _select_platform_combo(
            all_cpus=all_cpus,
            target_platform_budget=target_platform_budget,
            target_ram_gb=target_ram_gb,
            require_igpu=False,
        )
    elif profile == "productivity":
        target_gpu_budget = int(adjusted_budget * 0.22)
        target_platform_budget = int(adjusted_budget * 0.64)
        target_ram_gb = 32
        gpu = _select_gpu(all_gpus, target_gpu_budget, profile, requirements.resolution)
        cpu, motherboard, ram = _select_platform_combo(
            all_cpus=all_cpus,
            target_platform_budget=target_platform_budget,
            target_ram_gb=target_ram_gb,
            require_igpu=False,
        )
    else:  # gaming
        target_gpu_budget = int(adjusted_budget * 0.48)
        target_platform_budget = int(adjusted_budget * 0.38)
        target_ram_gb = 16
        gpu = _select_gpu(all_gpus, target_gpu_budget, profile, requirements.resolution)
        cpu, motherboard, ram = _select_platform_combo(
            all_cpus=all_cpus,
            target_platform_budget=target_platform_budget,
            target_ram_gb=target_ram_gb,
            require_igpu=False,
        )

    # 4. Select Storage dynamically by budget allocation
    storages = db.list_storages()
    target_storage_budget = max(40, int(adjusted_budget * 0.08))
    affordable_storages = [s for s in storages if s.price_usd <= target_storage_budget]
    if not affordable_storages:
        storage = storages[0]
    else:
        storage = max(affordable_storages, key=lambda s: (s.capacity_gb, s.read_speed_mb_s))

    # 5. Select PSU based on system load
    gpu_power = gpu.tdp_w if gpu else 0
    gpu_rec_psu = gpu.recommended_psu_w if gpu else (300 if profile == "office" else 400)
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

    budget = requirements.budget_usd if requirements else 0
    profile = _determine_workload_profile(requirements) if requirements else "gaming"

    # Target resolution: only meaningful for gaming workloads
    resolution = None
    if profile == "gaming":
        resolution = requirements.resolution if (requirements and requirements.resolution) else None
        if not resolution:
            # Default gaming resolution is 1080p (Full HD) unless explicitly requested
            resolution = "1080p"

    fps_estimates = _estimate_game_fps(build.gpu, resolution) if (profile == "gaming" and resolution) else {}

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

    # Meta header: only show target resolution for gaming
    if profile == "gaming" and resolution:
        meta_header = f"**Target Resolution:** `{resolution.upper()}` | **Profile:** `{profile_label}`"
    else:
        meta_header = f"**Profile:** `{profile_label}`"

    lines = [
        f"## 🖥️ Recommended PC Configuration ({profile_label})",
        "",
        meta_header,
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

    if budget and build.total_price > budget:
        overrun = build.total_price - budget
        if profile == "gaming" and resolution:
            lines.extend([
                "",
                f"> ⚠️ **Budget Constraint Notice:** The minimum viable configuration for `{resolution.upper()}` gaming is **${build.total_price:,}**, which exceeds your budget of **${budget:,}** by **${overrun:,}**.",
                f"> *Feasibility Note:* Gaming at **{resolution.upper()}** requires modern graphics hardware starting at $350+ for the GPU alone. To meet a ${budget} ceiling, consider dropping target resolution to 1080p or increasing budget.",
            ])
        else:
            profile_hints = {
                "ai_3d": "AI/ML workloads require high-VRAM GPUs which are costly. Consider a lower VRAM tier (e.g. 12GB) or increase the budget.",
                "productivity": "Workstation builds prioritize multi-core CPU and fast RAM. Consider a lower GPU tier or increasing the budget.",
                "office": "Consider reducing RAM capacity or choosing a lower-tier storage option.",
            }
            hint = profile_hints.get(profile, "Consider adjusting component tiers to fit the budget.")
            lines.extend([
                "",
                f"> ⚠️ **Budget Constraint Notice:** Selected configuration totals **${build.total_price:,}**, exceeding your budget of **${budget:,}** by **${overrun:,}**.",
                f"> *Suggestion:* {hint}",
            ])

    # Show AI/LLM capability breakdown for AI workloads
    if profile == "ai_3d" and build.gpu:
        vram = build.gpu.vram_gb
        is_nv = build.gpu.brand.lower() == "nvidia" or "geforce" in build.gpu.name.lower() or "rtx" in build.gpu.name.lower()
        backend_note = "NVIDIA CUDA & TensorRT (native PyTorch/Ollama/vLLM support)" if is_nv else "DirectML / ROCm"
        llm_size = "14B" if vram >= 16 else ("8B" if vram >= 12 else "7B" if vram >= 8 else "3B")
        lines.extend([
            "",
            "### 🧠 Estimated AI & Local LLM Capability:",
            f"- **Compute Engine:** `{build.gpu.name}` ({backend_note})",
            f"- **VRAM Capacity:** `{vram}GB` — fully offloads models up to ~`{llm_size}` parameters (e.g. Qwen 2.5, Llama 3.1) in 4-bit/8-bit precision",
            f"- **System Memory:** `{build.ram.capacity_gb}GB {build.ram.ram_type}` provides ample memory for large datasets and CPU-offloaded layers",
        ])

    # Show gaming benchmarks only when the build is actually targeting gaming
    if profile == "gaming" and fps_estimates and resolution:
        lines.extend([
            "",
            f"### 🎮 Estimated Gaming Benchmarks ({resolution.upper()} High/Ultra):",
        ])
        for game, fps in fps_estimates.items():
            lines.append(f"- **{game}:** ~`{fps} FPS`")

    # Workload Rationale tailored to profile
    if profile == "office":
        gpu_rationale = "Utilizing cost-effective CPU integrated graphics ($0) to minimize cost, noise, and power draw for web surfing and office applications."
    elif profile == "ai_3d":
        backend = "NVIDIA CUDA / Tensor Cores" if (build.gpu and (build.gpu.brand.lower() == "nvidia" or "rtx" in build.gpu.name.lower())) else "High-VRAM Compute"
        gpu_rationale = f"Dedicated GPU ({build.gpu.name}) selected for {build.gpu.vram_gb}GB VRAM and {backend} for model training and local LLM inference."
    elif profile == "productivity":
        gpu_rationale = f"Dedicated graphics ({build.gpu.name}) selected for multi-display productivity and hardware-accelerated video/rendering workflows."
    else:
        gpu_rationale = f"Dedicated graphics ({build.gpu.name}) chosen to meet framerate and visual fidelity targets."

    lines.extend([
        "",
        "### 🔍 Engineering Rationale:",
        f"1. **Workload Alignment:** Tailored for `{profile_label}`. {gpu_rationale}",
        f"2. **Platform & Socket:** Socket `{build.cpu.socket}` with `{build.ram.ram_type}` delivers optimal efficiency.",
        f"3. **Power Safety:** `{build.psu.wattage_w}W` power supply provides robust headroom over the peak draw of `{build.estimated_peak_power_w}W`.",
    ])

    if val_result and val_result.issues:
        lines.append("")
        lines.append("### ❌ Unresolved Constraints & Issues:")
        for issue in val_result.issues:
            lines.append(f"- **[{issue.rule_name}]** {issue.message}")

    if val_result and val_result.warnings:
        lines.append("")
        lines.append("### 💡 Notes & Recommendations:")
        for w in val_result.warnings:
            lines.append(f"- *{w.rule_name}*: {w.message}")

    return {"final_response": "\n".join(lines)}
