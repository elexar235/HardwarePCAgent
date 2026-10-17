"""Deterministic PC compatibility verification engine.

This module contains zero LLM hallucination: it uses strict mathematical
and specification checks to validate hardware builds.
"""

from typing import List
from src.models.build import CompatibilityIssue, PCBuild, ValidationResult


def check_display_output(build: PCBuild) -> List[CompatibilityIssue]:
    """Ensure system has either a discrete GPU or an integrated CPU graphics unit."""
    issues = []
    if build.gpu is None and not build.cpu.has_integrated_gpu:
        issues.append(
            CompatibilityIssue(
                severity="error",
                rule_name="No Video Output",
                component_a=f"{build.cpu.name} (No iGPU)",
                component_b="Discrete GPU (Missing)",
                message=(
                    f"No display output available! CPU '{build.cpu.name}' does not have integrated graphics, "
                    "and no dedicated graphics card (GPU) was included in the build."
                ),
                suggested_fix="Add a dedicated GPU or switch to a processor with built-in graphics.",
            )
        )
    return issues


def check_socket_compatibility(build: PCBuild) -> List[CompatibilityIssue]:
    issues = []
    if build.cpu.socket != build.motherboard.socket:
        issues.append(
            CompatibilityIssue(
                severity="error",
                rule_name="Socket Mismatch",
                component_a=f"{build.cpu.name} ({build.cpu.socket})",
                component_b=f"{build.motherboard.name} ({build.motherboard.socket})",
                message=(
                    f"Incompatible sockets! CPU '{build.cpu.name}' uses socket {build.cpu.socket}, "
                    f"but Motherboard '{build.motherboard.name}' has socket {build.motherboard.socket}."
                ),
                suggested_fix=(
                    f"Change motherboard to a {build.cpu.socket} model or switch CPU to {build.motherboard.socket}."
                ),
            )
        )
    return issues


def check_ram_compatibility(build: PCBuild) -> List[CompatibilityIssue]:
    issues = []
    # 1. Motherboard RAM type
    if build.ram.ram_type != build.motherboard.ram_type:
        issues.append(
            CompatibilityIssue(
                severity="error",
                rule_name="RAM Type Mismatch (Motherboard)",
                component_a=f"{build.ram.name} ({build.ram.ram_type})",
                component_b=f"{build.motherboard.name} ({build.motherboard.ram_type})",
                message=(
                    f"RAM type conflict! Motherboard requires {build.motherboard.ram_type}, "
                    f"but selected RAM is {build.ram.ram_type}."
                ),
                suggested_fix=f"Select {build.motherboard.ram_type} memory modules.",
            )
        )

    # 2. CPU RAM support
    if build.ram.ram_type not in build.cpu.supports_ram:
        issues.append(
            CompatibilityIssue(
                severity="error",
                rule_name="RAM Type Mismatch (CPU)",
                component_a=f"{build.cpu.name}",
                component_b=f"{build.ram.name} ({build.ram.ram_type})",
                message=(
                    f"CPU '{build.cpu.name}' memory controller does not support {build.ram.ram_type}. "
                    f"Supported types: {', '.join(build.cpu.supports_ram)}."
                ),
                suggested_fix=f"Choose RAM matching CPU supported standards ({build.cpu.supports_ram}).",
            )
        )
    return issues


def check_power_supply(build: PCBuild) -> List[CompatibilityIssue]:
    issues = []
    peak_draw = build.estimated_peak_power_w

    if build.gpu:
        recommended_min_w = max(build.gpu.recommended_psu_w, int(peak_draw * 1.25))
    else:
        recommended_min_w = int(peak_draw * 1.3)

    if build.psu.wattage_w < peak_draw:
        issues.append(
            CompatibilityIssue(
                severity="error",
                rule_name="Insufficient Power Supply",
                component_a=f"{build.psu.name} ({build.psu.wattage_w}W)",
                component_b="System Load",
                message=(
                    f"Critical power deficit! Estimated peak consumption is {peak_draw}W, "
                    f"which exceeds PSU capacity of {build.psu.wattage_w}W."
                ),
                suggested_fix=f"Upgrade PSU to at least {recommended_min_w}W.",
            )
        )
    elif build.psu.wattage_w < recommended_min_w:
        target_name = build.gpu.name if build.gpu else "System Configuration"
        issues.append(
            CompatibilityIssue(
                severity="warning",
                rule_name="Low Power Headroom",
                component_a=f"{build.psu.name} ({build.psu.wattage_w}W)",
                component_b=target_name,
                message=(
                    f"PSU wattage ({build.psu.wattage_w}W) is lower than recommended headroom "
                    f"({recommended_min_w}W). May reduce PSU lifespan under heavy load."
                ),
                suggested_fix=f"Consider a {recommended_min_w}W or higher rated PSU.",
            )
        )
    return issues


def check_physical_clearance(build: PCBuild) -> List[CompatibilityIssue]:
    issues = []
    # 1. GPU length vs Case clearance (if discrete GPU is installed)
    if build.gpu and build.gpu.length_mm > build.case.max_gpu_length_mm:
        issues.append(
            CompatibilityIssue(
                severity="error",
                rule_name="GPU Length Exceeded",
                component_a=f"{build.gpu.name} ({build.gpu.length_mm}mm)",
                component_b=f"{build.case.name} (Max {build.case.max_gpu_length_mm}mm)",
                message=(
                    f"GPU length ({build.gpu.length_mm}mm) exceeds case maximum clearance "
                    f"({build.case.max_gpu_length_mm}mm). The GPU will not physically fit into the chassis."
                ),
                suggested_fix="Choose a roomier case or a more compact (dual-fan) GPU variant.",
            )
        )

    # 2. Motherboard form factor vs Case support
    if build.motherboard.form_factor not in build.case.supported_form_factors:
        issues.append(
            CompatibilityIssue(
                severity="error",
                rule_name="Form Factor Incompatibility",
                component_a=f"{build.motherboard.name} ({build.motherboard.form_factor})",
                component_b=f"{build.case.name} ({build.case.supported_form_factors})",
                message=(
                    f"Motherboard form factor '{build.motherboard.form_factor}' cannot be mounted in "
                    f"case '{build.case.name}' (supported: {', '.join(build.case.supported_form_factors)})."
                ),
                suggested_fix="Pick a compatible case or adjust motherboard form factor.",
            )
        )

    # 3. CPU Cooler height vs Case clearance (if cooler is provided)
    if build.cooler and build.cooler.height_mm > build.case.max_cooler_height_mm:
        issues.append(
            CompatibilityIssue(
                severity="error",
                rule_name="Cooler Height Exceeded",
                component_a=f"{build.cooler.name} ({build.cooler.height_mm}mm)",
                component_b=f"{build.case.name} (Max {build.case.max_cooler_height_mm}mm)",
                message=(
                    f"Cooler height ({build.cooler.height_mm}mm) exceeds case maximum width clearance "
                    f"({build.case.max_cooler_height_mm}mm)."
                ),
                suggested_fix="Choose a lower profile tower cooler or an AIO liquid cooler.",
            )
        )
    return issues


def check_cooler_compatibility(build: PCBuild) -> List[CompatibilityIssue]:
    issues = []
    if not build.cooler:
        return issues

    if build.cpu.socket not in build.cooler.supported_sockets:
        issues.append(
            CompatibilityIssue(
                severity="error",
                rule_name="Cooler Socket Mismatch",
                component_a=f"{build.cooler.name}",
                component_b=f"{build.cpu.name} ({build.cpu.socket})",
                message=(
                    f"Cooler '{build.cooler.name}' does not include mounting brackets for socket {build.cpu.socket}."
                ),
                suggested_fix=f"Select a cooler with native {build.cpu.socket} mounting hardware.",
            )
        )

    if build.cooler.max_tdp_w < build.cpu.tdp_w:
        issues.append(
            CompatibilityIssue(
                severity="warning",
                rule_name="Thermal Dissipation Warning",
                component_a=f"{build.cooler.name} ({build.cooler.max_tdp_w}W TDP)",
                component_b=f"{build.cpu.name} ({build.cpu.tdp_w}W TDP)",
                message=(
                    f"Cooler capacity ({build.cooler.max_tdp_w}W) is lower than CPU thermal output "
                    f"({build.cpu.tdp_w}W). Risk of thermal throttling under sustained rendering/gaming."
                ),
                suggested_fix="Upgrade cooler to one rated for at least 150-200W TDP.",
            )
        )
    return issues


def check_balance_bottleneck(build: PCBuild) -> List[CompatibilityIssue]:
    issues = []
    if not build.gpu:
        return issues

    cpu_score = build.cpu.benchmark_score
    gpu_score = build.gpu.benchmark_score

    ratio = gpu_score / max(cpu_score, 1)

    if ratio > 2.8:
        issues.append(
            CompatibilityIssue(
                severity="warning",
                rule_name="Severe GPU Bottleneck by CPU",
                component_a=f"{build.cpu.name} (Score: {cpu_score})",
                component_b=f"{build.gpu.name} (Score: {gpu_score})",
                message=(
                    f"The CPU '{build.cpu.name}' is significantly underpowered for GPU '{build.gpu.name}'. "
                    "In CPU-bound titles, the CPU will bottleneck the graphics card."
                ),
                suggested_fix="Upgrade to a faster CPU to unlock the full potential of your GPU.",
            )
        )
    elif ratio < 0.5:
        issues.append(
            CompatibilityIssue(
                severity="warning",
                rule_name="CPU Heavily Outpaces GPU Tier",
                component_a=f"{build.cpu.name} (Score: {cpu_score})",
                component_b=f"{build.gpu.name} (Score: {gpu_score})",
                message=(
                    f"The CPU '{build.cpu.name}' significantly outpaces the GPU '{build.gpu.name}'. "
                    "While well-suited for code compilation, data processing, or audio/video tasks, "
                    "pure gaming workloads would benefit from rebalancing budget toward a higher GPU tier."
                ),
                suggested_fix="If primarily gaming, consider stepping down the CPU to invest in a faster GPU.",
            )
        )
    return issues


def validate_pc_build(build: PCBuild) -> ValidationResult:
    """Run full deterministic validation suite against a PC build."""
    all_issues: List[CompatibilityIssue] = []

    all_issues.extend(check_display_output(build))
    all_issues.extend(check_socket_compatibility(build))
    all_issues.extend(check_ram_compatibility(build))
    all_issues.extend(check_power_supply(build))
    all_issues.extend(check_physical_clearance(build))
    all_issues.extend(check_cooler_compatibility(build))
    all_issues.extend(check_balance_bottleneck(build))

    errors = [issue for issue in all_issues if issue.severity == "error"]
    warnings = [issue for issue in all_issues if issue.severity == "warning"]

    return ValidationResult(
        is_compatible=(len(errors) == 0),
        issues=errors,
        warnings=warnings,
    )
