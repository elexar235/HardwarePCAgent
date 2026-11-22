"""User requirement extraction schemas."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class UserRequirements(BaseModel):
    budget_usd: Optional[int] = Field(
        None, description="Maximum total budget in USD"
    )
    resolution: Optional[Literal["1080p", "1440p", "4k"]] = Field(
        None, description="Target gaming display resolution (only for gaming workloads)"
    )
    target_fps: int = Field(60, description="Desired minimum average frame rate")
    target_games: List[str] = Field(
        default_factory=list, description="Games the user primarily plays or wants to play"
    )
    work_tasks: List[str] = Field(
        default_factory=list,
        description="Non-gaming workloads: 3D rendering, video editing, ML/AI, programming",
    )
    aesthetic_preference: Optional[str] = Field(
        None, description="Aesthetic details (RGB, white build, stealth black, compact ITX)"
    )
    priorities: List[str] = Field(
        default_factory=list,
        description="Key priorities: maximum FPS, quiet operation, future upgradeability",
    )
    budget_weights: Optional[dict[str, float]] = Field(
        default=None,
        description="Dynamic budget allocation fractions (e.g. {'gpu': 0.65, 'platform': 0.20, 'cooling': 0.08, 'case': 0.07})",
    )
    silence_focus: bool = Field(False, description="Whether silence/noise reduction is a primary priority")
    prefer_liquid_cooler: bool = Field(False, description="Whether liquid AIO cooling is requested")
    prefer_aesthetic_case: bool = Field(False, description="Whether a premium/showcase case is requested")
