"""The skill forge — turn a failure cluster into a trained, validated skill.

``spec_generator``  : FailureCluster -> SkillSpec (LLM).
``reconstruction``  : the 3DGS substrate interface (contribution C3; stubbed tonight).
``trainer_base``    : the Trainer ABC.
``trainers/``       : prompt / code / classifier trainers (each a concrete Trainer).
"""

from aora_forge.skill_forge.spec_generator import generate_spec
from aora_forge.skill_forge.trainer_base import Trainer, get_trainer

__all__ = ["generate_spec", "Trainer", "get_trainer"]
