from typing import Tuple

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages, State


class SingleTurnEnv(MultiTurnEnv):
    """
    Environment for single-turn tasks (chat or completion).
    """

    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        if len(state["responses"]) > 0:
            return True
        return False

    def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Tuple[Messages, State]:
        # never called in MultiTurnEnv.rollout
        return [{"role": "user", "content": ""}], state
