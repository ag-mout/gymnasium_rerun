"""An alternative to gymnasium.wrappers.RenderCollection that records each step to a Rerun recording.

* ``RenderRerun`` - Collects rendered frames into Rerun
"""


from copy import deepcopy
from typing import Any, Generic, SupportsFloat, Literal

import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame

import rerun as rr


__all__ = [
    "RenderRerun",
]


class RenderRerun(
    gym.Wrapper[ObsType, ActType, ObsType, ActType],
    Generic[ObsType, ActType, RenderFrame],
    gym.utils.RecordConstructorArgs,
):
    """Collect rendered frames of an environment such each ``step`` is saved to Rerun.

    Example - Add the RenderRerun wrapper and spawn a viewer to see the saved data.
        >>> import gymnasium as gym
        >>> from wrappers import RenderRerun 
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> env = RenderRerun(env, viewer="script")
        >>> _ = env.reset(seed=123)
        >>> for _ in range(5):
        ...     _ = env.step(env.action_space.sample())
        ...
        >>> input()  # Halts the execution, otherwise Rerun Viewer closes when the script finishes


    Example - Add the RenderRerun wrapper and save data to file.
        >>> import gymnasium as gym
        >>> from wrappers import RenderRerun 
        >>> env = gym.make("LunarLander-v3", render_mode="rgb_array")
        >>> env = RenderRerun(env, filename="example.rrd")
        >>> _ = env.reset(seed=123)
        >>> for _ in range(5):
        ...     _ = env.step(env.action_space.sample())
        ...
    Open the .rrd file by running `rerun example.rrd` in a terminal.

    """


    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        filename: str | None = None,
        skip_episodes: int = 100,
        viewer: Literal["script", "notebook", False] = False
    ):
        """Initialize a :class:`RenderRerun` instance.

        Args:
            env: The environment that is being wrapped
            filename (str): Optional to save the recording to a file.
            skip_episodes (int): 0 or 1 save all episodes, otherwise skip episodes to reduce file size. Default 100 means episodes 1, 101, 201, ... are saved.
            viewer (str or False): Default False. Other options "script" or "notebook" should be chosen based on respective code execution method.
        """
        gym.Wrapper.__init__(self, env)

        assert env.render_mode is not None
        assert not env.render_mode.endswith("_list")

        self.episode = 0
        self.frame = 0
        self.skip_episodes = skip_episodes
        self.viewer = viewer

        # rr.init(application_id="rerun_wrapper")
        self.rec = rr.RecordingStream(application_id="rerun_wrapper")


        self.sinks = []

        if filename:
            self.sinks.append(rr.FileSink(path=filename))

        if self.viewer:
            self.sinks.append(rr.GrpcSink())
        
        if self.sinks:
            self.render()
            self.rec.set_sinks(*self.sinks)
        

    @property
    def render_mode(self):
        """Returns the collection render_mode name."""
        return f"{self.env.render_mode}_rerun"


    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Perform a step in the base environment and collect a frame."""
        output = super().step(action)

        if (self.skip_episodes in [0, 1]) or (self.episode % self.skip_episodes == 1):
            self.logger(output, action)
        
        self.frame += 1
        return output


    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the base environment, eventually clear the frame_list, and collect a frame."""
        output = super().reset(seed=seed, options=options)

        self.episode += 1
        self.frame = 0

        return output


    def render(self) -> None:
        """Displays the Rerun viewer in a Jupyter Notebook."""
        if self.viewer == "script":
            self.rec.spawn()
            
        if self.viewer == "notebook":
            self.rec.notebook_show()

        return None


    def logger(self, output: tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]], action: ActType) -> None:
        """Logs the data to Rerun."""
        self.rec.set_time("frame", sequence=self.frame)

        # output = (obsv, reward, done, truncated, info)
        self.rec.log(f"episode{self.episode:05}/reward", rr.TextLog(str(output[1])))
        if output[2]:
            self.rec.log(f"episode{self.episode:05}/done", rr.TextLog("DONE!"))
        if output[3]:
            self.rec.log(f"episode{self.episode:05}/interrupted", rr.TextLog("Interrupted"))

        self.rec.log(f"episode{self.episode:05}/action", rr.TextLog(str(action)))
        self.rec.log(f"episode{self.episode:05}/frames", rr.Image(super().render()).compress(jpeg_quality=95))

        self.rec.flush()


    def close(self):
        """Disconnects Rerun and closes the wrapped environment."""
        self.rec.disconnect()
        return super().close()

