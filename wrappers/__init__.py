"""An alternative to gymnasium.wrappers.RenderCollection that records each step to a Rerun recording.

* ``RenderRerun`` - Collects rendered frames into Rerun
"""


from copy import deepcopy
from typing import Any, Generic, SupportsFloat, Literal

import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame

import rerun as rr, rerun.blueprint as rrb


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

        # Store any active RecordingStream instances in a list and iterate over it.
        self.recs: list[rr.RecordingStream] = []

        if filename:
            file_rec = rr.RecordingStream(application_id="rerun_wrapper_file")
            file_rec.save(filename)
            self.recs.append(file_rec)

        # Create a viewer recording stream only; do not auto-spawn — spawn when render() is called.
        self.viewer_rec = rr.RecordingStream(application_id="rerun_wrapper_viewer")
        if self.viewer:
            self.render()

        self.recs.append(self.viewer_rec)
        
        self.start_blueprint()
        

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
        """Reset the base environment, move to next episode and reset frames."""
        output = super().reset(seed=seed, options=options)

        self.episode += 1
        self.frame = 0

        return output


    def render(self) -> None:
        """Displays the Rerun viewer in a Jupyter Notebook."""
        # kept for compatibility, delegate to viewer recording stream if present
        if self.viewer_rec:
            if self.viewer == "script":
                self.viewer_rec.spawn()
            elif self.viewer == "notebook":
                self.viewer_rec.notebook_show()

        return None


    def logger(self, output: tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]], action: ActType) -> None:
        """Logs the data to Rerun."""
        episode_name = f"episode{self.episode:05}"

        # output = (obsv, reward, done, truncated, info)
        reward = rr.TextLog(str(output[1]))
        done = output[2]
        rr_done = rr.TextLog("DONE!")
        truncated = output[3]
        rr_truncated = rr.TextLog("Interrupted")
        action = rr.TextLog(str(action))
        image = rr.Image(super().render()).compress(jpeg_quality=95)
        for s in self.recs:
            s.set_time("frame", sequence=self.frame)

            s.log(f"{episode_name}/reward", reward)

            if done:
                s.log(f"{episode_name}/done", rr_done)

            if truncated:
                s.log(f"{episode_name}/interrupted", rr_truncated)

            s.log(f"{episode_name}/action", action)
            s.log(f"{episode_name}/frames", image)

            self.update_blueprint(episode_name)


    def start_blueprint(self):
        self.episode_names = set()
        self.tabs = []


    def update_blueprint(self, episode_name):
        if episode_name not in self.episode_names:
            self.episode_names.add(episode_name)
            self.tabs.append(
                rrb.Horizontal(
                            contents= [
                                rrb.Spatial2DView(
                                    name="frames",
                                    origin=f"/{episode_name}/frames"
                                )
                            ] + 
                            [
                                rrb.Vertical(contents=[
                                    rrb.TextLogView(
                                        name=name,
                                        origin=f"/{episode_name}/{name}")
                                        for name in ["action", "reward"]

                                ])
                            ],
                            name=episode_name,
                        )
            )
            
            blueprint = rrb.Blueprint(
                rrb.Tabs(
                    contents=self.tabs
                ),
                rrb.BlueprintPanel(state="collapsed"),
                rrb.SelectionPanel(state="collapsed"),
                rrb.TimePanel(state="expanded"),
            )

            for s in self.recs:
                s.send_blueprint(blueprint)



    def close(self):
        """Disconnects Rerun and closes the wrapped environment."""
        # Disconnect all recording streams
        for s in self.recs:
            try:
                s.disconnect()
            except Exception:
                pass

        return super().close()

