Record Reinforcement Learning gymnasium environments using Rerun.

![alt text](image.png)

For this example a wrapper was created to be called during training with stable-baselines3. Training for a large number of steps can result in larger than memory `.rdd` files, so the following options are included in the wrapper:
- `filename` - Saves the logged data to the specified file. **Not compatible with invoking the viewer!**
- `skip_episodes` - Jumps the indicated number of episodes to keep the `.rdd` file smaller.
- `viewer` - It's possible to choose between `script` or `notebook` depending on how the training code is executed.

## Run the code
To run this example you can install it with `uv`:
```shell
pip install uv
git clone https://github.com/ag-mout/rerun_demo
cd rerun_demo
uv sync
```
In the folder you can find two examples:
- `main.py` - Runs 1000 random steps, which are logged to the native viewer (called by `rr.spawn()`).
- `training_example.ipynb` - Trains a line follower model based on <https://github.com/ag-mout/gym-line-follower>, and saves a test run to a `.rrd` file to be opened in the native viewer after completion.

The notebook can also be executed on Google Colab at: https://colab.research.google.com/drive/1XEizcsiQgTHrAEZYWNfV-Hnv_kbnqNr9?usp=sharing

## Frequently Asked Questions

### Why can't I save to file and watch it in Rerun at the same time?
It's possible to use `set_sinks` and send the data to both a `FileSink` and a `GrpcSink` in the same execution. However I was finding some problems that could result in data loss for users. In the future the goal is to drop that restriction and enable both sinks.

### I can skip episodes, but what if I just want to keep the most recent episodes?
I've tried using `Clear` to delete old data, and I've tried with both `static` logs and `disable_timeline` options. It always resulted in a bloated file, so I chose to skip episodes to keep the file smaller. Future plans include storing only the most recent episodes, or splitting the recording in many files if disk space is not an issue for the user. This way the viewing limitation will be the available RAM, because Rerun loads the entire file to RAM (with plans for the SDK to enable larger than memory files to be viewed).

### Where can I find the original environment for the RL?
The original environment is from nplan and it's here: https://github.com/nplan/gym-line-follower  
However it's not working with the most recent version of `gymnasium`. The fork from khleedavid fixes some of the issues here: https://github.com/khleedavid/gym-line-follower  
And then I've updated the code due to compatibility issues with newer versions of `matplotlib`, and added `uv` to make it easier for anyone to install the same python environment (and easier for me to run it both locally and in Google Colab). It also has some tweaks on the action space and rewards to incentivize it to go faster: https://github.com/ag-mout/gym-line-follower

### Why does the example take so long to run?
The example will require about 1-2 hours on Google Colab until fully trained to achieve a speedy track completion.  
Why this example was chosen is simple: the first robot I coded, 15 years ago, was a line follower for a school project. It got to the end of the track with a very simple algorithm, but it was very slow. So I chose to do RL on a line follower to close the loop on that old project and achieve a much faster track completion.
