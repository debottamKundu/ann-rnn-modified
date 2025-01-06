import sys
import os
import numpy as np
import importlib

# hacky way, figure out proper way later
import pickle as pkl
sys.path.append(os.path.abspath("../"))
import re
import utils.run


if __name__ == "__main__":

    # hacky way, figure out proper way later


    # train_run_id = 'rnn_max_stim_strength=2.5_hidden_size=50_2024-11-08 17-46-31.791839' # old training run id

    train_run_id = "noisydatachoicee"  # new training run id
    run_dir = "/usr/people/kundu/code/ann-rnn-modified/runs"
    train_run_dir = os.path.join(run_dir, train_run_id)
    analyze_run_dir = os.path.join(train_run_dir, "analyze")

    checkpoint_paths = [
        os.path.join(train_run_dir, file_path)
        for file_path in os.listdir(train_run_dir)
        if file_path.endswith(".pt")
    ]

    # sort checkpoints properly
    # silly but numeric sort for checkpoint files
    numbered_files = [f for f in checkpoint_paths if re.search(r"\d+.pt$", f)]
    sorted_files = sorted(
        numbered_files, key=lambda x: int(re.search(r"(\d+).pt$", x).group(1))
    )

    checkpoints = np.sort(checkpoint_paths)

    for idx in range(0,len(checkpoints)):
        setup_results = utils.run.setup_analyze(
            train_run_id=train_run_id, sort_index=idx
        )

        run_envs_output = utils.run.run_envs(
            model=setup_results["model"], envs=setup_results["envs"]
        )
        fname_idx = int(
            checkpoints[idx]
            .rsplit("/")[-1]
            .rsplit("checkpoint_grad_steps=")[-1]
            .rstrip(".pt")
        )
        try:
            with open(
                f"/usr/people/kundu/code/ann-rnn-modified/data/noisydatachoicee/rnn_ann_model_results_10units_{fname_idx}.pkl",
                "wb",
            ) as f:
                pkl.dump(run_envs_output, f)
        except Exception as e:
            print(e)
