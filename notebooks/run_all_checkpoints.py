import sys
import os
import numpy as np
import importlib

# hacky way, figure out proper way later
import pickle as pkl

sys.path.append(os.path.abspath("../"))
import re
from utils.hooklessrun import run_envs, setup_analyze
from utils.models import BayesianActor, BayesianBlocklessActor


def compute_optimal_bayesian_actor(envs):
    bayes_actor = BayesianActor()
    bayes_actor.reset(
        num_sessions=len(envs),
        block_side_probs=envs[0].block_side_probs,
        possible_trial_strengths=envs[0].possible_trial_strengths,
        possible_trial_strengths_probs=envs[0].possible_trial_strengths_probs,
        trials_per_block_param=envs[0].trials_per_block_param,
    )
    print("Running bayesian actor")
    run_envs_output = run_envs(model=bayes_actor, envs=envs)
    optimal_bayesian_actor_results = dict(
        bayesian_actor_session_data=run_envs_output["session_data"]
    )
    return optimal_bayesian_actor_results


if __name__ == "__main__":

    # hacky way, figure out proper way later

    # train_run_id = "fixed_time_stim_no_reward"  # new training run id
    # run_dir = "/usr/people/kundu/code/ann-rnn-modified/runs"

    train_run_id = "fixed_at_3kitx"  # new training run id
    run_dir = "/Users/dkundu/Documents/phd/ann-rnn-modified/runs/"
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

    checkpoints = sorted_files

    for idx in range(0, len(checkpoints)):

        fname = checkpoints[idx]
        checkpoint_number = (
            fname.rsplit("/")[-1].rsplit("checkpoint_grad_steps=")[-1].rstrip(".pt")
        )
        setup_results = setup_analyze(
            train_run_id=train_run_dir, checkpoint_number=checkpoint_number
        )

        run_envs_output = run_envs(
            model=setup_results["model"], envs=setup_results["envs"]
        )

        try:
            with open(
                f"/Users/dkundu/Documents/phd/ann-rnn-modified/data/fixed_time_stim_fixed_weights/rnn_ann_model_results_10units_{checkpoint_number}.pkl",
                "wb",
            ) as f:
                pkl.dump(run_envs_output, f)
        except Exception as e:
            print(e)

        # bayesian_session_data = compute_optimal_bayesian_actor(setup_results["envs"])

        # filename = f"        /Users/dkundu/Documents/phd/ann-rnn-modified/data/fixed_time_stim_fixed_weights/bayesian_results_10units_{checkpoint_number}.pkl"
        # try:
        #     with open(filename, "wb") as f:
        #         pkl.dump(bayesian_session_data, f)
        # except Exception as e:
        #     print(e)
