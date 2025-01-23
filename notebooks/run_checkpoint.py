import sys
import os
import numpy as np
import importlib

# hacky way, figure out proper way later
import pickle as pkl
import re
import sys
sys.path.append(os.path.abspath("../"))
from utils.hooklessrun import run_envs, setup_analyze
import argparse
from utils.models import BayesianActor, BayesianBlocklessActor

def parse_arguments():
    """Set up and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a particular checkpoint for a model and save the output"
    )

    # Add arguments
    parser.add_argument(
        "-n", "--number", type=str, help="Checkpoint number", required=True
    )
    parser.add_argument(
        "-l", "--location", type=str, help="Model storage location in run directory", required=True
    )
    parser.add_argument(
        "-s", "--save", type=str, help="Save location for session output", required=True
    )

    # Parse arguments
    return parser.parse_args()


def compute_optimal_bayesian_actor(envs):
    bayes_actor = BayesianActor()
    bayes_actor.reset(
        num_sessions=len(envs),
        block_side_probs=envs[0].block_side_probs,
        possible_trial_strengths=envs[0].possible_trial_strengths,
        possible_trial_strengths_probs=envs[0].possible_trial_strengths_probs,
        trials_per_block_param=envs[0].trials_per_block_param,
    )
    print('Running bayesian actor')
    run_envs_output = run_envs(model=bayes_actor, envs=envs)
    optimal_bayesian_actor_results = dict(
        bayesian_actor_session_data=run_envs_output["session_data"]
    )
    return optimal_bayesian_actor_results


def main(args):
    #args = parse_arguments()

    train_run_id = args.location
    checkpoint_number = args.number
    save_location = args.save
    setup_results = setup_analyze(
        train_run_id=train_run_id, checkpoint_number=checkpoint_number
    )  # TODO:change this so that i can just pass the id
    run_envs_output = run_envs(
        model=setup_results["model"], envs=setup_results["envs"]
    )

    filename = f"{save_location}_{checkpoint_number}.pkl"
    try:
        with open(filename, "wb") as f:
            pkl.dump(run_envs_output, f)
    except Exception as e:
        print(e)


    # try running the bayesian observer
    bayesian_session_data = compute_optimal_bayesian_actor(setup_results["envs"])
    filename = f"{save_location}_{checkpoint_number}_bayesian.pkl"
    try:
        with open(filename, "wb") as f:
            pkl.dump(bayesian_session_data, f)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
