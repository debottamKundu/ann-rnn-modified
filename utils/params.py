train_params = {
    "model": {
        "architecture": "rnn",
        "kwargs": {
            "input_size": 3,
            "output_size": 2,
            "core_kwargs": {"num_layers": 1, "hidden_size": 15},
            "param_init": "default",
            "connectivity_kwargs": {
                "input_mask": "none",
                "recurrent_mask": "none",
                "readout_mask": "none",
            },
        },
    },
    "optimizer": {
        "optimizer": "sgd",
        "kwargs": {
            "lr": 1e-3,
            "momentum": 0.1,
            "nesterov": False,
            "weight_decay": 0,
        },
        "description": "Vanilla SGD",
    },
    "loss_fn": {"loss_fn": "nll"},
    "run": {
        "start_grad_step": 0,
        "num_grad_steps": 100001,
        "seed": 1,
    },
    "env": {
        "num_sessions": 1,  # batch size
        "kwargs": {
            "num_stimulus_strength": 6,
            "min_stimulus_strength": 0,
            "max_stimulus_strength": 1.5,
            "block_side_probs": ((0.8, 0.2), (0.2, 0.8)),
            "trials_per_block_param": 1 / 50,
            "blocks_per_session": 4,
            "min_trials_per_block": 20,
            "max_trials_per_block": 100,
            "max_obs_per_trial": 12,
            "rnn_steps_before_obs": 1,
            "time_delay_penalty": -0.005,
        },
    },
    "description": "Low time delay penalty of 0.005, 1 step before RNN is allowed to respond, lower max stim strength, 1.25 variance. higher penaly for mistakes compared to time delay",
}
