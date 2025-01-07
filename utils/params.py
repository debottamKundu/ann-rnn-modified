train_params = {
    "model": {
        "architecture": "rnn",
        "kwargs": {
            "input_size": 3,
            "output_size": 2,
            "core_kwargs": {"num_layers": 1, "hidden_size": 10},
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
        "num_grad_steps": 50000,
        "seed": 1,
    },
    "env": {
        "num_sessions": 1,  # batch size
        "kwargs": {
            "num_stimulus_strength": 2,
            "min_stimulus_strength": 1,
            "max_stimulus_strength": 2.5,
            "block_side_probs": ((0.5, 0.5), (0.5, 0.5)),
            "trials_per_block_param": 1 / 50,
            "blocks_per_session": 4,
            "min_trials_per_block": 20,
            "max_trials_per_block": 100,
            "max_obs_per_trial": 12,
            "rnn_steps_before_obs": 1,
            "time_delay_penalty": -0.05,
        },
    },
    "description":"Only one stimulus strength, max stimulus strength of 2.5 and variance of 1.0, more hidden layer, 2 stim strengthss",
}
