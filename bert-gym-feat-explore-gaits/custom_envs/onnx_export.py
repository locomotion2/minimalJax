import argparse
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch as th
from sb3_contrib import TQC

# path = "/home/raff_an/USERDIR/projects/bert/bert_utils/logs/rl/BertPronk-v1_4/BertPronk-v1.zip"
# output_path = Path("/home/raff_an/USERDIR/projects/bert/bert_utils/logs/rl/")


class OnnxablePolicy(th.nn.Module):
    def __init__(self, actor: th.nn.Module):
        super().__init__()
        # Removing the flatten layer because it can't be onnxed
        clip_mean = actor.clip_mean
        self.actor = th.nn.Sequential(
            actor.latent_pi,
            actor.mu,
            th.nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean),
            th.nn.Tanh(),
        )

    def forward(self, observation: th.Tensor) -> th.Tensor:
        return self.actor(observation)


def benchmark(pred_fn: Callable, n_iterations: int = 1000, name: str = ""):
    times = np.zeros((n_iterations,))
    for i in range(n_iterations):
        start_time = time.time_ns()
        pred_fn()
        end_time = time.time_ns()
        times[i] = end_time - start_time
    mean_time = times.mean() * 1e-6
    std_time = times.std() * 1e-6
    total_time = times.sum() * 1e-6
    print(f"{name} took {mean_time:.2f}ms +/- {std_time:.2f}ms - Total {total_time:.2f}ms")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, required=True)
    parser.add_argument("-o", "--output-folder", type=str, required=True)
    args = parser.parse_args()
    path = args.input_file
    output_path = Path(args.output_folder)
    os.makedirs(output_path, exist_ok=True)

    with th.no_grad():
        model = TQC.load(path, device="cpu")
        observation_size = model.observation_space.shape
        dummy_input = th.randn(1, *observation_size)
        observation = dummy_input.numpy()
        action = model.predict(observation, deterministic=True)[0]

        onnx_path = str(output_path / "tqc_actor.onnx")
        jit_path = str(output_path / "tqc_actor_traced.pt")
        onnxable_model = OnnxablePolicy(model.policy.actor).to("cpu")

        th.onnx.export(
            onnxable_model,
            dummy_input,
            onnx_path,
            opset_version=15,
            input_names=["input"],
        )
        traced_module = th.jit.trace(onnxable_model.eval(), dummy_input)
        frozen_module = th.jit.freeze(traced_module)
        # frozen_module = th.jit.optimize_for_inference(frozen_module)
        th.jit.save(frozen_module, jit_path)

        # Test loading
        loaded_module = th.jit.load(jit_path)
        ort_sess = ort.InferenceSession(onnx_path)
        action_onnx_2 = ort_sess.run(None, {"input": observation})[0]

        # Check that actions are correct
        action_jit = frozen_module(dummy_input).numpy()
        action_loaded_jit = loaded_module(dummy_input).numpy()
        action_onnxable_model = onnxable_model(dummy_input).numpy()

        benchmark(lambda: model.predict(observation, deterministic=True), name="PyTorch model")
        benchmark(lambda: traced_module(dummy_input).numpy(), name="traced_module")
        benchmark(lambda: frozen_module(dummy_input).numpy(), name="frozen_module")
        benchmark(lambda: ort_sess.run(None, {"input": observation}), name="onnx")

    assert np.allclose(action, action_onnxable_model), f"{action_onnxable_model}, {action}"
    assert np.allclose(action, action_jit), f"{action_jit}, {action}"
    assert np.allclose(action, action_loaded_jit), f"{action_loaded_jit}, {action}"
    assert np.allclose(action, action_onnx_2), f"{action_onnx_2}, {action}"
    print(f"Exported model to {output_path}")
