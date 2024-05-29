from typing import Callable
import numpy as np


def ordered_halving(val):
    # 对64位整数输入进行二进制翻转，避免重复采样，且具有一定的顺序性和可复现性
    val = val % (1 << 64)
    val = ((val & 0x5555555555555555) << 1) | ((val & 0xAAAAAAAAAAAAAAAA) >> 1)
    val = ((val & 0x3333333333333333) << 2) | ((val & 0xCCCCCCCCCCCCCCCC) >> 2)
    val = ((val & 0x0F0F0F0F0F0F0F0F) << 4) | ((val & 0xF0F0F0F0F0F0F0F0) >> 4)
    val = ((val & 0x00FF00FF00FF00FF) << 8) | ((val & 0xFF00FF00FF00FF00) >> 8)
    val = ((val & 0x0000FFFF0000FFFF) << 16) | ((val & 0xFFFF0000FFFF0000) >> 16)
    val = (val << 32) | (val >> 32)
    return val / (1 << 64)

def uniform(
    step: int,
    num_steps: int,
    num_frames: int,
    context_size: int,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)
    halved = ordered_halving(step)
    pad = int(round(num_frames * halved))

    for context_step in 1 << np.arange(context_stride):
        for j in range(
            int(halved * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap)
        ):
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]

def random(
    step: int,
    num_steps: int,
    num_frames: int,
    context_size: int,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    num_samples = num_frames // context_size

    for _ in range(num_samples):
        start = np.random.randint(0, num_frames - context_size)
        yield list(range(start, start + context_size))

def get_total_steps(
    scheduler,
    timesteps: list[int],
    num_steps: int,
    num_frames: int,
    context_size: int,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return sum(
        len(list(scheduler(i, num_steps, num_frames, context_size, context_stride, context_overlap, closed_loop)))
        for i in range(len(timesteps))
    )

def get_context_scheduler(name: str) -> Callable:
    match name:
        case "uniform":
            return uniform
        case "random":
            return random
        case _:
            raise ValueError(f"Unknown context_overlap policy {name}")
