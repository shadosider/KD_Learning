import time

import torch


def count_parameters(model) -> int:
    """返回模型参数总数。"""
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_latency(model, loader, device, warmup_batches: int = 5, max_batches=None):
    """返回 (每批耗时 ms, 每张图片耗时 ms)。

    先跑 warmup_batches 个批次预热,再统计至多 max_batches 个批次的平均耗时。
    CUDA 下用 CUDA Event 计时(GPU 侧时间),CPU 下用 perf_counter;
    数据加载不在计时范围内。
    """
    model.eval()

    use_events = device.type == "cuda"
    total_time = 0.0
    total_images = 0
    timed_batches = 0

    for i, (images, _) in enumerate(loader):
        images = images.to(device)

        if use_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(images)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0  # 秒
        else:
            start = time.perf_counter()
            model(images)
            elapsed = time.perf_counter() - start

        if i >= warmup_batches:
            total_time += elapsed
            total_images += images.size(0)
            timed_batches += 1
            if max_batches is not None and timed_batches >= max_batches:
                break

    ms_per_batch = total_time / timed_batches * 1000.0
    ms_per_image = total_time / total_images * 1000.0
    return ms_per_batch, ms_per_image
