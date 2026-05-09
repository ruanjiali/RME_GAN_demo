import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lib import loaders, modules
import test as demo_test

try:
    import streamlit as st
except Exception as e:
    raise SystemExit(f"streamlit 未安装或不可用：{e}")


def _make_dataset(dataset_name, dataset_dir, setup):
    if dataset_name == "radiounet":
        if setup == 1:
            return loaders.RadioUNet_s(phase="test", fix_samples=655, num_samples_low=10, num_samples_high=300, dir_dataset=dataset_dir)
        if setup == 2:
            return loaders.RadioUNet_s(phase="test", fix_samples=1, num_samples_low=655, num_samples_high=655 * 10, dir_dataset=dataset_dir)
        return loaders.RadioUNet_s(phase="test", fix_samples=0, num_samples_low=655, num_samples_high=655 * 10, dir_dataset=dataset_dir)

    if setup == 1:
        return loaders.RadioMapSeerPolygon(phase="test", fix_samples=655, num_samples_low=10, num_samples_high=300, dir_dataset=dataset_dir)
    if setup == 2:
        return loaders.RadioMapSeerPolygon(phase="test", fix_samples=1, num_samples_low=655, num_samples_high=655 * 10, dir_dataset=dataset_dir)
    return loaders.RadioMapSeerPolygon(phase="test", fix_samples=0, num_samples_low=655, num_samples_high=655 * 10, dir_dataset=dataset_dir)


def _load_model(weights_path):
    model = modules.RadioWNet(phase="firstU")
    loaded = demo_test._torch_load_compat(weights_path)
    state = demo_test._extract_generator_state_dict(loaded)
    model.load_state_dict(state)
    return model


def _read_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def _gray_to_heatmap(gray):
    g_min = float(gray.min())
    g_max = float(gray.max())
    denom = max(g_max - g_min, 1e-6)
    x = (gray - g_min) / denom

    # Simple blue->cyan->yellow->red colormap for clear demo visualization.
    points = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
    r_vals = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    g_vals = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    b_vals = np.array([0.5, 1.0, 0.0, 0.0], dtype=np.float32)

    r = np.interp(x, points, r_vals)
    g = np.interp(x, points, g_vals)
    b = np.interp(x, points, b_vals)
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb, g_min, g_max


def _gray_to_rgb(gray):
    v = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([v, v, v], axis=-1)


st.set_page_config(page_title="无线网络覆盖预测 Demo", layout="wide")
st.title("无线网络覆盖预测 Demo（Streamlit）")

with st.sidebar:
    st.subheader("配置")
    dataset_name = st.selectbox("数据集", ["radiomapseer_polygon", "radiounet"])
    dataset_dir = st.text_input("dataset_dir", value=r"D:\RME-GAN\RadioMapSeer" if dataset_name == "radiomapseer_polygon" else "")
    setup = st.selectbox("setup（稀疏采样）", [1, 2, 3], index=0)

    mode = st.selectbox("模式", ["baseline_prior", "baseline_samples", "model"], index=0)
    weights_path = st.text_input("weights（仅 mode=model）", value=r"radiomapseer_polygon_uniform_1\checkpoint_last.pt")

    db_min = st.number_input("db_min", value=-140.0, step=1.0)
    db_max = st.number_input("db_max", value=-40.0, step=1.0)
    threshold_db = st.number_input("threshold_db", value=7.0, step=0.5)

    batch_size = st.number_input("batch_size", value=15, step=1, min_value=1)
    num_workers = st.number_input("num_workers", value=0, step=1, min_value=0)
    max_batches = st.number_input("max_batches（用于快速演示）", value=1, step=1, min_value=1)
    save_images = st.checkbox("保存图片到 out_dir", value=True)
    show_heatmap = st.checkbox("预测与真值使用彩色热力图", value=True)
    out_dir = st.text_input("out_dir", value="demo_out_streamlit")

    run = st.button("运行 Demo", type="primary")


if run:
    if not dataset_dir:
        st.error("dataset_dir 不能为空")
        st.stop()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    st.info(f"device: {device}")

    ds = _make_dataset(dataset_name=dataset_name, dataset_dir=dataset_dir, setup=int(setup))

    model = None
    if mode == "model":
        if not weights_path:
            st.error("mode=model 时必须填写 weights")
            st.stop()
        model = _load_model(weights_path)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with st.spinner("正在评测并生成可视化..."):
        metrics = demo_test.evaluate_with_mode(
            mode=mode,
            model=model,
            dataset=ds,
            device=device,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            save_dir=str(out_path),
            save_images=bool(save_images),
            max_batches=int(max_batches),
            value_scale=1.0,
            value_offset=0.0,
            db_min=float(db_min),
            db_max=float(db_max),
            threshold_db=float(threshold_db),
        )

    payload = {
        "dataset": dataset_name,
        "dataset_dir": dataset_dir,
        "setup": int(setup),
        "mode": mode,
        "db_min": float(db_min),
        "db_max": float(db_max),
        "threshold_db": float(threshold_db),
        "metrics": metrics,
    }
    if save_images:
        demo_test._write_metrics_json(str(out_path), payload)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("指标")
        st.json(
            {
                "mode": mode,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "p90": metrics["p90"],
                "p95": metrics["p95"],
                "within_threshold_ratio": metrics["within_threshold_ratio"],
                "infer_seconds": metrics["infer_seconds"],
                "pixels": metrics["pixels"],
            }
        )

        st.download_button(
            "下载 metrics.json",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="metrics.json",
            mime="application/json",
        )

    with right:
        st.subheader("样例可视化")
        with st.expander("图片说明（文件名 → 含义）", expanded=True):
            st.markdown(
                "- `in_buildings_*.png`：输入通道 buildings/env（建筑物/环境栅格）\n"
                "- `in_tx_*.png`：输入通道 tx_mask（发射点位掩码）\n"
                "- `in_samples_*.png`：输入通道 sparse_samples（稀疏测量点）\n"
                "- `in_prior_*.png`：输入通道 prior（先验图/粗预测）\n"
                "- `pred_*.png`：预测覆盖图（model 或 baseline）\n"
                "- `gt_*.png`：真值/标签覆盖图（用于对比）\n"
            )
        saved = metrics.get("saved_images", []) or []
        if not saved:
            st.warning("未生成图片。请勾选“保存图片到 out_dir”。")
        else:
            for s in saved[:6]:
                st.markdown(f"**batch={s['batch']} index={s['index']}**")
                cols = st.columns(3)
                files = [
                    ("输入: buildings/env", s["paths"]["in_buildings"]),
                    ("输入: tx_mask", s["paths"]["in_tx"]),
                    ("输入: sparse_samples", s["paths"]["in_samples"]),
                    ("输入: prior", s["paths"]["in_prior"]),
                    ("输出: pred", s["paths"]["pred"]),
                    ("对比: gt", s["paths"]["gt"]),
                ]
                for i, (title, filename) in enumerate(files):
                    gray = _read_gray(out_path / filename)
                    use_heat = show_heatmap and (title.startswith("输出") or title.startswith("对比") or title.endswith("prior"))
                    if use_heat:
                        disp, vmin, vmax = _gray_to_heatmap(gray)
                    else:
                        disp, vmin, vmax = _gray_to_rgb(gray), float(gray.min()), float(gray.max())
                    caption = f"{title}（{filename}） range=[{vmin:.1f}, {vmax:.1f}]"
                    cols[i % 3].image(disp, caption=caption, use_container_width=True)

    st.caption(f"输出目录：{out_path.resolve()}")

else:
    st.markdown(
        "说明：选择 baseline_prior / baseline_samples 可以在不训练模型的情况下完成 Demo；"
        "选择 model 则需要提供训练得到的 checkpoint 或权重文件。"
    )
