import os
import glob
import time
import sys

# --- 1. 环境变量配置 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'models_cache')

import streamlit as st
import numpy as np
import torch
import open_clip
import translators as ts
import cv2
from PIL import Image

# --- 配置 ---
MODEL_NAME = "ViT-H-14-quickgelu"
PRETRAINED = "dfn5b"
# 支持的视频格式
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv')

st.set_page_config(page_title="AI场景语义搜索", layout="wide", page_icon="🎬")


# ==========================================
#              核心功能函数
# ==========================================

# --- 1. 实时读取视频帧 ---
def get_frame_from_video(video_path, timestamp_sec):
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


# --- 2. 免Key 自动翻译 ---
def auto_translate(text):
    if not any("\u4e00" <= char <= "\u9fff" for char in text):
        return text
    translators_list = ['alibaba', 'bing', 'google']
    for t_engine in translators_list:
        try:
            res = ts.translate_text(text, translator=t_engine, to_language='en')
            if res and isinstance(res, str):
                st.toast(f"✅ {t_engine}翻译: {text} -> {res}", icon="🔄")
                return res
        except Exception:
            continue
    st.toast("⚠️ 翻译接口繁忙，使用原文搜索", icon="😐")
    return text


# --- 3. 加载模型 (仅CLIP) ---
@st.cache_resource
def load_clip_model():
    if torch.cuda.is_available():
        device = 'cuda'
        precision = 'fp16'
        st.sidebar.success("使用 GPU 模式")
    else:
        device = 'cpu'
        precision = 'fp32'
        st.sidebar.warning("使用 CPU 模式")

    st.sidebar.text(f"🚀 加载模型: {MODEL_NAME}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=device, precision=precision
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer, device


# --- 4. 生成索引 (核心逻辑) ---
def generate_index(video_path, model, preprocess, device, progress_callback=None):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = f"{base_name}.npz"

    if os.path.exists(output_file):
        return "SKIP", f"已存在: {base_name}"
    if not os.path.exists(video_path):
        return "ERROR", "文件不存在"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return "ERROR", "无法读取帧"

    all_embeddings = []
    all_timestamps = []
    batch_pil = []
    batch_time = []
    BATCH_SIZE = 32

    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 每 1 秒截 1 帧
        if int(current_frame) % int(fps * 1) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            batch_pil.append(pil_img)
            batch_time.append(current_frame / fps)

            if len(batch_pil) >= BATCH_SIZE:
                images_tensor = torch.stack([preprocess(img) for img in batch_pil]).to(device)
                if device == 'cuda':
                    images_tensor = images_tensor.half()

                with torch.no_grad():
                    emb = model.encode_image(images_tensor)
                    emb /= emb.norm(dim=-1, keepdim=True)

                all_embeddings.append(emb.cpu().numpy())
                all_timestamps.extend(batch_time)
                batch_pil = []
                batch_time = []

                if progress_callback:
                    progress_callback(current_frame / total_frames)

        current_frame += 1

    cap.release()

    # 处理尾巴
    if len(batch_pil) > 0:
        images_tensor = torch.stack([preprocess(img) for img in batch_pil]).to(device)
        if device == 'cuda':
            images_tensor = images_tensor.half()
        with torch.no_grad():
            emb = model.encode_image(images_tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu().numpy())
        all_timestamps.extend(batch_time)

    if len(all_embeddings) > 0:
        final_emb = np.vstack(all_embeddings)
        np.savez(
            output_file,
            embeddings=final_emb,
            timestamps=np.array(all_timestamps),
            video_path=os.path.abspath(video_path)
        )
        return "SUCCESS", f"完成: {base_name}"
    else:
        return "ERROR", "无特征"


# --- 5. 加载已有索引 ---
@st.cache_resource
def load_index_data(npz_files):
    list_emb = []
    list_time = []
    list_video_path = []
    list_source_name = []

    for file_path in npz_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            src_name = os.path.splitext(os.path.basename(file_path))[0]

            t = data['timestamps']
            list_emb.append(data['embeddings'])
            list_time.append(t)

            # 兼容旧数据路径
            if 'video_path' in data:
                v_path = str(data['video_path'])
            else:
                v_path = os.path.abspath(src_name + ".mp4")

            list_video_path.extend([v_path] * len(t))
            list_source_name.extend([src_name] * len(t))
        except Exception:
            pass

    if list_emb:
        return (np.vstack(list_emb),
                np.concatenate(list_time),
                np.array(list_video_path),
                np.array(list_source_name))
    return None


# ==========================================
#              主界面逻辑
# ==========================================
def main():
    st.title("🎬 AI视频场景搜索")

    # 1. 加载模型
    try:
        model, preprocess, tokenizer, device = load_clip_model()
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()

    # ==========================
    #  侧边栏区域 1: 视频库维护
    # ==========================
    st.sidebar.markdown("### 📂 第一步：视频库维护")

    current_dir = os.getcwd()
    video_files = [f for f in os.listdir(current_dir) if f.lower().endswith(VIDEO_EXTENSIONS)]
    npz_files = glob.glob(os.path.join(current_dir, "*.npz"))
    indexed_names = {os.path.splitext(os.path.basename(n))[0] for n in npz_files}

    pending_videos = []
    for v in video_files:
        v_name = os.path.splitext(v)[0]
        if v_name not in indexed_names:
            pending_videos.append(v)

    col_a, col_b = st.sidebar.columns(2)
    col_a.metric("总视频", len(video_files))
    col_b.metric("待处理", len(pending_videos))

    if pending_videos:
        st.sidebar.warning(f"🟠 发现 {len(pending_videos)} 个新视频！")
        if st.sidebar.button("🔨 一键更新所有索引", type="primary"):
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()

            for i, v_file in enumerate(pending_videos):
                status_text.text(f"正在分析: {v_file}")

                def update_progress(p):
                    pass

                status, msg = generate_index(v_file, model, preprocess, device, update_progress)
                progress_bar.progress((i + 1) / len(pending_videos))

                if status == "SUCCESS":
                    st.toast(f"✅ 完成: {v_file}")
                elif status == "ERROR":
                    st.sidebar.error(f"{v_file}: {msg}")

            st.sidebar.success("🎉 处理完毕！")
            time.sleep(1)
            st.rerun()
    else:
        st.sidebar.success("🟢 索引已最新")

    st.sidebar.markdown("---")

    # ==========================
    #  侧边栏区域 2: 搜索范围
    # ==========================
    if not npz_files:
        st.info("👋 请先在左侧点击【一键更新】生成索引。")
        st.stop()

    st.sidebar.markdown("### 🔍 第二步：选择搜索范围")
    file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in npz_files}
    all_names = list(file_map.keys())

    selected_names = st.sidebar.multiselect(
        "选择要在哪些视频里搜索:",
        options=all_names,
        default=all_names
    )

    if not selected_names:
        st.warning("⚠️ 请至少选择一个视频！")
        st.stop()

    # 2. 加载数据
    data_resources = load_index_data(npz_files)
    if not data_resources:
        st.stop()

    full_emb, full_time, full_vpath, full_source = data_resources

    # 3. 内存过滤
    mask = np.isin(full_source, selected_names)
    active_emb = full_emb[mask]
    active_time = full_time[mask]
    active_vpath = full_vpath[mask]
    active_source = full_source[mask]

    st.sidebar.caption(f"当前池中共有: {len(active_time)} 个画面")

    # ==========================
    #      主界面：搜索
    # ==========================
    query = st.text_input("🔍 请输入内容 (模型只支持英文输入，输入中文后会走接口翻译成英文)", placeholder="输入画面描述...")

    # 高级选项
    with st.expander("⚙️ 显示设置"):
        top_k = st.slider("显示结果数量", 1, 50, 9)
        min_score = st.slider("最低匹配度阈值", 0.0, 1.0, 0.2, step=0.05)

    if query:
        st.markdown("### 🎯 搜索结果")

        # 翻译
        with st.spinner("正在解析语义..."):
            search_text = auto_translate(query)

        # CLIP 文本编码
        text_tokens = tokenizer([search_text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        text_np = text_features.cpu().numpy()
        scores = (active_emb @ text_np.T).squeeze()

        # --- 核心修改：时间去重逻辑 ---
        # 1. 先把所有结果按分数从高到低排好
        # 这里取更多的候选结果（比如排名前500），为了在去重后还能凑够 top_k
        sorted_indices = scores.argsort()[::-1]

        filtered_results = []

        # 2. 遍历候选结果，进行去重
        for idx in sorted_indices:
            # 如果找够了用户想要数量，就停止
            if len(filtered_results) >= top_k:
                break

            v_path = active_vpath[idx]
            t = active_time[idx]
            score = scores[idx]
            src = active_source[idx]

            # 3. 检查是否与已有结果“撞车”（同一视频且时间差在 60 秒内）
            is_duplicate = False
            for res in filtered_results:
                if res['video_path'] == v_path and abs(res['timestamp'] - t) < 60:
                    is_duplicate = True
                    break

            # 4. 如果不是重复的，就加入结果集
            if not is_duplicate:
                filtered_results.append({
                    'video_path': v_path,
                    'timestamp': t,
                    'score': score,
                    'source_name': src
                })

        # --- 展示逻辑 ---
        filtered_results = [res for res in filtered_results if res['score'] >= min_score]

        if not filtered_results:
            st.info("😔 没有找到匹配度足够高的结果，试试降低阈值或换个描述？")
            st.stop()

        cols = st.columns(3)
        for i, res in enumerate(filtered_results):
            col_idx = i % 3
            with cols[col_idx]:
                v_path = res['video_path']
                t = res['timestamp']
                score = res['score']
                src = res['source_name']

                img = get_frame_from_video(v_path, t)

                if img is not None:
                    st.image(img, use_container_width=True)
                    st.markdown(f"**🎬 {src}**")
                    m, s = divmod(int(t), 60)
                    st.caption(f"⏱️ {m:02d}:{s:02d} | 匹配度: {score:.3f}")
                else:
                    st.error(f"源文件丢失: {src}")


if __name__ == "__main__":
    main()