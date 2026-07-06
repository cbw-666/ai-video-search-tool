import os
import glob
import time
import sys

# --- 1. 环境变量配置 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'models_cache')

import streamlit as st
import torch
import open_clip
from PIL import Image
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import translators as ts

# --- 配置 ---
MODEL_NAME = "ViT-H-14-quickgelu"
PRETRAINED = "dfn5b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.flv')

st.set_page_config(page_title="剪辑师AI助手 - 明星精准搜", layout="wide", page_icon="🎬")

# ==========================================
#              核心功能函数
# ==========================================

# --- 1. 免Key 自动翻译 ---
def auto_translate(text):
    if not any("\u4e00" <= char <= "\u9fff" for char in text):
        return text

    translators_list = ['alibaba', 'bing', 'google']
    for t_engine in translators_list:
        try:
            res = ts.translate_text(text, translator=t_engine, to_language='en')
            if res and isinstance(res, str):
                st.toast(f"阿里翻译: {text} -> {res}")
                return res
        except Exception:
            continue

    st.toast("翻译接口繁忙，使用原文搜索")
    return text

# --- 2. 加载 CLIP 模型 ---
@st.cache_resource
def load_clip_model():
    st.sidebar.text(f"正在加载 CLIP...\n({DEVICE})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE, precision='fp16'
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer

# --- 3. 加载 InsightFace ---
@st.cache_resource
def load_face_model():
    st.sidebar.text("正在加载人脸识别模型...")
    root_dir = os.path.join(os.getcwd(), '.insightface')
    app = FaceAnalysis(name='buffalo_l', root=root_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(480, 480))
    return app

# --- 4. 生成视频索引 ---
def generate_index(video_path, model, preprocess, progress_callback=None):
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

        if int(current_frame) % int(fps * 1) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            batch_pil.append(pil_img)
            batch_time.append(current_frame / fps)

            if len(batch_pil) >= BATCH_SIZE:
                images_tensor = torch.stack([preprocess(img) for img in batch_pil]).to(DEVICE).half()
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

    if len(batch_pil) > 0:
        images_tensor = torch.stack([preprocess(img) for img in batch_pil]).to(DEVICE).half()
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

# --- 5. 加载所有索引 ---
def load_all_indices(npz_files):
    list_emb = []
    list_time = []
    list_video_path = []
    list_source_name = []

    for file_path in npz_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            src_name = os.path.splitext(os.path.basename(file_path))[0]
            embs = data['embeddings']
            times = data['timestamps']

            if 'video_path' in data:
                v_path = str(data['video_path'])
            else:
                v_path = os.path.abspath(src_name + ".mp4")

            list_emb.append(embs)
            list_time.append(times)
            count = len(times)
            list_video_path.extend([v_path] * count)
            list_source_name.extend([src_name] * count)
        except Exception:
            pass

    if not list_emb: return None
    return {
        "embeddings": np.vstack(list_emb),
        "timestamps": np.concatenate(list_time),
        "video_paths": np.array(list_video_path),
        "source_names": np.array(list_source_name)
    }

# --- 6. 核心搜索逻辑 ---
def robust_search(clip_model, preprocess, tokenizer, face_app, combined_data, ref_img_pil, text_name,
                  clip_candidates=300, face_sim_threshold=0.45):
    
    ref_cv2 = cv2.cvtColor(np.array(ref_img_pil), cv2.COLOR_RGB2BGR)
    ref_faces = face_app.get(ref_cv2)
    if len(ref_faces) == 0:
        st.error("参考图中未检测到人脸！")
        return None
    ref_faces = sorted(ref_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    target_embedding = ref_faces[0].embedding

    video_embs = torch.tensor(combined_data['embeddings']).to(DEVICE).half()

    with torch.no_grad():
        text_tokens = tokenizer([text_name]).to(DEVICE)
        text_emb = clip_model.encode_text(text_tokens)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
        text_score = (text_emb @ video_embs.T).squeeze(0)

        img_tensor = preprocess(ref_img_pil).unsqueeze(0).to(DEVICE).half()
        img_emb = clip_model.encode_image(img_tensor)
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
        image_score = (img_emb @ video_embs.T).squeeze(0)

        final_score = 0.6 * text_score + 0.4 * image_score

    top_indices = final_score.topk(min(clip_candidates, len(video_embs))).indices.cpu().numpy()

    raw_results = []
    progress_bar = st.progress(0)
    total_check = len(top_indices)

    for i, idx in enumerate(top_indices):
        if i % 10 == 0:
            progress_bar.progress((i + 1) / total_check)

        v_path = combined_data['video_paths'][idx]
        t = combined_data['timestamps'][idx]
        src = combined_data['source_names'][idx]

        if not os.path.exists(v_path): continue
        cap = cv2.VideoCapture(v_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret: continue

        frame_faces = face_app.get(frame)
        max_sim = -1.0
        for face in frame_faces:
            sim = np.dot(target_embedding, face.embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(face.embedding))
            if sim > max_sim: max_sim = sim

        if max_sim > face_sim_threshold:
            raw_results.append(
                {"idx": idx, "score": max_sim, "timestamp": t, "video_path": v_path, "source_name": src})

    progress_bar.empty()

    # 时间去重逻辑
    raw_results.sort(key=lambda x: x["score"], reverse=True)
    filtered_results = []

    for res in raw_results:
        is_duplicate = False
        for chosen in filtered_results:
            if res["video_path"] == chosen["video_path"] and abs(res["timestamp"] - chosen["timestamp"]) <= 60:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered_results.append(res)

    return filtered_results

# ==========================================
#              主界面逻辑
# ==========================================
def main():
    st.title("AI明星搜索")

    try:
        clip_model, preprocess, tokenizer = load_clip_model()
        face_app = load_face_model()
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()

    st.sidebar.header("视频库管理")
    
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
        st.sidebar.warning(f"发现 {len(pending_videos)} 个新视频！")
        
        if st.sidebar.button("一键更新所有索引", type="primary"):
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            for i, v_file in enumerate(pending_videos):
                status_text.text(f"正在分析: {v_file}")
                
                def update_progress(p):
                    pass 

                status, msg = generate_index(v_file, clip_model, preprocess, update_progress)
                progress_bar.progress((i + 1) / len(pending_videos))
                
                if status == "SUCCESS":
                    st.toast(f"完成: {v_file}")
                elif status == "ERROR":
                    st.sidebar.error(f"{v_file}: {msg}")
            
            st.sidebar.success("全部处理完毕！")
            time.sleep(1.5)
            st.rerun()
    else:
        st.sidebar.success("所有视频均已建立索引")

    st.sidebar.markdown("---")

    if not npz_files:
        st.info("请先在左侧点击【一键更新】生成索引。")
        st.stop()

    st.sidebar.header("搜索范围")
    file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in npz_files}
    sel_videos = st.sidebar.multiselect("选择要在哪些视频里搜索:", list(file_map.keys()), default=list(file_map.keys()))

    if not sel_videos:
        st.warning("请至少选择一个视频源")
        st.stop()

    full_data = load_all_indices(npz_files)
    mask = np.isin(full_data['source_names'], sel_videos)
    active_data = {k: v[mask] for k, v in full_data.items()}
    st.sidebar.caption(f"当前池中共有: {len(active_data['timestamps'])} 个画面")

    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_file = st.file_uploader("1. 上传明星照片", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img_pil = Image.open(uploaded_file).convert("RGB")
            st.image(img_pil, width=200, caption="目标人物")

    with col2:
        st.markdown("#### 2. 身份确认")
        raw_name = st.text_input("明星名字", placeholder="尽量输入英文名字")

        with st.expander("高级参数"):
            threshold = st.slider("人脸相似度阈值", 0.0, 1.0, 0.40, help="越低越容易搜到侧脸，越高越准")
            scan_limit = st.slider("初筛扫描帧数", 50, 800, 300, help="CLIP海选范围")

        if st.button("开始搜索", type="primary"):
            if not uploaded_file or not raw_name:
                st.warning("请上传照片并输入名字")
                st.stop()

            translated_name = raw_name
            with st.spinner("正在智能翻译..."):
                translated_name = auto_translate(raw_name)

            results = robust_search(
                clip_model, preprocess, tokenizer, face_app,
                active_data, img_pil,
                translated_name,
                clip_candidates=scan_limit,
                face_sim_threshold=threshold
            )

            if not results:
                st.warning("未找到匹配镜头")
            else:
                st.success(f"找到 {len(results)} 个独立镜头")
                cols = st.columns(3)
                for i, res in enumerate(results[:30]):
                    with cols[i % 3]:
                        if os.path.exists(res['video_path']):
                            cap = cv2.VideoCapture(res['video_path'])
                            cap.set(cv2.CAP_PROP_POS_MSEC, res['timestamp'] * 1000)
                            ret, frame = cap.read()
                            cap.release()
                            if ret:
                                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                                st.markdown(f"**{res['source_name']}**")
                                m, s = divmod(res['timestamp'], 60)
                                st.caption(f"⏱️ {int(m):02d}:{int(s):02d} | 相似度: {res['score']:.2f}")

if __name__ == "__main__":
    main()