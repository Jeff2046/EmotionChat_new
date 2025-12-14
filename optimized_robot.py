# optimized_robot_int8.py
"""
优化版情绪聊天机器人 - INT8模型版

使用Qwen2.5-1.5B-Instruct INT8量化模型
- 比INT4质量好很多
- 支持GPU加速
- 智能规则回退
"""

import os
import time
import threading
import queue
import logging
import re
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
import sounddevice as sd
import json
import onnxruntime as ort
from vosk import Model as VoskModel, KaldiRecognizer
from openvino.runtime import Core
from transformers import AutoTokenizer
import pyttsx3
from collections import deque

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

FER_ONNX = os.path.join(MODEL_DIR, "emotion-ferplus-8.onnx")
VOSK_DIR = os.path.join(MODEL_DIR, "vosk-cn")
QWEN_OV_DIR = os.path.join(MODEL_DIR, "qwen2.5-1.5b-instruct-int8-ov")
QWEN_OV_XML = os.path.join(QWEN_OV_DIR, "openvino_model.xml")

# Runtime params
CAM_SEARCH_MAX = 5
EMOTION_SMOOTHING_WINDOW = 5
EMOTION_CONFIDENCE_THRESHOLD = 0.25
ENABLE_EMOTION_DEBUG = False
emotion_history = deque(maxlen=EMOTION_SMOOTHING_WINDOW)
ASR_SAMPLE_RATE = 16000
ASR_BLOCKSIZE = 8000
ASR_DEBOUNCE_SEC = 2.0
GEN_MAX_SEQ = 256
GEN_MAX_NEW_TOKENS = 30
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 50
GEN_TOP_P = 0.9

# Queues
ASR_QUEUE_MAX = 8
LLM_PROMPT_QUEUE_MAX = 3
LLM_REPLY_QUEUE_MAX = 3
TTS_QUEUE_MAX = 5

# Mode selection
USE_RULE_BASED_ONLY = False  # INT8模型，使用LLM

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("robot")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Emotion Mapping ----------------
EMOTION_MAP = {
    'neutral': '平静',
    'happiness': '开心',
    'surprise': '惊讶',
    'sadness': '难过',
    'anger': '生气',
    'disgust': '厌恶',
    'fear': '害怕',
    'contempt': '轻蔑',
    'unknown': '未知'
}

# ---------------- Rule-Based Fallback ----------------
def get_rule_based_response(user_input: str, emotion: str) -> str:
    """规则回退系统"""
    import random
    
    rules = {
        '天气': {
            'happiness': ['是啊，天气真不错！', '今天天气很好呢！'],
            'neutral': ['天气还可以。', '嗯，天气挺好的。'],
            'default': ['天气确实不错。']
        },
        '不开心|难过|伤心': {
            'sadness': ['怎么了？愿意和我说说吗？', '别难过，我在这里陪你。'],
            'default': ['发生什么事了？和我聊聊吧。']
        },
        '你好|您好|hi|hello': {
            'happiness': ['你好！很高兴见到你！', '你好呀！'],
            'neutral': ['你好！有什么可以帮你的吗？'],
            'default': ['你好！']
        },
        '棒|赞|厉害|优秀|开心|高兴': {
            'happiness': ['太好了！真为你高兴！', '是啊，真棒！'],
            'default': ['那很好啊！', '听起来不错！']
        },
        '生气|愤怒|烦': {
            'anger': ['深呼吸，慢慢说，我在听。', '别生气，告诉我发生了什么。'],
            'default': ['怎么了？什么让你不高兴了？']
        }
    }
    
    user_lower = user_input.lower()
    for pattern, responses in rules.items():
        if any(kw in user_input or kw in user_lower for kw in pattern.split('|')):
            if emotion in responses:
                return random.choice(responses[emotion]) if isinstance(responses[emotion], list) else responses[emotion]
            elif 'default' in responses:
                return random.choice(responses['default']) if isinstance(responses['default'], list) else responses['default']
    
    # 默认回复
    defaults = {
        'happiness': ['看起来你心情很好！', '你今天状态不错呢！'],
        'sadness': ['我能理解你的感受。', '我在这里陪着你。'],
        'anger': ['深呼吸，慢慢来。', '我理解你的感受。'],
        'neutral': ['我明白了。', '嗯，我在听。'],
    }
    responses = defaults.get(emotion, ['嗯，我在听。'])
    return random.choice(responses)

# ---------------- FER (ONNX) ----------------
fer_session = None
fer_input_name = None
FER_LABELS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

if os.path.exists(FER_ONNX):
    try:
        fer_session = ort.InferenceSession(FER_ONNX, providers=["CPUExecutionProvider"])
        fer_input_name = fer_session.get_inputs()[0].name
        log.info("FER loaded")
    except Exception:
        log.exception("Failed to load FER")
        fer_session = None

def preprocess_face_for_fer(face_img):
    """
    改进的FER预处理 - 直接替换原函数
    """
    try:
        # 转灰度
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()
        
        # 关键改进1: 直方图均衡化
        gray = cv2.equalizeHist(gray)
        
        # 关键改进2: 自适应对比度增强
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # resize
        img = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_CUBIC)
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # 调整维度
        img = np.expand_dims(np.expand_dims(img, 0), 0)
        
        return img
        
    except Exception as e:
        print(f"预处理失败: {e}")
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        img = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
        return np.expand_dims(np.expand_dims(img, 0), 0)

def predict_emotion_fer(face_img):
    """
    强化版 FER 预测器：
    - 温度 softmax
    - neutral 抑制
    - 动态阈值
    - 情绪平滑
    """
    global emotion_history, fer_session, fer_input_name, FER_LABELS

    if fer_session is None:
        return "neutral", 0.0

    # --- 1. 预处理 ---
    inp = preprocess_face_for_fer(face_img)

    # --- 2. 推理 ---
    logits = fer_session.run(None, {fer_input_name: inp})[0][0]

    # --- 3. 温度 softmax（降低 neutral 优势）---
    T = 1.8
    probs = np.exp((logits - np.max(logits)) / T)
    probs = probs / probs.sum()

    # --- 4. 选取前两名（neutral 抑制）---
    top2_idx = probs.argsort()[-2:][::-1]
    first, second = top2_idx[0], top2_idx[1]

    label = FER_LABELS[first]
    confidence = float(probs[first])

    # neutral 抑制机制：如果 neutral 第一但差距小于 0.08，则输出第二名
    if label == "neutral" and (probs[first] - probs[second]) < 0.08:
        label = FER_LABELS[second]
        confidence = float(probs[second])

    # --- 5. 动态阈值 ---
    dynamic_threshold = max(0.10, min(0.22, np.mean(probs)))
    if confidence < dynamic_threshold:
        label = "neutral"

    # --- 6. 情绪平滑（重要） ---
    emotion_history.append((label, confidence))

    if len(emotion_history) >= 3:
        counts = {}
        for emo, conf in emotion_history:
            counts[emo] = counts.get(emo, 0) + 1

        final = max(counts.items(), key=lambda x: x[1])[0]

        # 只有在最近窗口情绪出现至少两次才改变
        if counts[final] >= 2:
            label = final
            confidence = np.mean([c for e,c in emotion_history if e == final])

    if ENABLE_EMOTION_DEBUG:
        print("[FER+] probs=",
              ", ".join([f"{FER_LABELS[i]}:{probs[i]:.2f}" for i in range(len(FER_LABELS))]),
              f" -> {label} ({confidence:.2f})")

    return label, confidence
        
    # except Exception as e:
    #    print(f"FER推理失败: {e}")
    #    import traceback
    #    traceback.print_exc()
    #    return "unknown", 0.0

# ============ 新增：改进的人脸检测 ============
def detect_faces_improved(frame, face_detector):
    """
    改进的人脸检测
    添加此函数
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 多尺度检测，参数调优
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.05,  # 更小的步长
        minNeighbors=5,    # 稍微降低阈值
        minSize=(50, 50),  # 最小人脸
        maxSize=(300, 300),  # 最大人脸
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return faces

# ---------------- VOSK ASR ----------------
vosk_model = None
if os.path.exists(VOSK_DIR):
    try:
        vosk_model = VoskModel(VOSK_DIR)
        log.info("Vosk model loaded")
    except Exception:
        log.exception("Failed to load Vosk")

asr_queue = queue.Queue(maxsize=ASR_QUEUE_MAX)
_recognizer = None
_asr_stream = None

def _audio_callback(indata, frames, time_info, status):
    global _recognizer
    if vosk_model is None:
        return
    try:
        if _recognizer is None:
            _recognizer = KaldiRecognizer(vosk_model, ASR_SAMPLE_RATE)
        try:
            data = indata.tobytes()
        except AttributeError:
            data = bytes(memoryview(indata) if hasattr(indata, '__buffer__') else indata)
        
        if _recognizer.AcceptWaveform(data):
            res = json.loads(_recognizer.Result())
            text = res.get("text", "").strip()
            if text and len(text) >= 2:
                try:
                    asr_queue.put_nowait(text)
                except queue.Full:
                    try:
                        asr_queue.get_nowait()
                        asr_queue.put_nowait(text)
                    except:
                        pass
    except Exception:
        log.exception("ASR callback error")

def start_asr_stream():
    global _asr_stream
    if vosk_model is None or _asr_stream is not None:
        return
    try:
        _asr_stream = sd.RawInputStream(
            samplerate=ASR_SAMPLE_RATE, blocksize=ASR_BLOCKSIZE,
            dtype='int16', channels=1, callback=_audio_callback
        )
        _asr_stream.start()
        log.info("Started ASR stream")
    except Exception:
        log.exception("Failed to start ASR")

def stop_asr_stream():
    global _asr_stream
    if _asr_stream:
        try:
            _asr_stream.stop()
            _asr_stream.close()
            _asr_stream = None
        except:
            pass

# ---------------- TTS ----------------
tts_q = queue.Queue(maxsize=TTS_QUEUE_MAX)
_tts_thread = None
_tts_stop = threading.Event()

def _tts_worker():
    """改进的TTS工作线程 - 增强稳定性"""
    import pyttsx3
    import re
    
    log = logging.getLogger("robot.tts")
    
    def init_engine():
        """初始化TTS引擎"""
        try:
            log.info("Initializing pyttsx3 engine...")
            eng = pyttsx3.init()
        
            # 获取并设置中文语音
            voices = eng.getProperty('voices')
            chinese_voice = None
            for v in voices:
                vid = getattr(v, 'id', '').lower()
                vname = getattr(v, 'name', '').lower()
                if 'zh' in vid or 'chinese' in vname or 'huihui' in vname:
                    chinese_voice = v
                    break
        
            if chinese_voice:
                eng.setProperty('voice', chinese_voice.id)
                log.info(f"✓ TTS voice: {chinese_voice.name}")
        
            eng.setProperty('rate', 180)
            eng.setProperty('volume', 1.0)
        
            # 🔑 关键修复:预热引擎
            #log.info("Warming up TTS engine...")
            #eng.say("准备就绪")
            eng.runAndWait()  # 第一次使用runAndWait确保完全初始化
            log.info("✓ TTS engine warmed up")
        
            return eng
        
        except Exception as e:
            log.exception(f"✗ TTS init failed: {e}")
            return None
    
    # 初始化引擎
    engine = init_engine()
    if engine is None:
        log.error("TTS engine unavailable, worker exiting")
        return
    
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    log.info("TTS worker ready")
    
    while not _tts_stop.is_set():
        try:
            # 获取待播放文本
            text = tts_q.get(timeout=0.5)
        except queue.Empty:
            continue
        
        if not text:
            continue
        
        try:
            # 清理文本
            clean = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9,。!?、]', '', str(text))
            if not clean:
                log.warning(f"Text became empty after cleaning: '{text}'")
                continue
            
            log.info(f"🔊 TTS playing: {clean}")
            
            # 关键修复:使用startLoop避免阻塞
            engine.say(clean)
            #engine.runAndWait()
            engine.startLoop(False)  # 非阻塞模式
            engine.iterate()  # 处理一次事件
            engine.endLoop()  # 结束循环
            
            
            log.info("✓ TTS playback completed")
            
            # 播放成功,重置错误计数
            consecutive_errors = 0
            
        except RuntimeError as e:
            consecutive_errors += 1
            log.error(f"✗ TTS RuntimeError ({consecutive_errors}/{max_consecutive_errors}): {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                log.warning("Too many errors, reinitializing engine...")
                try:
                    del engine
                except:
                    pass
                
                engine = init_engine()
                if engine is None:
                    log.error("Failed to reinitialize, exiting")
                    break
                
                consecutive_errors = 0
                
        except Exception as e:
            consecutive_errors += 1
            log.exception(f"✗ TTS unexpected error: {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                log.error("Too many errors, exiting")
                break
    
    log.info("TTS worker shutdown")

def start_tts():
    global _tts_thread
    if _tts_thread and _tts_thread.is_alive():
        return
    _tts_stop.clear()
    _tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    _tts_thread.start()

def stop_tts():
    _tts_stop.set()

def speak_async(text: str):
    """异步播放语音 - 添加日志"""
    try:
        #log.info(f"📝 Adding to TTS queue: {text}")
        tts_q.put_nowait(text)
        log.info(f"✓ Added to queue, current size: {tts_q.qsize()}")
    except queue.Full:
        log.warning("TTS queue full, clearing and retrying")
        try:
            while not tts_q.empty():
                tts_q.get_nowait()
            tts_q.put_nowait(text)
            log.info("✓ Queue cleared and text added")
        except Exception as e:
            log.error(f"Failed to add to TTS queue: {e}")

# ---------------- Qwen INT8 Model ----------------
tokenizer = None
compiled_model = None
qwen_ready = False

if os.path.exists(QWEN_OV_XML):
    try:
        tokenizer = AutoTokenizer.from_pretrained(QWEN_OV_DIR, local_files_only=True, trust_remote_code=True)
        log.info(f"Loaded tokenizer from {QWEN_OV_DIR}")
        
        ov_core = Core()
        devices = ov_core.available_devices
        log.info(f"Available devices: {devices}")
        
        target_device = "GPU" if "GPU" in devices else "CPU"
        log.info(f"Using {target_device} for INT8 model")
        
        model = ov_core.read_model(QWEN_OV_XML)
        config = {"PERFORMANCE_HINT": "LATENCY"}
        if target_device == "GPU":
            config["CACHE_DIR"] = os.path.join(MODEL_DIR, "cache")
        
        compiled_model = ov_core.compile_model(model, target_device, config)
        log.info("INT8 model compiled successfully")
        qwen_ready = True
    except Exception:
        log.exception("Failed to load Qwen INT8 model")
        qwen_ready = False

def build_chat_prompt(user_input: str, emotion: str) -> str:
    """
    构建包含情绪的prompt - 替换原函数
    """
    emotion_cn = EMOTION_MAP.get(emotion, emotion)
    
    # 根据情绪调整system prompt
    emotion_hints = {
        'happiness': '用户现在很开心，请用轻松愉快的语气回复。',
        'sadness': '用户现在有些难过，请用温柔安慰的语气回复。',
        'anger': '用户现在有些生气，请用理解和安抚的语气回复。',
        'surprise': '用户现在很惊讶，请用共鸣的语气回复。',
        'fear': '用户现在有些害怕，请用支持鼓励的语气回复。',
        'disgust': '用户现在感到厌恶，请用理解的语气回复。',
        'contempt': '用户现在有些轻蔑，请用尊重的语气回复。',
        'neutral': '用户现在情绪平静，请自然地回复。',
    }
    
    hint = emotion_hints.get(emotion, '请自然地回复用户。')
    
    system_msg = f"你是一个善解人意的AI助手。{hint}用简短、温暖的中文回复，不超过100字。"
    
    user_msg = f"{user_input}"
    
    # Qwen2.5标准格式
    prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
    return prompt

def qwen_generate(prompt: str, user_input: str, emotion: str) -> str:
    """INT8模型生成"""
    if not qwen_ready or not compiled_model or not tokenizer:
        return get_rule_based_response(user_input, emotion)
    
    t0 = time.time()
    try:
        # 创建新的推理请求
        infer_req = compiled_model.create_infer_request()
        
        # 编码
        enc = tokenizer(prompt, return_tensors="np", max_length=GEN_MAX_SEQ, truncation=True)
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)
        
        current_seq = input_ids.tolist()[0]
        generated_tokens = []
        special_tokens = {151643, 151644, 151645}
        
        for step in range(GEN_MAX_NEW_TOKENS):
            # 准备输入
            seq_len = len(current_seq)
            token_ids = np.array([current_seq], dtype=np.int64)
            att_mask = np.ones_like(token_ids, dtype=np.int64)
            pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            beam_idx = np.zeros((1,), dtype=np.int32)
            
            inputs = {
                "input_ids": token_ids,
                "attention_mask": att_mask,
                "position_ids": pos_ids,
                "beam_idx": beam_idx
            }
            
            # 推理
            infer_req.infer(inputs)
            logits = infer_req.get_output_tensor().data[0, -1, :].astype(np.float32)
            
            # 惩罚特殊tokens
            for sid in special_tokens:
                if sid < len(logits):
                    logits[sid] = -np.inf
            
            # 适度重复惩罚
            for j in range(min(5, len(generated_tokens))):
                tok = generated_tokens[-(j+1)]
                if tok < len(logits):
                    logits[tok] -= (3.0 - j * 0.3)
            
            # 采样
            scaled = logits / GEN_TEMPERATURE
            probs = np.exp(scaled - np.max(scaled))
            probs = probs / probs.sum()
            
            # Top-k
            if GEN_TOP_K > 0:
                top_k_indices = np.argpartition(-probs, GEN_TOP_K)[:GEN_TOP_K]
                mask = np.zeros_like(probs)
                mask[top_k_indices] = 1.0
                probs = probs * mask
                probs = probs / probs.sum()
            
            next_token = int(np.random.choice(len(probs), p=probs))
            
            current_seq.append(next_token)
            generated_tokens.append(next_token)
            
            # 停止条件
            if next_token in special_tokens:
                break
            
            if len(generated_tokens) >= 10:
                text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                if text.endswith(('。', '！', '？')):
                    break
                if len(text) >= 30:
                    break
        
        # 解码
        reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        reply = re.sub(r'<\|[^>]+\|>', '', reply)
        reply = re.sub(r'[\n\r]+', ' ', reply)
        
        # 清理污染
        if '助手' in reply or '用户' in reply:
            if '助手：' in reply:
                reply = reply.split('助手：')[-1]
            if '用户：' in reply:
                reply = reply.split('用户：')[0]
            reply = reply.strip()
        
        # 只保留第一句
        for punct in ['。', '！', '？']:
            if punct in reply:
                reply = reply[:reply.index(punct)+1]
                break
        
        if len(reply) > 30:
            reply = reply[:30]
        
        t1 = time.time()
        
        # 验证输出
        clean = re.sub(r'[：:，。！？\s]+', '', reply)
        if len(clean) < 3 or any(w in reply for w in ['用户', '助手', 'Assistant']):
            log.warning(f"Invalid LLM output, using fallback: '{reply}'")
            return get_rule_based_response(user_input, emotion)
        
        log.info(f"[{t1-t0:.2f}s|{len(generated_tokens)}tok|LLM] '{reply}'")
        return reply
        
    except Exception as e:
        log.exception(f"LLM generation failed: {e}")
        return get_rule_based_response(user_input, emotion)

# ---------------- Response Worker ----------------
response_prompt_q = queue.Queue(maxsize=LLM_PROMPT_QUEUE_MAX)
response_reply_q = queue.Queue(maxsize=LLM_REPLY_QUEUE_MAX)
_response_thread = None
_response_stop = threading.Event()

def _response_worker():
    mode = "RULE-BASED" if USE_RULE_BASED_ONLY else "LLM+RULE"
    log.info(f"Response worker started ({mode} MODE)")
    
    while not _response_stop.is_set():
        try:
            item = response_prompt_q.get(timeout=0.5)
            if len(item) != 3:
                continue
            user_input, req_id, emotion = item
        except queue.Empty:
            continue
        except Exception:
            log.exception("Error unpacking queue item")
            continue
        
        try:
            if USE_RULE_BASED_ONLY:
                reply = get_rule_based_response(user_input, emotion)
            else:
                prompt = build_chat_prompt(user_input, emotion)
                reply = qwen_generate(prompt, user_input, emotion)
        except Exception:
            log.exception("Response generation error")
            reply = get_rule_based_response(user_input, emotion)
        
        try:
            response_reply_q.put_nowait((req_id, reply))
        except queue.Full:
            try:
                response_reply_q.get_nowait()
                response_reply_q.put_nowait((req_id, reply))
            except:
                pass

def start_response_worker():
    global _response_thread
    if _response_thread and _response_thread.is_alive():
        return
    _response_stop.clear()
    _response_thread = threading.Thread(target=_response_worker, daemon=True)
    _response_thread.start()

def stop_response_worker():
    _response_stop.set()

# ---------------- Camera ----------------
def open_camera_search(max_id=CAM_SEARCH_MAX):
    for i in range(max_id):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                log.info(f"Using camera id {i}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
        cap.release()
    return None

# ---------------- Main Loop ----------------
def main_loop():
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = open_camera_search()
    if not cap:
        log.error("No camera available")
        return

    start_asr_stream()
    start_tts()
    start_response_worker()

    last_asr_time = 0.0
    last_processed_asr = ""
    req_id_counter = 0

    log.info("Main loop started. Press 'q' to exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # 人脸检测和情绪识别
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # faces = face_detector.detectMultiScale(gray, 1.1, 4)
            faces = detect_faces_improved(frame, face_detector)
            
            display_emotion = "neutral"
            display_conf = 0.0
            
            if len(faces) > 0 and fer_session:
                x, y, w, h = faces[0]
                # face_roi = frame[y:y+h, x:x+w]
                padding = int(h * 0.15)
                x_new = max(0, x - padding)
                y_new = max(0, y - padding)
                w_new = min(frame.shape[1] - x_new, w + 2 * padding)
                h_new = min(frame.shape[0] - y_new, h + 2 * padding)
                face_roi = frame[y_new:y_new+h_new, x_new:x_new+w_new]

                label, conf = predict_emotion_fer(face_roi)
                display_emotion, display_conf = label, conf
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # cv2.putText(frame, f"{EMOTION_MAP.get(label, label)} {conf:.2f}", 
                #           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if ENABLE_EMOTION_DEBUG and display_emotion != 'neutral':
                    # 显示更明显的标记
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
                    # 在窗口显示完整的情绪信息
                    debug_text = f"Emotion: {EMOTION_MAP.get(display_emotion)} (conf: {display_conf:.2f})"
                    cv2.putText(frame, debug_text, (10, 60),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            else:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 状态显示
            mode_text = "INT8-LLM MODE" if qwen_ready and not USE_RULE_BASED_ONLY else "RULE MODE"
            cv2.putText(frame, mode_text, (10, frame.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("EmotionChatBot (Press q to quit)", frame)

            # 处理ASR
            asr_text = None
            while not asr_queue.empty():
                candidate = asr_queue.get_nowait().strip()
                if candidate:
                    asr_text = candidate
            
            if asr_text:
                tnow = time.time()
                if asr_text != last_processed_asr or (tnow - last_asr_time) >= ASR_DEBOUNCE_SEC:
                    last_processed_asr = asr_text
                    last_asr_time = tnow
                    req_id_counter += 1
                    
                    log.info(f"ASR #{req_id_counter}: '{asr_text}' | 情绪={EMOTION_MAP.get(display_emotion)}({display_emotion}) conf={display_conf:.2f}")
                    
                    while not response_prompt_q.empty():
                        try:
                            response_prompt_q.get_nowait()
                        except:
                            break
                    
                    try:
                        response_prompt_q.put_nowait((asr_text, req_id_counter, display_emotion))
                    except queue.Full:
                        pass
            
            # 处理回复
            while not response_reply_q.empty():
                rid, reply = response_reply_q.get_nowait()
                log.info(f"Reply #{rid}: {reply}")
                speak_async(reply)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_asr_stream()
        stop_tts()
        stop_response_worker()
        time.sleep(1)

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("EmotionChatBot - INT8 LLM Edition")
    log.info("Using Qwen2.5-1.5B-Instruct INT8")
    log.info("=" * 60)
    main_loop()
    log.info("Shutdown complete")
