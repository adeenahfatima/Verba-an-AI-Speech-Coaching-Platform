import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from faster_whisper import WhisperModel
import nltk
import os
import gc
import tempfile
from flask_bcrypt import Bcrypt
from datetime import datetime
import numpy as np
from scipy.ndimage import median_filter
from pydub import AudioSegment
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Download NLTK data only if not already present
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('words', quiet=True)

load_dotenv()

import sys

# Serve frontend static files from the frontend directory
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-in-prod')
CORS(app)  # Enable CORS for all routes

# --- Database setup ---
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///verba.db')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

bcrypt = Bcrypt(app)

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    uploads = db.relationship('Upload', backref='user', lazy=True)

class Upload(db.Model):
    __tablename__ = 'upload'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(256), nullable=False)
    transcript = db.Column(db.Text)
    total_words = db.Column(db.Integer)
    filler_count = db.Column(db.Integer)
    pause_count = db.Column(db.Integer)
    wpm = db.Column(db.Float)
    # Advanced metrics
    pitch_variation_percent = db.Column(db.Integer, nullable=True)
    pitch_std = db.Column(db.Float)
    pitch_mean = db.Column(db.Float)
    volume_mean = db.Column(db.Float)
    volume_std = db.Column(db.Float)
    noise_level = db.Column(db.Float)
    vocab_richness = db.Column(db.Float)
    advanced_vocab_count = db.Column(db.Integer)
    sentence_var = db.Column(db.Float)
    score = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    advanced_words = db.Column(db.Text)  # Store as JSON string
    pitch_mean_hz = db.Column(db.Float, nullable=True)
    pitch_label = db.Column(db.String(64), nullable=True)

# --- End database setup ---

# Create tables on startup
try:
    with app.app_context():
        db.create_all()
    print("Database tables created (or already exist).")
except Exception as e:
    print(f"Error during db.create_all(): {e}")

# Load Whisper model once at startup
from faster_whisper import WhisperModel
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# Cache the common-words set once at startup
COMMON_WORDS_SET = set(nltk.corpus.words.words())

def get_float(segment, key):
    value = segment.get(key, 0)
    try:
        return float(value)
    except Exception:
        return 0.0

def safe_float(val):
    try:
        if val is None:
            return None
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return float(val)
    except Exception:
        return None

def generate_actionable_tips(metrics):
    """
    Given a dict of analysis metrics, return a list of actionable tips for the user.
    """
    tips = []
    wpm = metrics.get('wpm')
    filler_count = metrics.get('filler_count')
    pause_count = metrics.get('pause_count')
    pitch_variation_percent = metrics.get('pitch_variation_percent')
    volume_std = metrics.get('volume_std')
    noise_level = metrics.get('noise_level')
    vocab_richness = metrics.get('vocab_richness')
    advanced_vocab_count = metrics.get('advanced_vocab_count')
    sentence_var = metrics.get('sentence_var')

    # Pace
    if wpm is not None:
        if wpm < 110:
            tips.append("Try to speak a little faster for better engagement.")
        elif wpm > 160:
            tips.append("Try to slow down for better clarity.")
        else:
            tips.append("Your speaking pace is good!")

    # Filler words
    if filler_count is not None:
        if filler_count > 5:
            tips.append("Practice reducing filler words like 'um', 'uh', and 'like'.")
        elif filler_count > 0:
            tips.append("Great job! Try to reduce filler words even further.")
        else:
            tips.append("Excellent! No filler words detected.")

    # Pauses
    if pause_count is not None:
        if pause_count == 0:
            tips.append("Try to add natural pauses to let your audience absorb information.")
        elif pause_count > 5:
            tips.append("Try to reduce long pauses to keep your speech flowing.")
        else:
            tips.append("Your use of pauses is good.")

    # Pitch variation
    if pitch_variation_percent is not None:
        if pitch_variation_percent < 20:
            tips.append("Vary your pitch more to sound more engaging and expressive.")
        elif pitch_variation_percent > 70:
            tips.append("Your pitch variation is excellent!")
        else:
            tips.append("Good pitch variation. Keep it up!")

    # Volume variation
    if volume_std is not None:
        if volume_std < 0.01:
            tips.append("Try to add more vocal energy and variation to your speech.")
        elif volume_std > 0.05:
            tips.append("Your vocal energy is great!")
        else:
            tips.append("Good vocal energy. Keep it up!")

    # Noise level
    if noise_level is not None:
        if noise_level > 0.4:
            tips.append("Try to record in a quieter environment for better clarity.")
        else:
            tips.append("Background noise is at a good level.")

    # Vocabulary richness
    if vocab_richness is not None:
        if vocab_richness < 0.2:
            tips.append("Try to use a wider range of words to make your speech more interesting.")
        elif vocab_richness > 0.5:
            tips.append("Excellent vocabulary usage!")
        else:
            tips.append("Good vocabulary variety. Keep practicing!")

    # Advanced vocabulary
    if advanced_vocab_count is not None:
        if advanced_vocab_count < 3:
            tips.append("Try to use more advanced words to impress your audience.")
        elif advanced_vocab_count > 10:
            tips.append("Great use of advanced vocabulary!")

    # Sentence variety
    if sentence_var is not None:
        if sentence_var < 2:
            tips.append("Try to vary your sentence lengths for a more dynamic speech.")
        elif sentence_var > 5:
            tips.append("Excellent sentence variety!")
        else:
            tips.append("Good sentence variety. Keep it up!")

    if not tips:
        tips.append("Great job on completing your speech! Keep practicing and you'll see even more improvement.")
    return tips


def get_ai_advice(transcript, metrics):
    """
    Call Gemini API to get AI-powered advice for the user's speech.
    Falls back to rule-based advice if the API call fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set — skipping AI advice")
        return _fallback_advice(metrics)

    # Identify weaknesses for the prompt
    weaknesses = []
    if metrics.get('wpm') is not None:
        if metrics['wpm'] < 110:
            weaknesses.append("Speaking pace is too slow.")
        elif metrics['wpm'] > 160:
            weaknesses.append("Speaking pace is too fast.")
    if metrics.get('filler_count', 0) > 5:
        weaknesses.append("Too many filler words.")
    if metrics.get('pause_count', 0) > 5:
        weaknesses.append("Too many long pauses.")
    if metrics.get('pitch_variation_percent') is not None and metrics['pitch_variation_percent'] < 20:
        weaknesses.append("Pitch variation is low.")
    if metrics.get('volume_std') is not None and metrics['volume_std'] < 0.01:
        weaknesses.append("Vocal energy is low.")
    if metrics.get('noise_level') is not None and metrics['noise_level'] > 0.4:
        weaknesses.append("Background noise is high.")
    if metrics.get('vocab_richness') is not None and metrics['vocab_richness'] < 0.2:
        weaknesses.append("Vocabulary range is limited.")
    if metrics.get('advanced_vocab_count') is not None and metrics['advanced_vocab_count'] < 3:
        weaknesses.append("Few advanced words used.")
    if metrics.get('sentence_var') is not None and metrics['sentence_var'] < 2:
        weaknesses.append("Sentence variety is low.")

    weaknesses_str = "\n".join(weaknesses) if weaknesses else "No major weaknesses detected, but improvement is always possible."

    prompt = (
        "You are a public speaking coach. "
        "Given the following speech transcript and analysis metrics, provide at least one personalized, "
        "encouraging, and actionable tip to help the speaker improve. "
        "Be specific and supportive. If weaknesses are listed, focus on them.\n"
        f"Transcript (excerpt): {transcript[:1000]}\n"
        f"Metrics: {metrics}\n"
        f"Weaknesses: {weaknesses_str}\n"
        "Advice (2-3 sentences):"
    )

    # --- FIX: Correct Gemini API format ---
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 150,
            "temperature": 0.7
        }
    }

    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": api_key},
            json=payload,
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            # Parse Gemini response structure
            advice = (
                result
                .get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            if advice:
                return advice
            print("Gemini returned empty advice, using fallback.")
        else:
            print(f"Gemini API error: {response.status_code} — {response.text}")

    except Exception as e:
        print(f"Error calling Gemini API: {e}")

    return _fallback_advice(metrics, weaknesses)


def _fallback_advice(metrics, weaknesses=None):
    """Rule-based fallback when Gemini is unavailable."""
    if weaknesses:
        return "Here are some areas to focus on: " + "; ".join(weaknesses)
    return "Keep practicing! Even small improvements in pace, clarity, and vocabulary will make a big difference over time."


# ---------------------------------------------------------------------------
# Whisper chunk transcription — runs in a thread pool
# ---------------------------------------------------------------------------

def transcribe_chunk(args):
    """Transcribe a single audio chunk. Designed to run in a ThreadPoolExecutor."""
    idx, chunk, tmp_path = args
    chunk_file = f"{tmp_path}_chunk_{idx}.wav"
    try:
        chunk.export(chunk_file, format="wav")
        # faster-whisper returns (segments_generator, info)
        segments_gen, _ = whisper_model.transcribe(chunk_file, beam_size=1, language="en")
        segments = list(segments_gen)
        text = " ".join(seg.text for seg in segments)
        # Convert faster-whisper Segment objects to plain dicts for downstream use
        seg_dicts = [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments]
        return idx, text, seg_dicts
    finally:
        try:
            os.remove(chunk_file)
        except OSError:
            pass


@app.route('/login.html')
def serve_login():
    return app.send_static_file('login.html')

@app.route('/register.html')
def serve_register():
    return app.send_static_file('register.html')

@app.route('/dashboard.html')
def serve_dashboard():
    return app.send_static_file('dashboard.html')

@app.route('/upload.html')
def serve_upload():
    return app.send_static_file('upload.html')

@app.route('/AboutUs.html')
def serve_about():
    return app.send_static_file('AboutUs.html')

@app.route('/', methods=['GET'])
def home():
    return app.send_static_file('login.html')

@app.route('/api/status', methods=['GET'])
@app.route('/status', methods=['GET'])
def status():
    try:
        if whisper_model is not None:
            return jsonify({'model_loaded': True, 'status': 'ready'})
        else:
            return jsonify({'model_loaded': False, 'status': 'loading'})
    except Exception as e:
        return jsonify({'model_loaded': False, 'status': 'error', 'error': str(e)})

@app.route('/transcribe', methods=['GET'])
def transcribe():
    return jsonify({
        "error": "This route is deprecated. Use /transcribe_upload for proper file processing.",
        "message": "Upload files through the frontend to get full analysis and database storage."
    }), 400


# --- Background job tracking ---
import uuid
import threading

JOBS = {}
JOBS_LOCK = threading.Lock()

def set_job_progress(job_id, status, percent, **extra):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update({"status": status, "percent": percent, **extra})

@app.route('/api/job_status/<job_id>', methods=['GET'])
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

@app.route('/api/transcribe_upload', methods=['POST'])
@app.route('/transcribe_upload', methods=['POST'])
def transcribe_upload():
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
    if 'audio-upload' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['audio-upload']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = file.filename.lower()
    orig_suffix = os.path.splitext(filename)[1] or '.bin'
    with tempfile.NamedTemporaryFile(delete=False, suffix=orig_suffix) as tmp_orig:
        file.save(tmp_orig.name)
        orig_path = tmp_orig.name

    job_id = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "percent": 0}

    thread = threading.Thread(
        target=process_audio_job,
        args=(job_id, orig_path, user_id, file.filename),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id}), 202


def process_audio_job(job_id, orig_path, user_id, original_filename):
    with app.app_context():
        try:
            set_job_progress(job_id, "converting", 5)

            # Convert to WAV using pydub/ffmpeg (handles mp3, ogg, m4a, webm, etc.)
            try:
                audio_seg = AudioSegment.from_file(orig_path)
                tmp_path = orig_path + '_converted.wav'
                audio_seg.export(tmp_path, format='wav')
            except Exception as conv_err:
                print(f"Audio conversion failed: {conv_err}")
                set_job_progress(job_id, "error", 0, error="Could not process audio file. Please upload a valid audio file (mp3, wav, ogg, m4a, etc.).")
                return
            finally:
                try:
                    os.remove(orig_path)
                except OSError:
                    pass

            print(f"Processing audio file: {original_filename} -> {tmp_path}")

            # --- FIX: Actually skip pitch analysis for long audio ---
            duration_seconds = len(audio_seg) / 1000
            skip_pitch_analysis = duration_seconds > 600  # skip if >10 minutes

            set_job_progress(job_id, "analyzing_pitch", 20)

            pitch_std = pitch_mean = volume_mean = volume_std = noise_level = None
            pitch_variation_percent = None
            pitch_mean_hz = None
            pitch_label = "Pitch not detected"

            if skip_pitch_analysis:
                print(f"Audio is {duration_seconds:.0f}s — skipping pitch analysis for speed.")
                pitch_label = "Pitch analysis skipped (audio too long)"
            else:
                try:
                    import warnings
                    warnings.filterwarnings('ignore')
                    import parselmouth

                    snd = parselmouth.Sound(tmp_path)
                    Fs = snd.sampling_frequency
                    x = snd.values.T.flatten()

                    if len(x) == 0:
                        raise ValueError("Empty audio file")

                    pitch = snd.to_pitch(time_step=0.01, pitch_floor=50.0, pitch_ceiling=500.0)
                    pitch_values = pitch.selected_array['frequency']
                    valid_pitch_values = pitch_values[
                        (pitch_values > 0) &
                        (pitch_values >= 50) &
                        (pitch_values <= 500) &
                        (~np.isnan(pitch_values)) &
                        (~np.isinf(pitch_values))
                    ]

                    print(f"Total pitch frames: {len(pitch_values)}")
                    print(f"Valid pitch frames: {len(valid_pitch_values)}")

                    if len(valid_pitch_values) > 5:
                        Q1 = np.percentile(valid_pitch_values, 10)
                        Q3 = np.percentile(valid_pitch_values, 90)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 2.0 * IQR
                        upper_bound = Q3 + 2.0 * IQR
                        filtered_pitch = valid_pitch_values[
                            (valid_pitch_values >= lower_bound) & (valid_pitch_values <= upper_bound)
                        ]
                        if len(filtered_pitch) < len(valid_pitch_values) * 0.2:
                            low = np.percentile(valid_pitch_values, 5)
                            high = np.percentile(valid_pitch_values, 95)
                            filtered_pitch = valid_pitch_values[
                                (valid_pitch_values >= low) & (valid_pitch_values <= high)
                            ]
                        if len(filtered_pitch) > 0:
                            pitch_mean = float(np.mean(filtered_pitch))
                            pitch_std = float(np.std(filtered_pitch))
                            pitch_mean_hz = round(pitch_mean)
                            min_std = 2.0
                            max_std = 20.0
                            variation = np.clip((pitch_std - min_std) / (max_std - min_std), 0, 1)
                            pitch_variation_percent = round(variation * 100)
                            if pitch_mean < 85:
                                pitch_label = "Very deep voice"
                            elif pitch_mean < 110:
                                pitch_label = "Deep voice"
                            elif pitch_mean < 140:
                                pitch_label = "Low-moderate voice"
                            elif pitch_mean < 180:
                                pitch_label = "Moderate voice"
                            elif pitch_mean < 220:
                                pitch_label = "Higher voice"
                            elif pitch_mean < 300:
                                pitch_label = "High voice"
                            else:
                                pitch_label = "Very high voice"
                            print(f"Pitch: mean={pitch_mean:.2f}Hz, std={pitch_std:.2f}, variation={pitch_variation_percent}%, label={pitch_label}")
                        else:
                            pitch_std = pitch_mean = 0.0
                            pitch_variation_percent = 0
                            pitch_mean_hz = 0
                    else:
                        print(f"Insufficient voiced segments: only {len(valid_pitch_values)} frames")
                        pitch_std = pitch_mean = 0.0
                        pitch_variation_percent = 0
                        pitch_mean_hz = 0

                    # Volume / noise analysis
                    frame_length = int(0.050 * Fs)
                    hop_length = int(0.025 * Fs)
                    if len(x) >= frame_length:
                        frames = np.lib.stride_tricks.sliding_window_view(x, frame_length)[::hop_length]
                        rms = np.sqrt(np.mean(frames**2, axis=1))
                        valid_rms = rms[rms > 1e-10]
                        if len(valid_rms) > 0:
                            volume_mean = float(np.mean(valid_rms))
                            volume_std = float(np.std(valid_rms))
                            threshold = 0.1 * np.mean(valid_rms)
                            noise_level = float(np.sum(valid_rms < threshold) / len(valid_rms))
                        else:
                            volume_mean = volume_std = noise_level = None
                    else:
                        volume_mean = volume_std = noise_level = None

                    try:
                        del snd, x
                    except NameError:
                        pass

                except ImportError:
                    print("parselmouth not installed.")
                    set_job_progress(job_id, "error", 0, error="Speech analysis library not available. Please contact support.")
                    return
                except Exception as audio_err:
                    import traceback
                    traceback.print_exc()
                    pitch_std = pitch_mean = volume_mean = volume_std = noise_level = None
                    pitch_variation_percent = None
                    pitch_mean_hz = None
                    pitch_label = "Pitch not detected"

            gc.collect()
            set_job_progress(job_id, "transcribing", 45)

            # --- FIX: Parallel chunk transcription ---
            audio = AudioSegment.from_wav(tmp_path)
            chunk_ms = 10 * 60 * 1000  # 10-minute chunks (reduced I/O vs 5-min)
            chunks = [audio[i:i + chunk_ms] for i in range(0, len(audio), chunk_ms)]

            chunks_args = [(idx, chunk, tmp_path) for idx, chunk in enumerate(chunks)]
            results = {}

            # Use up to 3 parallel workers; more than that risks memory pressure on small VMs
            max_workers = min(3, len(chunks))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(transcribe_chunk, arg): arg[0] for arg in chunks_args}
                for future in as_completed(futures):
                    idx, text, segs = future.result()
                    results[idx] = (text, segs)
                    # Update progress proportionally between 45% → 75%
                    pct = 45 + int((len(results) / len(chunks)) * 30)
                    set_job_progress(job_id, "transcribing", pct)

            # Reassemble in chunk order
            full_text = " ".join(results[i][0] for i in sorted(results))
            all_segments = [seg for i in sorted(results) for seg in results[i][1]]

            transcript = full_text.strip()
            segments = all_segments

            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            gc.collect()

            set_job_progress(job_id, "scoring", 80)

            # --- Text analysis ---
            filler_words = ["um", "uh", "like", "you know"]
            words = nltk.word_tokenize(transcript.lower())
            total_words = len(words)
            filler_count = sum(words.count(filler) for filler in filler_words)

            pause_count = 0
            for i in range(1, len(segments)):
                gap = float(segments[i].get('end', 0)) - float(segments[i - 1].get('end', 0))
                if gap > 2:
                    pause_count += 1

            duration = float(segments[-1].get('end', 1)) if segments else 1.0
            wpm = (total_words / duration) * 60 if duration > 0 else 0

            vocab_richness = advanced_vocab_count = sentence_var = None
            advanced_words = []
            try:
                unique_words = set(words)
                vocab_richness = len(unique_words) / total_words if total_words > 0 else None
                advanced_words = [w for w in unique_words if w.isalpha() and w not in COMMON_WORDS_SET]
                advanced_vocab_count = len(advanced_words)
                sentences = nltk.sent_tokenize(transcript)
                sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
                sentence_var = float(np.std(sentence_lengths)) if len(sentence_lengths) > 1 else None
            except Exception as text_err:
                print(f"Transcript analysis error: {text_err}")

            # --- Scoring ---
            score = 0
            try:
                if 110 <= wpm <= 160:
                    score += 15
                elif 90 <= wpm < 110 or 160 < wpm <= 180:
                    score += 10
                elif 70 <= wpm < 90 or 180 < wpm <= 200:
                    score += 5
                if filler_count == 0:
                    score += 15
                elif filler_count <= 2:
                    score += 10
                elif filler_count <= 5:
                    score += 5
                if pause_count <= 2:
                    score += 10
                elif pause_count <= 5:
                    score += 5
                if pitch_variation_percent is not None:
                    score += round(pitch_variation_percent / 10)
                if volume_std is not None:
                    if volume_std < 0.01:
                        score += 10
                    elif volume_std < 0.03:
                        score += 7
                    elif volume_std < 0.05:
                        score += 4
                if noise_level is not None:
                    if noise_level < 0.2:
                        score += 10
                    elif noise_level < 0.4:
                        score += 7
                    elif noise_level < 0.6:
                        score += 4
                if vocab_richness is not None:
                    if vocab_richness > 0.5:
                        score += 10
                    elif vocab_richness > 0.3:
                        score += 7
                    elif vocab_richness > 0.15:
                        score += 4
                if advanced_vocab_count is not None:
                    if advanced_vocab_count > 10:
                        score += 5
                    elif advanced_vocab_count > 5:
                        score += 3
                if sentence_var is not None:
                    if sentence_var > 5:
                        score += 5
                    elif sentence_var > 2:
                        score += 3
            except Exception as score_err:
                print(f"Score calculation error: {score_err}")
                score = 0

            print(
                f"Analysis complete: {total_words} words, {filler_count} fillers, "
                f"{pause_count} pauses, {wpm:.2f} wpm, pitch_std={pitch_std}, "
                f"vol_std={volume_std}, noise={noise_level}, vocab={vocab_richness}, "
                f"adv_vocab={advanced_vocab_count}, sent_var={sentence_var}, score={score}"
            )

            # --- Save to DB ---
            upload = Upload(
                user_id=user_id,
                filename=original_filename,
                transcript=transcript,
                total_words=total_words,
                filler_count=filler_count,
                pause_count=pause_count,
                wpm=round(wpm, 2),
                pitch_std=pitch_std,
                pitch_mean=pitch_mean,
                pitch_mean_hz=pitch_mean_hz,
                pitch_label=pitch_label,
                pitch_variation_percent=pitch_variation_percent,
                volume_mean=volume_mean,
                volume_std=volume_std,
                noise_level=noise_level,
                vocab_richness=vocab_richness,
                advanced_vocab_count=advanced_vocab_count,
                sentence_var=sentence_var,
                score=score,
                advanced_words=json.dumps(advanced_words)
            )
            db.session.add(upload)
            db.session.commit()

            if pitch_mean_hz is not None and pitch_mean_hz > 0:
                pitch_explanation = f"{pitch_mean_hz} Hz — {pitch_label}"
            else:
                pitch_explanation = "N/A — Pitch not detected (audio may be too quiet or unclear)"

            set_job_progress(job_id, "generating_tips", 92)

            analysis_metrics = {
                "wpm": safe_float(wpm),
                "filler_count": filler_count,
                "pause_count": pause_count,
                "pitch_variation_percent": safe_float(pitch_variation_percent),
                "volume_std": safe_float(volume_std),
                "noise_level": safe_float(noise_level),
                "vocab_richness": safe_float(vocab_richness),
                "advanced_vocab_count": advanced_vocab_count,
                "sentence_var": safe_float(sentence_var)
            }
            actionable_tips = generate_actionable_tips(analysis_metrics)
            ai_advice = get_ai_advice(transcript, analysis_metrics)

            result_payload = {
                "transcript": transcript,
                "total_words": total_words,
                "filler_count": filler_count,
                "pause_count": pause_count,
                "wpm": safe_float(wpm),
                "pitch_std": safe_float(pitch_std),
                "pitch_mean": safe_float(pitch_mean),
                "pitch_mean_hz": pitch_mean_hz,
                "pitch_mean_explained": pitch_explanation,
                "pitch_variation_percent": safe_float(pitch_variation_percent),
                "volume_mean": safe_float(volume_mean),
                "volume_std": safe_float(volume_std),
                "noise_level": safe_float(noise_level),
                "vocab_richness": safe_float(vocab_richness),
                "advanced_vocab_count": advanced_vocab_count,
                "advanced_words": advanced_words,
                "sentence_var": safe_float(sentence_var),
                "score": score,
                "actionable_tips": actionable_tips,
                "ai_advice": ai_advice
            }
            set_job_progress(job_id, "done", 100, result=result_payload)

        except Exception as e:
            print(f"Error processing upload: {str(e)}")
            import traceback
            traceback.print_exc()
            set_job_progress(job_id, "error", 0, error="Audio processing failed. Please try a different or clearer audio file.")


@app.route('/api/uploads/<int:user_id>', methods=['GET'])
@app.route('/uploads/<int:user_id>', methods=['GET'])
def get_uploads(user_id):
    uploads = Upload.query.filter_by(user_id=user_id).order_by(Upload.timestamp.desc()).all()
    return jsonify([
        {
            'id': u.id,
            'filename': u.filename,
            'transcript': u.transcript,
            'total_words': u.total_words,
            'filler_count': u.filler_count,
            'pause_count': u.pause_count,
            'wpm': u.wpm,
            'pitch_std': u.pitch_std,
            'pitch_mean': u.pitch_mean,
            'pitch_mean_hz': u.pitch_mean_hz,
            'pitch_label': u.pitch_label,
            'pitch_variation_percent': u.pitch_variation_percent,
            'pitch_mean_explained': (
                f"{u.pitch_mean_hz} Hz — {u.pitch_label}"
                if u.pitch_mean_hz and u.pitch_mean_hz > 0
                else "N/A — Pitch not detected"
            ),
            'volume_mean': u.volume_mean,
            'volume_std': u.volume_std,
            'noise_level': u.noise_level,
            'vocab_richness': u.vocab_richness,
            'advanced_vocab_count': u.advanced_vocab_count,
            'sentence_var': u.sentence_var,
            'score': u.score,
            'timestamp': u.timestamp.isoformat() if u.timestamp else None,
            'advanced_words': json.loads(u.advanced_words) if u.advanced_words else []
        } for u in uploads
    ])


@app.route('/api/profile/<int:user_id>', methods=['GET'])
@app.route('/profile/<int:user_id>', methods=['GET'])
def get_profile(user_id):
    uploads = Upload.query.filter_by(user_id=user_id).all()
    total_uploads = len(uploads)
    best_wpm = max((u.wpm for u in uploads if u.wpm is not None), default=0)
    lowest_filler = min((u.filler_count for u in uploads if u.filler_count is not None), default=0)
    longest_speech = max((u.total_words for u in uploads if u.total_words is not None), default=0)
    return jsonify({
        'total_uploads': total_uploads,
        'best_wpm': best_wpm,
        'lowest_filler': lowest_filler,
        'longest_speech': longest_speech
    })


@app.route('/api/register', methods=['POST'])
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    if not username or not email or not password:
        return jsonify({'error': 'Missing required fields'}), 400
    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({'error': 'Username or email already exists'}), 400
    pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    user = User(username=username, email=email, password_hash=pw_hash)
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully'})


@app.route('/api/login', methods=['POST'])
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username_or_email = data.get('username') or data.get('email')
    password = data.get('password')
    if not username_or_email or not password:
        return jsonify({'error': 'Missing required fields'}), 400
    user = User.query.filter(
        (User.username == username_or_email) | (User.email == username_or_email)
    ).first()
    if not user or not bcrypt.check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid username/email or password'}), 401
    return jsonify({
        'message': 'Login successful',
        'user': {'id': user.id, 'username': user.username, 'email': user.email}
    })


if __name__ == '__main__':
    print("Flask app loaded")
    print("Starting Flask server...")
    print("Model loading...")
    print("Server ready!")
    print(f"Current working directory: {os.getcwd()}")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
