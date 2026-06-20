import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import whisper
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
# Render gives postgres:// but SQLAlchemy needs postgresql://
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

# Create tables on startup (runs whether started via gunicorn or `python api.py`)
try:
    with app.app_context():
        db.create_all()
    print("Database tables created (or already exist).")
except Exception as e:
    print(f"Error during db.create_all(): {e}")

model = whisper.load_model("tiny")  # Load once at startup

# Cache the common-words set once at startup instead of rebuilding it
# (and re-slicing 235k+ words) on every single upload request.
COMMON_WORDS_SET = set(nltk.corpus.words.words()[:2000])

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

    # Ensure at least one tip is always returned
    if not tips:
        tips.append("Great job on completing your speech! Keep practicing and you'll see even more improvement.")
    return tips

def get_ai_advice(transcript, metrics):
    """
    Call Hugging Face Inference API to get bonus AI-powered advice for the user's speech.
    Always provide at least some advice, even if the AI returns nothing.
    """
    api_key = os.environ.get('HF_API_KEY')
    if not api_key:
        return None
    model = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"


    # Identify weaknesses for the prompt
    weaknesses = []
    if metrics.get('wpm') is not None:
        if metrics['wpm'] < 110:
            weaknesses.append("Your speaking pace is too slow.")
        elif metrics['wpm'] > 160:
            weaknesses.append("Your speaking pace is too fast.")
    if metrics.get('filler_count', 0) > 5:
        weaknesses.append("You use too many filler words.")
    if metrics.get('pause_count', 0) > 5:
        weaknesses.append("You have too many long pauses.")
    if metrics.get('pitch_variation_percent') is not None and metrics['pitch_variation_percent'] < 20:
        weaknesses.append("Your pitch variation is low.")
    if metrics.get('volume_std') is not None and metrics['volume_std'] < 0.01:
        weaknesses.append("Your vocal energy is low.")
    if metrics.get('noise_level') is not None and metrics['noise_level'] > 0.4:
        weaknesses.append("Your background noise is high.")
    if metrics.get('vocab_richness') is not None and metrics['vocab_richness'] < 0.2:
        weaknesses.append("Your vocabulary range is limited.")
    if metrics.get('advanced_vocab_count') is not None and metrics['advanced_vocab_count'] < 3:
        weaknesses.append("You use few advanced words.")
    if metrics.get('sentence_var') is not None and metrics['sentence_var'] < 2:
        weaknesses.append("Your sentence variety is low.")

    weaknesses_str = "\n".join(weaknesses) if weaknesses else "No major weaknesses detected, but improvement is always possible."

    prompt = (
        "You are a public speaking coach. "
        "Given the following speech transcript and analysis metrics, provide at least one personalized, encouraging, and actionable tip to help the speaker improve. "
        "Be specific and supportive. If weaknesses are listed, focus on them.\n"
        f"Transcript: {transcript}\n"
        f"Metrics: {metrics}\n"
        f"Weaknesses: {weaknesses_str}\n"
        "Advice:"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 80}
    }
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json=payload,
            timeout=15
        )
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                advice = result[0]['generated_text'].strip()
            elif isinstance(result, dict) and 'generated_text' in result:
                advice = result['generated_text'].strip()
            elif isinstance(result, list) and len(result) > 0 and 'generated_text' in result[-1]:
                advice = result[-1]['generated_text'].strip()
            elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], str):
                advice = result[0].strip()
            else:
                advice = None
            if advice:
                return advice
        else:
            print(f"Hugging Face API error: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Error calling Hugging Face API: {e}")
    # Fallback: rule-based advice if AI fails
    if weaknesses:
        return "Here are some areas to focus on: " + "; ".join(weaknesses)
    return "Keep practicing! Even if you did well, there's always room for improvement."

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
        # Check if model is loaded
        if model is not None:
            return jsonify({'model_loaded': True, 'status': 'ready'})
        else:
            return jsonify({'model_loaded': False, 'status': 'loading'})
    except Exception as e:
        return jsonify({'model_loaded': False, 'status': 'error', 'error': str(e)})

@app.route('/transcribe', methods=['GET'])
def transcribe():
    """
    Test route for transcription - this route is deprecated.
    Use /transcribe_upload for proper file processing and database storage.
    """
    return jsonify({
        "error": "This route is deprecated. Use /transcribe_upload for proper file processing.",
        "message": "Upload files through the frontend to get full analysis and database storage."
    }), 400

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

    try:
        # Save the uploaded file with its original extension first
        filename = file.filename.lower()
        orig_suffix = os.path.splitext(filename)[1] or '.bin'
        with tempfile.NamedTemporaryFile(delete=False, suffix=orig_suffix) as tmp_orig:
            file.save(tmp_orig.name)
            orig_path = tmp_orig.name

        # Convert to a real WAV file using pydub/ffmpeg, regardless of input format
        # (browsers/devices send mp3, ogg/opus, m4a, aac, wma, webm, etc. — pydub
        # auto-detects the real format from file content via ffmpeg, not extension)
        try:
            audio_seg = AudioSegment.from_file(orig_path)
            tmp_path = orig_path + '_converted.wav'
            audio_seg.export(tmp_path, format='wav')
        except Exception as conv_err:
            print(f"Audio conversion failed: {conv_err}")
            return jsonify({'error': 'Could not process audio file. Please upload a valid audio file (mp3, wav, ogg, m4a, etc.).'}), 400
        finally:
            # Clean up the original upload now that we have the converted version
            try:
                os.remove(orig_path)
            except OSError:
                pass

        print(f"Processing audio file: {file.filename} -> {tmp_path}")
        # --- Advanced Audio Metrics ---
        pitch_std = pitch_mean = volume_mean = volume_std = noise_level = None
        pitch_variation_percent = None
        pitch_mean_hz = None
        pitch_label = "Pitch not detected"
        try:
            import warnings
            warnings.filterwarnings('ignore')
            import parselmouth
            snd = parselmouth.Sound(tmp_path)
            Fs = snd.sampling_frequency
            x = snd.values.T.flatten()
            if len(x) == 0:
                print("Audio file is empty")
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
                filtered_pitch = valid_pitch_values[(valid_pitch_values >= lower_bound) & (valid_pitch_values <= upper_bound)]
                if len(filtered_pitch) < len(valid_pitch_values) * 0.2:
                    low = np.percentile(valid_pitch_values, 5)
                    high = np.percentile(valid_pitch_values, 95)
                    filtered_pitch = valid_pitch_values[(valid_pitch_values >= low) & (valid_pitch_values <= high)]
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
                    print(f"Pitch analysis successful: mean={pitch_mean:.2f}Hz, std={pitch_std:.2f}, variation={pitch_variation_percent}%, label={pitch_label}")
                else:
                    print("No valid pitch values after filtering")
                    pitch_std = 0.0
                    pitch_mean = 0.0
                    pitch_variation_percent = 0
                    pitch_mean_hz = 0
            else:
                print(f"Insufficient voiced segments: only {len(valid_pitch_values)} frames")
                pitch_std = 0.0
                pitch_mean = 0.0
                pitch_variation_percent = 0
                pitch_mean_hz = 0
            frame_length = int(0.050 * Fs)
            hop_length = int(0.025 * Fs)
            if len(x) >= frame_length:
                frames = np.lib.stride_tricks.sliding_window_view(x, frame_length)[::hop_length]
                rms = np.sqrt(np.mean(frames**2, axis=1))
                valid_rms = rms[rms > 1e-10]
                if len(valid_rms) > 0:
                    volume_mean = float(np.mean(valid_rms))
                    volume_std = float(np.std(valid_rms))
                    mean_rms = np.mean(valid_rms)
                    threshold = 0.1 * mean_rms
                    low_energy_ratio = np.sum(valid_rms < threshold) / len(valid_rms)
                    noise_level = float(low_energy_ratio)
                else:
                    volume_mean = volume_std = noise_level = None
            else:
                volume_mean = volume_std = noise_level = None
        except ImportError:
            print("parselmouth not installed. Please install with: pip install praat-parselmouth")
            return jsonify({"error": "Speech analysis library not available. Please contact support."}), 500
        except Exception as audio_err:
            import traceback
            traceback.print_exc()
            pitch_std = pitch_mean = volume_mean = volume_std = noise_level = None
            pitch_variation_percent = None
            pitch_mean_hz = None
            pitch_label = "Pitch not detected"
        # Free parselmouth/numpy memory before the heavier Whisper transcription step
        try:
            del snd, x
        except NameError:
            pass
        gc.collect()

        # --- Whisper Transcription ---
        result = model.transcribe(tmp_path, fp16=False)
        transcript = result['text']
        if isinstance(transcript, list):
            transcript = " ".join(transcript)
        segments = result['segments']
        del result
        # Clean up the temp WAV file now — nothing downstream needs the audio anymore
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        gc.collect()
        filler_words = ["um", "uh", "like", "you know"]
        words = nltk.word_tokenize(transcript.lower())
        total_words = len(words)
        filler_count = sum(words.count(filler) for filler in filler_words)
        pause_count = 0
        for i in range(1, len(segments)):
            gap = get_float(segments[i], 'start') - get_float(segments[i-1], 'end')
            if gap > 2:
                pause_count += 1
        duration = get_float(segments[-1], 'end') if segments else 1.0
        wpm = (total_words / duration) * 60 if duration > 0 else 0
        vocab_richness = advanced_vocab_count = sentence_var = None
        try:
            unique_words = set(words)
            vocab_richness = len(unique_words) / total_words if total_words > 0 else None
            common_words = COMMON_WORDS_SET
            advanced_words = [w for w in unique_words if w.isalpha() and w not in common_words]
            advanced_vocab_count = len(advanced_words)
            sentences = nltk.sent_tokenize(transcript)
            sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
            sentence_var = float(np.std(sentence_lengths)) if len(sentence_lengths) > 1 else None
        except Exception as text_err:
            print(f"Transcript analysis error: {text_err}")
            vocab_richness = advanced_vocab_count = sentence_var = None
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
        print(f"Analysis complete: {total_words} words, {filler_count} fillers, {pause_count} pauses, {wpm:.2f} wpm, pitch std: {pitch_std}, volume std: {volume_std}, noise: {noise_level}, vocab richness: {vocab_richness}, advanced vocab: {advanced_vocab_count}, sentence var: {sentence_var}, score: {score}")
        # Find this section in your transcribe_upload function and replace it:

        upload = Upload(
            user_id=user_id,
            filename=file.filename,
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

        # Generate actionable tips
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

        # Get AI-powered advice (bonus)
        ai_advice = get_ai_advice(transcript, analysis_metrics)

        return jsonify({
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
        })
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        return jsonify({"error": "Audio processing failed. Please try a different or clearer audio file."}), 500

# Find your get_uploads function and replace the return statement:

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
            'pitch_mean_hz': u.pitch_mean_hz,  # ADD THIS LINE
            'pitch_label': u.pitch_label,      # ADD THIS LINE
            'pitch_variation_percent': u.pitch_variation_percent,  # ADD THIS LINE
            'pitch_mean_explained': f"{u.pitch_mean_hz} Hz — {u.pitch_label}" if u.pitch_mean_hz and u.pitch_mean_hz > 0 else "N/A — Pitch not detected",  # ADD THIS LINE
            'volume_mean': u.volume_mean,
            'volume_std': u.volume_std,
            'noise_level': u.noise_level,
            'vocab_richness': u.vocab_richness,
            'advanced_vocab_count': u.advanced_vocab_count,
            'sentence_var': u.sentence_var,
            'score': u.score,
            'timestamp': u.timestamp.isoformat() if u.timestamp else None,
            'advanced_words': json.loads(u.advanced_words) if u.advanced_words else []  # ADD THIS LINE
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
    user = User.query.filter((User.username == username_or_email) | (User.email == username_or_email)).first()
    if not user or not bcrypt.check_password_hash(user.password_hash, password):
        return jsonify({'error': 'Invalid username/email or password'}), 401
    return jsonify({'message': 'Login successful', 'user': {'id': user.id, 'username': user.username, 'email': user.email}})

if __name__ == '__main__':
    print("Flask app loaded")
    print("Starting Flask server...")
    print("Model loading...")
    print("Server ready!")
    print(f"Current working directory: {os.getcwd()}")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
