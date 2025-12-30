# Speaker Recognition for Home Assistant

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Home Assistant](https://img.shields.io/badge/Home%20Assistant-integration-blue)

**Identify speakers by their voice using machine learning.** This project provides a complete speaker recognition solution for Home Assistant, including a REST API service, Python client library, custom integration, and Home Assistant addon.

## ✨ Features

- 🎤 **Voice-based speaker identification** using neural embeddings
- 🏠 **Native Home Assistant integration** with STT and conversation agents
- 🐳 **Easy deployment** via Home Assistant addon or standalone Docker
- 🔌 **REST API** for flexible integration with any platform
- 📦 **Python client library** for programmatic access
- 🎯 **High accuracy** powered by Resemblyzer voice embeddings
- ⚡ **Fast recognition** with cached embeddings
- 🔧 **Configurable** via UI or YAML

## 📋 Table of Contents

- [Installation](#installation)
  - [Home Assistant Addon](#home-assistant-addon)
  - [Python Package](#python-package)
  - [Docker](#docker)
- [Usage](#usage)
  - [Training](#training)
  - [Recognition](#recognition)
  - [Home Assistant Integration](#home-assistant-integration)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Home Assistant Addon

The easiest way to use speaker recognition in Home Assistant:

1. Add this repository to your Home Assistant addon store
2. Install the **Speaker Recognition** addon
3. Configure the addon settings:
   - **Host**: `0.0.0.0` (default)
   - **Port**: `8099` (default)
   - **Embeddings Directory**: `/share/speaker_recognition/embeddings`
   - **Log Level**: `info`
4. Start the addon
5. Install the **Speaker Recognition** integration via the UI

### Python Package

Install the client-only package (no ML dependencies):

```bash
pip install speaker-recognition
```

Install with server capabilities (requires Python <3.10):

```bash
pip install speaker-recognition[server]
```

### Docker

Run the standalone service:

```bash
docker run -d \
  -p 8099:8099 \
  -v ./embeddings:/app/embeddings \
  ghcr.io/eulemitkeule/speaker-recognition:latest
```

## 📖 Usage

### Training

Train the system with voice samples for each speaker:

#### Using Python Client

```python
from speaker_recognition import SpeakerRecognitionClient
from speaker_recognition.models import TrainingRequest, VoiceSample, AudioInput

async with SpeakerRecognitionClient("http://localhost:8099") as client:
    training = await client.train(
        TrainingRequest(
            voice_samples=[
                VoiceSample(
                    user="Alice",
                    audio_input=AudioInput(
                        audio_data="<base64-encoded-audio>",
                        sample_rate=16000
                    )
                ),
                VoiceSample(
                    user="Bob",
                    audio_input=AudioInput(
                        audio_data="<base64-encoded-audio>",
                        sample_rate=16000
                    )
                )
            ]
        )
    )
    print(f"Trained {training.speakers_count} speakers")
```

#### Using REST API

```bash
curl -X POST http://localhost:8099/train \
  -H "Content-Type: application/json" \
  -d '{
    "voice_samples": [
      {
        "user": "Alice",
        "audio_input": {
          "audio_data": "<base64-audio>",
          "sample_rate": 16000
        }
      }
    ]
  }'
```

### Recognition

Identify a speaker from audio:

#### Using Python Client

```python
from speaker_recognition import SpeakerRecognitionClient
from speaker_recognition.models import RecognitionRequest, AudioInput

async with SpeakerRecognitionClient("http://localhost:8099") as client:
    result = await client.recognize(
        RecognitionRequest(
            audio_input=AudioInput(
                audio_data="<base64-encoded-audio>",
                sample_rate=16000
            )
        )
    )
    print(f"Speaker: {result.speaker} (confidence: {result.confidence:.2%})")
```

### Home Assistant Integration

Once the integration is configured:

1. **Configure the backend** in the main integration entry
2. **Map voices to users** in the integration settings
3. **Add STT entity** as a sub-entry for speech-to-text with speaker ID
4. **Add Conversation Agent** as a sub-entry for voice commands with speaker context

The integration will automatically identify speakers and make the information available to your automations.

## 🔌 API Documentation

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

#### `POST /train`
Train the model with voice samples.

**Request:**
```json
{
  "voice_samples": [
    {
      "user": "string",
      "audio_input": {
        "audio_data": "base64-string",
        "sample_rate": 16000
      }
    }
  ]
}
```

**Response:**
```json
{
  "speakers_count": 2,
  "message": "Training completed successfully"
}
```

#### `POST /recognize`
Recognize a speaker from audio.

**Request:**
```json
{
  "audio_input": {
    "audio_data": "base64-string",
    "sample_rate": 16000
  }
}
```

**Response:**
```json
{
  "speaker": "Alice",
  "confidence": 0.95
}
```

## ⚙️ Configuration

### Addon Configuration

```yaml
host: "0.0.0.0"
port: 8099
log_level: "info"
access_log: true
embeddings_dir: "/share/speaker_recognition/embeddings"
```

### Environment Variables

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8099`)
- `LOG_LEVEL`: Logging level (default: `info`)
- `ACCESS_LOG`: Enable access logs (default: `true`)
- `EMBEDDINGS_DIR`: Directory for storing embeddings (default: `./embeddings`)

## 🛠️ Development

### Prerequisites

- Python 3.9 (for server development)
- Python 3.8+ (for client-only development)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/eulemitkeule/speaker-recognition.git
cd speaker-recognition

# Install dependencies
uv sync --all-groups

# Run tests
uv run pytest tests/ -v

# Run linting
uv run ruff check .

# Run type checking
uv run mypy --strict speaker_recognition
```

### Running Locally

```bash
# Start the server
uv run python -m speaker_recognition

# Or with custom options
uv run python -m speaker_recognition --host 0.0.0.0 --port 8099
```

### Project Structure

```
speaker-recognition/
├── speaker_recognition/          # Main package
│   ├── api.py                   # FastAPI application
│   ├── client.py                # HTTP client
│   ├── models.py                # Pydantic models
│   └── recognizer.py            # Recognition logic
├── custom_components/           # Home Assistant integration
│   └── speaker_recognition/
├── speaker_recognition_addon/   # Home Assistant addon
├── tests/                       # Test suite
└── example_data/               # Example audio files
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Quality

- Follow PEP 8 style guidelines
- Use descriptive variable and function names
- Add type annotations
- Write tests for new features
- Keep methods focused and concise

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) - Neural voice embeddings
- [Home Assistant](https://www.home-assistant.io/) - Home automation platform
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework

## 📞 Support

- 🐛 [Report bugs](https://github.com/eulemitkeule/speaker-recognition/issues)
- 💡 [Request features](https://github.com/eulemitkeule/speaker-recognition/issues)
- 📖 [Documentation](https://github.com/eulemitkeule/speaker-recognition)

---

Made with ❤️ for the Home Assistant community
