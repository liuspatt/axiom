# AxiomAI

[![Hex.pm](https://img.shields.io/hexpm/v/axiom_ai.svg)](https://hex.pm/packages/axiom_ai)
[![Documentation](https://img.shields.io/badge/docs-hexdocs-blue.svg)](https://hexdocs.pm/axiom_ai/)

A unified Elixir client for multiple AI providers.

## Installation

```elixir
def deps do
  [
    {:axiom_ai, "~> 0.1.0"}
  ]
end
```

## Quick Start

```elixir
# Vertex AI (Google Cloud)
client = AxiomAi.new(:vertex_ai, %{project_id: "your-gcp-project"})
{:ok, response} = AxiomAi.chat(client, "Hello!")

# OpenAI
client = AxiomAi.new(:openai, %{api_key: "your-openai-key"})
{:ok, response} = AxiomAi.chat(client, "Hello!")

# Anthropic Claude
client = AxiomAi.new(:anthropic, %{api_key: "your-anthropic-key"})
{:ok, response} = AxiomAi.chat(client, "Hello!")

# Local AI models
client = AxiomAi.new(:local, %{predefined_model: "qwen2.5-0.5b"})
{:ok, response} = AxiomAi.chat(client, "Hello!")

# Whisper speech-to-text
client = AxiomAi.new(:local, %{predefined_model: "whisper-large-v3-turbo"})
{:ok, response} = AxiomAi.chat(client, "/path/to/audio.wav|Transcribe this audio")
```

## Supported Providers

| Provider | Authentication | Example |
|----------|---------------|---------|
| **Vertex AI** | Service Account, ADC | `AxiomAi.new(:vertex_ai, %{project_id: "my-project"})` |
| **OpenAI** | API Key | `AxiomAi.new(:openai, %{api_key: "sk-..."})` |
| **Anthropic** | API Key | `AxiomAi.new(:anthropic, %{api_key: "sk-..."})` |
| **DeepSeek** | API Key | `AxiomAi.new(:deepseek, %{api_key: "sk-..."})` |
| **AWS Bedrock** | AWS Credentials | `AxiomAi.new(:bedrock, %{model: "anthropic.claude-3-haiku-20240307-v1:0"})` |
| **Local AI** | None | `AxiomAi.new(:local, %{predefined_model: "qwen2.5-0.5b"})` |

## Predefined Local Models

```elixir
# List available models
models = AxiomAi.LocalModels.list_models()
IO.inspect(models)
```

**Available Models:**

**Text Generation:**
- `qwen2.5-0.5b` - Qwen2.5 0.5B
- `qwen2.5-1.5b` - Qwen2.5 1.5B  
- `qwen2.5-3b` - Qwen2.5 3B
- `llama3-8b` - Llama 3 8B
- `mistral-7b` - Mistral 7B
- `codellama-7b` - Code Llama 7B

**Speech-to-Text:**
- `whisper-large-v3` - Whisper Large v3
- `whisper-large-v3-turbo` - Whisper Large v3 Turbo

**OCR:**
- `nanonets-ocr-s` - Nanonets OCR Small

## Custom Local Providers

AxiomAI's local provider supports multiple execution methods beyond predefined models, allowing you to run custom AI models and HTTP endpoints.

### 1. HTTP Endpoints (OpenAI-Compatible APIs)

Connect to any OpenAI-compatible API server like Ollama, vLLM, or custom model servers:

```elixir
# OpenAI-compatible API (vLLM, FastAPI, etc.)
client = AxiomAi.new(:local, %{
  endpoint: "http://localhost:8000",
  api_format: :openai,              # :openai or :ollama
  model: "meta-llama/Llama-2-7b-hf",
  api_key: "optional-api-key",      # if required by your server
  temperature: 0.7,
  max_tokens: 1024
})

{:ok, response} = AxiomAi.chat(client, "Hello, how are you?")

# Ollama server
ollama_client = AxiomAi.new(:local, %{
  endpoint: "http://localhost:11434",
  api_format: :ollama,
  model: "llama2:7b",
  temperature: 0.8,
  max_tokens: 2048
})

{:ok, response} = AxiomAi.chat(ollama_client, "Explain quantum computing")
```

### 2. Direct Python Integration

Run AI models directly in Python using either script files or embedded code:

#### Option A: Python Script Files

```elixir
# Using external Python script
client = AxiomAi.new(:local, %{
  python_script: """
  import torch
  from transformers import AutoTokenizer, AutoModelForCausalLM
  import json
  import sys
  
  def main():
      input_data = json.loads(sys.argv[1])
      model_path = input_data['model_path']
      prompt = input_data['prompt']
      max_tokens = input_data.get('max_tokens', 1024)
      temperature = input_data.get('temperature', 0.7)
      
      # Load model and tokenizer
      tokenizer = AutoTokenizer.from_pretrained(model_path)
      model = AutoModelForCausalLM.from_pretrained(
          model_path,
          torch_dtype=torch.float16,
          device_map="auto"
      )
      
      # Generate response
      inputs = tokenizer(prompt, return_tensors="pt")
      with torch.no_grad():
          outputs = model.generate(
              **inputs,
              max_new_tokens=max_tokens,
              temperature=temperature,
              do_sample=True,
              pad_token_id=tokenizer.eos_token_id
          )
      
      response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
      print(json.dumps({"response": response}))
  
  if __name__ == "__main__":
      main()
  """,
  model_path: "microsoft/DialoGPT-medium",
  temperature: 0.7,
  max_tokens: 512
})

{:ok, response} = AxiomAi.chat(client, "What is machine learning?")
```

#### Option B: Embedded Python Code (Pythonx)

```elixir
# Using embedded Python with automatic dependency management
client = AxiomAi.new(:local, %{
  python_deps: """
  [project]
  name = "custom_inference"
  version = "0.1.0"
  requires-python = ">=3.8"
  dependencies = [
    "torch >= 2.0.0",
    "transformers >= 4.35.0",
    "accelerate >= 0.20.0"
  ]
  """,
  python_code: """
  import torch
  from transformers import AutoTokenizer, AutoModelForCausalLM
  
  # Global variables for model caching
  _model = None
  _tokenizer = None
  _current_model_path = None
  
  def load_model(model_path):
      global _model, _tokenizer, _current_model_path
      
      if _current_model_path != model_path:
          _tokenizer = AutoTokenizer.from_pretrained(model_path)
          _model = AutoModelForCausalLM.from_pretrained(
              model_path,
              torch_dtype=torch.float16,
              device_map="auto"
          )
          _current_model_path = model_path
      
      return _tokenizer, _model
  
  def generate_response(model_path, prompt, max_tokens=1024, temperature=0.7):
      tokenizer, model = load_model(model_path)
      
      inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
      
      with torch.no_grad():
          generated_ids = model.generate(
              **inputs,
              max_new_tokens=max_tokens,
              temperature=temperature,
              do_sample=True,
              pad_token_id=tokenizer.eos_token_id
          )
      
      response = tokenizer.decode(
          generated_ids[0][inputs.input_ids.shape[1]:], 
          skip_special_tokens=True
      )
      return response
  """,
  model_path: "gpt2-medium",
  temperature: 0.8,
  max_tokens: 256
})

{:ok, response} = AxiomAi.chat(client, "Write a short story about AI")
```

### 3. Vision Models

Run multimodal models that can process both text and images:

```elixir
# Vision-language model
vision_client = AxiomAi.new(:local, %{
  python_deps: """
  [project]
  name = "vision_inference"
  requires-python = ">=3.8"
  dependencies = [
    "torch >= 2.0.0",
    "transformers >= 4.35.0",
    "pillow >= 9.0.0"
  ]
  """,
  python_code: """
  import torch
  from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
  from PIL import Image
  
  _model = None
  _processor = None
  _current_model_path = None
  
  def load_model(model_path):
      global _model, _processor, _current_model_path
      
      if _current_model_path != model_path:
          _processor = AutoProcessor.from_pretrained(model_path)
          _model = AutoModelForCausalLM.from_pretrained(
              model_path,
              torch_dtype=torch.float32,
              device_map="cpu"
          )
          _current_model_path = model_path
      
      return _processor, _model
  
  def generate_response(model_path, prompt, max_tokens=512, temperature=0.7):
      processor, model = load_model(model_path)
      
      # Handle image+text input: "image_path|describe this image"
      if "|" in prompt:
          image_path, text_prompt = prompt.split("|", 1)
          image = Image.open(image_path.strip()).convert('RGB')
          inputs = processor(text=text_prompt.strip(), images=image, return_tensors="pt")
      else:
          inputs = processor(text=prompt, return_tensors="pt")
      
      with torch.no_grad():
          generated_ids = model.generate(
              **inputs,
              max_new_tokens=max_tokens,
              temperature=temperature,
              do_sample=False
          )
      
      response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
      return response
  """,
  model_path: "microsoft/kosmos-2-patch14-224",
  temperature: 0.1,
  max_tokens: 256
})

# Process image with text prompt
{:ok, response} = AxiomAi.chat(vision_client, "/path/to/image.jpg|What do you see in this image?")
```

### 4. Speech-to-Text Models

Transcribe audio files using speech recognition models:

```elixir
# Whisper model for speech transcription
speech_client = AxiomAi.new(:local, %{
  python_deps: """
  [project]
  name = "speech_inference"
  requires-python = ">=3.8"
  dependencies = [
    "torch >= 2.0.0",
    "transformers >= 4.35.0",
    "librosa >= 0.10.0"
  ]
  """,
  python_code: """
  import torch
  from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
  import librosa
  
  _model = None
  _processor = None
  _current_model_path = None
  
  def load_model(model_path):
      global _model, _processor, _current_model_path
      
      if _current_model_path != model_path:
          _processor = AutoProcessor.from_pretrained(model_path)
          _model = AutoModelForSpeechSeq2Seq.from_pretrained(
              model_path,
              torch_dtype=torch.float16,
              device_map="auto"
          )
          _current_model_path = model_path
      
      return _processor, _model
  
  def generate_response(model_path, prompt, max_tokens=448, temperature=0.0):
      processor, model = load_model(model_path)
      
      # Handle audio input: "audio_path|transcription task"
      if "|" in prompt:
          audio_path = prompt.split("|")[0].strip()
          audio, _ = librosa.load(audio_path, sr=16000)
          
          inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
          
          with torch.no_grad():
              predicted_ids = model.generate(
                  inputs["input_features"],
                  max_new_tokens=max_tokens,
                  temperature=temperature,
                  do_sample=False
              )
          
          transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
          return transcription
      else:
          return "Error: Please provide audio file path"
  """,
  model_path: "openai/whisper-base",
  temperature: 0.0,
  max_tokens: 448
})

# Transcribe audio file
{:ok, response} = AxiomAi.chat(speech_client, "/path/to/audio.wav|Transcribe this audio")
```

### 5. Using Templates for Quick Setup

AxiomAI provides built-in templates for common model types:

```elixir
# Use built-in templates with overrides
alias AxiomAi.LocalModels.Templates

# Text generation template
text_config = Templates.create_from_template(:pythonx_text, %{
  model_path: "meta-llama/Llama-2-7b-chat-hf",
  temperature: 0.9,
  max_tokens: 2048
})

client = AxiomAi.new(:local, text_config)

# Vision template
vision_config = Templates.create_from_template(:pythonx_vision, %{
  model_path: "microsoft/kosmos-2-patch14-224"
})

vision_client = AxiomAi.new(:local, vision_config)

# HTTP endpoint templates
ollama_config = Templates.create_from_template(:http_ollama, %{
  endpoint: "http://localhost:11434",
  model: "llama2:13b"
})

ollama_client = AxiomAi.new(:local, ollama_config)

# List available templates
IO.inspect(Templates.list_templates())
# [:pythonx_text, :pythonx_vision, :pythonx_speech, :http_openai, :http_ollama, :custom]
```

### Configuration Options

All local provider configurations support these common options:

```elixir
%{
  # Execution type (automatically detected):
  predefined_model: "qwen2.5-0.5b",        # Use predefined model
  endpoint: "http://localhost:8000",        # HTTP endpoint
  python_script: "script content...",       # Python script execution
  python_code: "code content...",          # Embedded Python code
  
  # Model parameters:
  model_path: "path/to/model",             # Model identifier or path
  model: "model-name",                     # Model name for HTTP APIs
  temperature: 0.7,                        # Sampling temperature (0.0-2.0)
  max_tokens: 1024,                        # Maximum tokens to generate
  
  # HTTP-specific:
  api_format: :openai,                     # :openai or :ollama
  api_key: "optional-key",                 # API key if required
  
  # Python-specific:
  python_deps: "pyproject.toml content",   # Python dependencies (Pythonx only)
}
```

### Requirements

- **HTTP Endpoints**: No additional requirements
- **Python Scripts**: Python 3.8+ with required packages installed
- **Pythonx Integration**: Elixir pythonx package handles Python environment automatically

## Authentication

### Vertex AI
```bash
# Local development
gcloud auth application-default login

# Production
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### Environment Variables
```bash
# .env file
GOOGLE_CLOUD_PROJECT=your-project-id
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
DEEPSEEK_API_KEY=your-deepseek-key
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

## Configuration Options

```elixir
client = AxiomAi.new(:vertex_ai, %{
  project_id: "your-project",
  model: "gemini-1.5-pro",      # optional, default: "gemini-1.5-pro"
  region: "us-central1",        # optional, default: "us-central1"
  temperature: 0.7,             # optional, default: 0.7
  max_tokens: 1000,             # optional, default: 65536 for chat, 1024 for completion
  top_k: 40,                    # optional, default: 40
  top_p: 0.95                   # optional, default: 0.95
})
```

## Streaming Support

AxiomAI supports streaming responses for real-time text generation. Currently implemented for Vertex AI:

```elixir
# Simple streaming
client = AxiomAi.new(:vertex_ai, %{project_id: "your-project"})
{:ok, stream} = AxiomAi.stream(client, "Tell me a story")

# Process the stream
stream
|> Enum.each(fn
  {:chunk, chunk} -> IO.write(chunk)
  {:status, code} -> IO.puts("Status: #{code}")
  {:headers, headers} -> IO.inspect(headers)
  {:error, reason} -> IO.puts("Error: #{inspect(reason)}")
end)

# Streaming with conversation history
{:ok, stream} = AxiomAi.stream(client, "You are a helpful assistant", [], "Hello!")
```

**Streaming Status:**
- ✅ **Vertex AI**: Full streaming support
- ❌ **OpenAI**: Not implemented yet
- ❌ **Anthropic**: Not implemented yet
- ❌ **DeepSeek**: Not implemented yet
- ❌ **Bedrock**: Not implemented yet
- ❌ **Local**: Not implemented yet

## Credentials Configuration

### Vertex AI Service Account Setup

#### Method 1: Service Account File Path
```elixir
config = %{
  project_id: "your-gcp-project",
  service_account_path: "/path/to/service-account.json",
  model: "gemini-1.5-pro",
  region: "us-central1"
}

client = AxiomAi.new(:vertex_ai, config)
```

#### Method 2: Service Account Key Data
```elixir
# Load credentials from file
{:ok, creds_json} = File.read("/path/to/service-account.json")
{:ok, creds_map} = Jason.decode(creds_json)

config = %{
  project_id: "your-gcp-project",
  service_account_key: creds_map,
  model: "gemini-1.5-pro",
  region: "us-central1"
}

client = AxiomAi.new(:vertex_ai, config)
```

#### Method 3: Direct Access Token
```elixir
config = %{
  project_id: "your-gcp-project",
  access_token: "ya29.your-access-token",
  model: "gemini-1.5-pro",
  region: "us-central1"
}

client = AxiomAi.new(:vertex_ai, config)
```

#### Method 4: Application Default Credentials (ADC)
```elixir
# Automatically detects environment and uses appropriate method:
# - Cloud Run/GCE: Uses metadata service
# - Local: Uses gcloud CLI
config = %{
  project_id: "your-gcp-project",
  model: "gemini-1.5-pro",
  region: "us-central1"
}

client = AxiomAi.new(:vertex_ai, config)
```

### Cloud Run Deployment

**✅ Recommended: Use the default service account**
```elixir
# Cloud Run automatically provides credentials via metadata service
config = %{
  project_id: "your-gcp-project",
  model: "gemini-1.5-pro"
}
```

The library automatically detects Cloud Run environment and uses the metadata service for authentication. No additional configuration needed.

**Alternative: Mount service account file**
```dockerfile
# In your Dockerfile
COPY service-account.json /app/credentials.json
```

```elixir
# In your application
config = %{
  project_id: "your-gcp-project",
  service_account_path: "/app/credentials.json",
  model: "gemini-1.5-pro"
}
```

### Required IAM Permissions
Ensure your service account has the following roles:
- `roles/aiplatform.user` - For Vertex AI API access
- `roles/ml.developer` - For ML model operations (optional)

```bash
# Grant permissions
gcloud projects add-iam-policy-binding your-gcp-project \
  --member="serviceAccount:your-service-account@your-gcp-project.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

## License

MIT License