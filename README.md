# AxiomAI

A unified Elixir client for multiple AI providers including Vertex AI, OpenAI, Anthropic, AWS Bedrock, and local PyTorch models with embedded Python support.

## Features

-   Multiple AI provider support (Vertex AI, OpenAI, Anthropic, AWS Bedrock, DeepSeek)
-   Local Python model execution with isolated environments
-   Built-in document processing and audio transcription
-   Seamless environment switching between different Python dependencies

## Installation

Add `axiom_ai` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:axiom_ai, "~> 0.1.8"}
  ]
end
```

## Examples

### 1. GCP Vertex AI Provider

```elixir
# Create a Vertex AI client
client = AxiomAi.new(:vertex_ai, %{
  project_id: "your-gcp-project-id",
  region: "us-central1",
  model: "gemini-1.5-pro",
  service_account_path: "/path/to/service-account.json"
})

# Send a chat message
{:ok, response} = AxiomAi.chat(client, "Explain quantum computing")
IO.puts(response.response)

# Generate completion
{:ok, response} = AxiomAi.complete(client, "The future of AI is", %{
  max_tokens: 100,
  temperature: 0.7
})
IO.puts(response.completion)
```

### 2. Local Whisper Model

This example demonstrates how to use a local Whisper model for audio transcription. The `python_code` is executed in an isolated environment, and the `generate_response` function is the entry point called from Elixir.

```elixir
# Create a local Whisper client
whisper_client = AxiomAi.new(:local, %{
  python_version: ">=3.9",
  python_env_name: "whisper_env",
  python_deps: [
    "torch >= 2.1.0",
    "transformers >= 4.45.0",
    "soundfile >= 0.12.1",
    "librosa >= 0.10.0"
  ],
  python_code: """
    import torch
    # ... (full Python code with imports and helper functions)

    def get_device():
        # ...
    def get_cuda_optimizations():
        # ...
    def compress_audio(audio_path):
        # ...
    def preprocess_audio(audio_path, target_sr=16000):
        # ...
    def load_whisper_model(model_path):
        # ...
    def transcribe_audio(model_path, audio_file_path, language=None, task="transcribe"):
        # ...

    # This function is the entry point called from Elixir.
    # The `prompt` argument will contain the audio file path.
    def generate_response(model_path, prompt, max_tokens=448, temperature=0.0):
      result = transcribe_audio(model_path, prompt)
      return result.get("text", "")
    """,
  model_path: "openai/whisper-large-v3",
  temperature: 0.0,
  max_tokens: 256
})

# Transcribe an audio file
audio_file = "/path/to/your/audio.wav"
{:ok, response} = AxiomAi.complete(whisper_client, audio_file)

case response.completion do
  %{"transcription" => transcription} ->
    IO.puts("Transcription: #{transcription}")
  %{"error" => error} ->
    IO.puts("Error: #{error}")
end
```

## Configuration

### Vertex AI Authentication

```elixir
# Using service account file
client = AxiomAi.new(:vertex_ai, %{
  project_id: "your-project-id",
  service_account_path: "/path/to/service-account.json"
})

# Using Application Default Credentials
client = AxiomAi.new(:vertex_ai, %{
  project_id: "your-project-id"
})
```

### Local Models

Local models run in isolated Python environments with automatic dependency management:

```elixir
client = AxiomAi.new(:local, %{
  python_env_name: "my_env", # Environment name
  python_deps: ["torch", "numpy"], # Python dependencies
  python_code: "...", # Python code with generate_response function
  model_path: "/path/to/model", # Model file path
  category: :text_generation # Environment category
})
```

## Acknowledgements

Thank you to Cocoa Xu (@cocoa-xu) for building the first prototype of embedded Python ([source](https://github.com/livebook-dev/pythonx)).
