# AxiomAI

[![Hex.pm](https://img.shields.io/hexpm/v/axiom_ai.svg)](https://hex.pm/packages/axiom_ai)
[![Documentation](https://img.shields.io/badge/docs-hexdocs-blue.svg)](https://hexdocs.pm/axiom_ai/)

A unified Elixir client for multiple AI providers including Vertex AI, OpenAI, Anthropic Claude, DeepSeek, AWS Bedrock, and local AI models.

## Prerequisites

- Elixir 1.14 or later
- Erlang/OTP 25 or later
- Mix build tool

## Installation

Add `axiom_ai` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:axiom_ai, "~> 0.1.0"}
  ]
end
```

Or install directly from GitHub:

```elixir
def deps do
  [
    {:axiom_ai, git: "https://github.com/liuspatt/axiom.git"}
  ]
end
```

Then run:

```bash
mix deps.get
```

## Quick Start

1. **Install dependencies:**
   ```bash
   mix deps.get
   ```

2. **Set up authentication** (see [Authentication](#authentication-for-vertex-ai) section below)

3. **Start using the library:**
   ```elixir
   # Example with Vertex AI
   client = AxiomAi.new(:vertex_ai, %{project_id: "your-gcp-project"})
   {:ok, response} = AxiomAi.chat(client, "Hello, how are you?")
   ```

## Local Development Setup

### For GCP/Vertex AI Development

1. **Install Google Cloud SDK:**
   ```bash
   # macOS
   brew install google-cloud-sdk
   
   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate with Google Cloud:**
   ```bash
   # Option 1: Application Default Credentials (recommended for local dev)
   gcloud auth application-default login
   
   # Option 2: Set service account credentials
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   ```

3. **Set your GCP project:**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

4. **Enable required APIs:**
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

### Environment Variables

Create a `.env` file (add to `.gitignore`) for local development:

```bash
# GCP Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Other provider API keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
DEEPSEEK_API_KEY=your-deepseek-key

# AWS Configuration (if using Bedrock)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=us-east-1
```

### Testing the Setup

Run the included example to verify your setup:

```bash
# Test with predefined local models
mix run examples/local_models_usage.exs

# Test with your GCP setup
iex -S mix
```

In IEx:
```elixir
# Test Vertex AI connection
client = AxiomAi.new(:vertex_ai, %{project_id: "your-project-id"})
{:ok, response} = AxiomAi.chat(client, "Hello!")
IO.puts(response.response)
```

## Usage

### Vertex AI

```elixir
# Create a client
client = AxiomAi.new(:vertex_ai, %{
  project_id: "your-gcp-project",
  region: "us-central1",  # optional, defaults to us-central1
  model: "gemini-1.5-pro" # optional, defaults to gemini-1.5-pro
})

# Chat with the model
{:ok, response} = AxiomAi.chat(client, "Hello, how are you?")
IO.puts(response.response)

# Generate completions
{:ok, completion} = AxiomAi.complete(client, "The weather today is", %{max_tokens: 50})
IO.puts(completion.completion)
```

### Authentication for Vertex AI

AxiomAI supports multiple authentication methods for Vertex AI. Choose the method that best fits your environment:

#### 1. Application Default Credentials (Recommended for Local Development)

This is the easiest method for local development and automatically works in many Google Cloud environments:

```bash
# Login and set up ADC
gcloud auth application-default login

# Verify authentication
gcloud auth application-default print-access-token
```

```elixir
# No additional config needed - ADC is used automatically
client = AxiomAi.new(:vertex_ai, %{
  project_id: "your-project"
})
```

**When to use:** Local development, Google Cloud Shell, Compute Engine instances with default service accounts.

#### 2. Service Account Key File (Recommended for Production)

Create and download a service account key file for production environments:

```bash
# Create a service account
gcloud iam service-accounts create axiom-ai-service \
  --description="Service account for AxiomAI" \
  --display-name="AxiomAI Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:axiom-ai-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# Create and download key file
gcloud iam service-accounts keys create ~/axiom-ai-credentials.json \
  --iam-account=axiom-ai-service@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

```elixir
client = AxiomAi.new(:vertex_ai, %{
  project_id: "your-project",
  service_account_path: "/path/to/service-account.json"
})
```

**When to use:** Production environments, CI/CD pipelines, containers.

#### 3. Service Account Key (In-memory)

Load service account credentials directly into your application:

```elixir
# Load from file
service_account_key = File.read!("/path/to/service-account.json") |> Jason.decode!()

# Or define directly (not recommended for production)
service_account_key = %{
  "type" => "service_account",
  "client_email" => "your-service-account@your-project.iam.gserviceaccount.com",
  "private_key" => "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "project_id" => "your-project"
}

client = AxiomAi.new(:vertex_ai, %{
  project_id: "your-project",
  service_account_key: service_account_key
})
```

**When to use:** When you need to embed credentials in your application or load them from environment variables.

#### 4. Direct Access Token

Use a pre-obtained access token (useful for testing or special authentication flows):

```bash
# Get an access token manually
gcloud auth print-access-token
```

```elixir
client = AxiomAi.new(:vertex_ai, %{
  project_id: "your-project",
  access_token: "your-access-token"
})
```

**When to use:** Testing, custom authentication flows, or when you already have an access token.

#### Troubleshooting Authentication

Common issues and solutions:

- **"Permission denied" errors:** Ensure your service account has the `roles/aiplatform.user` role
- **"Project not found":** Verify your project ID is correct and the AI Platform API is enabled
- **"Invalid credentials":** Check that your service account key file is valid JSON and not corrupted
- **ADC not working:** Run `gcloud auth application-default login` again or check `$GOOGLE_APPLICATION_CREDENTIALS`

```bash
# Debug authentication
gcloud auth list
gcloud config list project
gcloud services list --enabled | grep aiplatform
```

### Configuration Options

#### Vertex AI
- `project_id` (required): Your GCP project ID
- `region` (optional): GCP region, defaults to "us-central1"
- `model` (optional): Model name, defaults to "gemini-1.5-pro"
- `access_token` (optional): Direct access token for authentication
- `service_account_path` (optional): Path to service account JSON file
- `service_account_key` (optional): Service account key as a map
- `temperature` (optional): Controls randomness (0.0-1.0)
- `max_tokens` (optional): Maximum tokens to generate
- `top_k` (optional): Top-k sampling parameter
- `top_p` (optional): Top-p sampling parameter

**Authentication Priority Order:**
1. `access_token` (if provided)
2. `service_account_key` (if provided)
3. `service_account_path` (if provided)
4. Application Default Credentials (fallback)

### OpenAI

```elixir
client = AxiomAi.new(:openai, %{
  api_key: "your-openai-api-key",
  model: "gpt-3.5-turbo"  # optional, defaults to gpt-4
})

{:ok, response} = AxiomAi.chat(client, "Hello!")
```

### Anthropic Claude

```elixir
client = AxiomAi.new(:anthropic, %{
  api_key: "your-anthropic-api-key",
  model: "claude-3-haiku-20240307"  # optional, defaults to claude-3-sonnet-20240229
})

{:ok, response} = AxiomAi.chat(client, "Hello!")
```

### DeepSeek

```elixir
client = AxiomAi.new(:deepseek, %{
  api_key: "your-deepseek-api-key",
  model: "deepseek-chat"  # optional, defaults to deepseek-chat
})

{:ok, response} = AxiomAi.chat(client, "Hello!")
```

### AWS Bedrock

```elixir
# Using AWS credentials from environment or IAM roles
client = AxiomAi.new(:bedrock, %{
  model: "anthropic.claude-3-haiku-20240307-v1:0",
  region: "us-east-1"  # optional, defaults to us-east-1
})

# Or with explicit AWS credentials
client = AxiomAi.new(:bedrock, %{
  model: "anthropic.claude-3-haiku-20240307-v1:0",
  region: "us-east-1",
  access_key: "your-aws-access-key",
  secret_key: "your-aws-secret-key"
})

{:ok, response} = AxiomAi.chat(client, "Hello!")
```

### Local AI

#### HTTP Endpoints (OpenAI-compatible, Ollama, etc.)

```elixir
client = AxiomAi.new(:local, %{
  endpoint: "http://localhost:8080",
  model: "default"  # optional
})

{:ok, response} = AxiomAi.chat(client, "Hello!")
```

#### Predefined Models

Use pre-configured local models for easy access:

```elixir
# Use a predefined Qwen model
client = AxiomAi.new(:local, %{
  predefined_model: "qwen2.5-0.5b"
})

{:ok, response} = AxiomAi.chat(client, "Hello, how are you?")
IO.puts(response.response)

# List available predefined models
predefined_models = AxiomAi.LocalModels.list_models()
IO.inspect(predefined_models)
```

**Available Predefined Models:**
- `qwen2.5-0.5b`: Qwen2.5 0.5B (pythonx)
- `qwen2.5-1.5b`: Qwen2.5 1.5B (pythonx)  
- `qwen2.5-3b`: Qwen2.5 3B (pythonx)
- `codellama-7b`: Code Llama 7B (http)
- `llama3-8b`: Llama 3 8B (http)
- `mistral-7b`: Mistral 7B (http)

#### Custom Python Models

```elixir
# Execute custom Python script
client = AxiomAi.new(:local, %{
  python_script: """
  import json
  import sys
  
  def generate_response(model_path, prompt, max_tokens, temperature):
    # Your custom model logic here
    return f"Response from {model_path}: {prompt}"
  
  if __name__ == "__main__":
    data = json.loads(sys.argv[1])
    response = generate_response(
      data["model_path"],
      data["prompt"], 
      data["max_tokens"],
      data["temperature"]
    )
    print(json.dumps({"response": response}))
  """,
  model_path: "/path/to/your/model"
})

{:ok, response} = AxiomAi.chat(client, "Hello!")
```

## Supported Providers

| Provider | Status | Authentication | Models |
|----------|--------|----------------|--------|
| **Vertex AI** | ✅ Complete | Service Account, ADC, Access Token | Gemini models |
| **OpenAI** | ✅ Complete | API Key | GPT-3.5, GPT-4, etc. |
| **Anthropic** | ✅ Complete | API Key | Claude 3 models |
| **DeepSeek** | ✅ Complete | API Key | DeepSeek Chat |
| **AWS Bedrock** | ✅ Complete | AWS Credentials, IAM Roles | Claude, Titan, Llama, AI21 |
| **Local AI** | ✅ Complete | Optional API Key | Custom/OpenAI-compatible |

## Configuration Options

### Common Options
- `temperature` (optional): Controls randomness (0.0-1.0)
- `max_tokens` (optional): Maximum tokens to generate
- `top_p` (optional): Top-p sampling parameter

### Provider-Specific Options

#### AWS Bedrock
- `model` (required): Bedrock model ID (e.g., "anthropic.claude-3-haiku-20240307-v1:0")
- `region` (optional): AWS region, defaults to "us-east-1"
- `access_key` (optional): AWS access key ID
- `secret_key` (optional): AWS secret access key

**Supported Bedrock Models:**
- Anthropic Claude: `anthropic.claude-3-*`
- Amazon Titan: `amazon.titan-*`
- Meta Llama: `meta.llama*`
- AI21 Jurassic: `ai21.j2-*`

## Roadmap

- [x] Vertex AI provider
- [x] OpenAI provider
- [x] Anthropic provider
- [x] DeepSeek provider
- [x] AWS Bedrock provider
- [x] Local AI provider
- [ ] Streaming responses
- [ ] Function calling support
- [ ] Image/multimodal support

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.