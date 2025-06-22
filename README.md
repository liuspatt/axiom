# AxiomAI

A unified Elixir client for multiple AI providers including Vertex AI, OpenAI, Anthropic Claude, DeepSeek, AWS Bedrock, and local AI models.

## Installation

Add `axiom_ai` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:axiom_ai, "~> 0.1.0"}
  ]
end
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

AxiomAI supports multiple authentication methods for Vertex AI:

1. **Application Default Credentials (Recommended for development)**
   ```bash
   gcloud auth application-default login
   ```
   ```elixir
   # No additional config needed - ADC is used automatically
   client = AxiomAi.new(:vertex_ai, %{
     project_id: "your-project"
   })
   ```

2. **Service Account Key File (Recommended for production)**
   ```elixir
   client = AxiomAi.new(:vertex_ai, %{
     project_id: "your-project",
     service_account_path: "/path/to/service-account.json"
   })
   ```

3. **Service Account Key (In-memory)**
   ```elixir
   service_account_key = %{
     "type" => "service_account",
     "client_email" => "your-service-account@your-project.iam.gserviceaccount.com",
     "private_key" => "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
   }

   client = AxiomAi.new(:vertex_ai, %{
     project_id: "your-project",
     service_account_key: service_account_key
   })
   ```

4. **Direct Access Token**
   ```elixir
   client = AxiomAi.new(:vertex_ai, %{
     project_id: "your-project",
     access_token: "your-access-token"
   })
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

```elixir
client = AxiomAi.new(:local, %{
  endpoint: "http://localhost:8080",
  model: "default"  # optional
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