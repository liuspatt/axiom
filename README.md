# AxiomAI

A unified Elixir client for multiple AI providers including Vertex AI, OpenAI, Anthropic, and local PyTorch models.

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

## Roadmap

- [x] Vertex AI provider
- [ ] OpenAI provider
- [ ] Anthropic provider
- [ ] Local PyTorch models provider
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