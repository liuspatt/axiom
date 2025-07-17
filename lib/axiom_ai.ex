defmodule AxiomAi do
  @moduledoc """
  AxiomAI - A unified Elixir client for multiple AI providers.

  This library provides a consistent interface for interacting with various AI providers
  including Vertex AI, OpenAI, Anthropic, and local PyTorch models.

  ## Example

      iex> client = AxiomAi.new(:vertex_ai, %{project_id: "my-project", region: "us-central1"})
      iex> AxiomAi.chat(client, "Hello, how are you?")
      {:ok, %{response: "I'm doing well, thank you for asking!"}}
  """

  alias AxiomAi.{Client, Config}

  @type provider :: :vertex_ai | :openai | :anthropic | :local
  @type client :: %Client{}
  @type config :: map()
  @type message :: String.t()
  @type response :: {:ok, map()} | {:error, any()}

  @doc """
  Creates a new client for the specified AI provider.

  ## Parameters
    - provider: The AI provider to use (:vertex_ai, :openai, :anthropic, :local)
    - config: Provider-specific configuration map

  ## Examples

      iex> AxiomAi.new(:vertex_ai, %{project_id: "my-project", region: "us-central1"})
      %AxiomAi.Client{provider: :vertex_ai, config: %{...}}
  """
  @spec new(provider(), config()) :: client()
  def new(provider, config \\ %{}) do
    validated_config = Config.validate(provider, config)
    %Client{provider: provider, config: validated_config}
  end

  @doc """
  Sends a chat message to the AI provider.

  ## Parameters
    - client: The client instance
    - message: The message to send

  ## Examples

      iex> client = AxiomAi.new(:vertex_ai, %{project_id: "my-project"})
      iex> AxiomAi.chat(client, "What is the weather like?")
      {:ok, %{response: "I don't have access to real-time weather data..."}}
  """
  @spec chat(client(), message()) :: response()
  def chat(%Client{} = client, message) when is_binary(message) do
    Client.chat(client, message)
  end

  @doc """
  Sends a chat message with system prompt, history, and user prompt.

  ## Parameters
    - client: The client instance
    - system_prompt: The system prompt to set context
    - history: List of previous messages in the conversation
    - prompt: The current user message

  ## Examples

      iex> client = AxiomAi.new(:vertex_ai, %{project_id: "my-project"})
      iex> AxiomAi.chat(client, "You are a helpful assistant", [], "Hello!")
      {:ok, %{response: "Hello! How can I help you today?"}}
  """
  @spec chat(client(), String.t(), list(), String.t()) :: response()
  def chat(%Client{} = client, system_prompt, history, prompt)
      when is_binary(system_prompt) and is_list(history) and is_binary(prompt) do
    Client.chat(client, system_prompt, history, prompt)
  end

  @doc """
  Generates completions based on a prompt.

  ## Parameters
    - client: The client instance
    - prompt: The prompt to complete
    - options: Additional options (optional)

  ## Examples

      iex> client = AxiomAi.new(:vertex_ai, %{project_id: "my-project"})
      iex> AxiomAi.complete(client, "The sky is", %{max_tokens: 10})
      {:ok, %{completion: "blue and clear today."}}
  """
  @spec complete(client(), String.t(), map()) :: response()
  def complete(%Client{} = client, prompt, options \\ %{}) when is_binary(prompt) do
    Client.complete(client, prompt, options)
  end

  @doc """
  Streams a chat message to the AI provider.

  ## Parameters
    - client: The client instance
    - message: The message to send

  ## Examples

      iex> client = AxiomAi.new(:vertex_ai, %{project_id: "my-project"})
      iex> {:ok, stream} = AxiomAi.stream(client, "Tell me a story")

      # Process the stream
      stream |> Enum.each(fn
        {:chunk, chunk} -> IO.write(chunk)
        {:status, code} -> IO.puts("Status: " <> inspect(code))
        {:error, reason} -> IO.puts("Error: " <> inspect(reason))
      end)
  """
  @spec stream(client(), message()) :: {:ok, Enumerable.t()} | {:error, any()}
  def stream(%Client{} = client, message) when is_binary(message) do
    Client.stream(client, message)
  end

  @doc """
  Streams a chat message with system prompt, history, and user prompt.

  ## Parameters
    - client: The client instance
    - system_prompt: The system prompt to set context
    - history: List of previous messages in the conversation
    - prompt: The current user message

  ## Examples

      iex> client = AxiomAi.new(:vertex_ai, %{project_id: "my-project"})
      iex> {:ok, stream} = AxiomAi.stream(client, "You are a helpful assistant", [], "Hello!")

      # Process the stream
      stream |> Enum.each(fn
        {:chunk, chunk} -> IO.write(chunk)
        {:status, code} -> IO.puts("Status: " <> inspect(code))
        {:error, reason} -> IO.puts("Error: " <> inspect(reason))
      end)
  """
  @spec stream(client(), String.t(), list(), String.t()) ::
          {:ok, Enumerable.t()} | {:error, any()}
  def stream(%Client{} = client, system_prompt, history, prompt)
      when is_binary(system_prompt) and is_list(history) and is_binary(prompt) do
    Client.stream(client, system_prompt, history, prompt)
  end
end
