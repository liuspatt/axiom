defmodule AxiomAi.Client do
  @moduledoc """
  Client struct and core functionality for AxiomAI.
  """

  alias AxiomAi.Provider

  defstruct [:provider, :config]

  @type t :: %__MODULE__{
          provider: atom(),
          config: map()
        }

  @doc """
  Sends a chat message using the configured provider.
  """
  @spec chat(t(), String.t()) :: {:ok, map()} | {:error, any()}
  def chat(%__MODULE__{provider: provider, config: config}, message) do
    Provider.chat(provider, config, message)
  end

  @doc """
  Sends a chat message with system prompt, history, and user prompt using the configured provider.
  """
  @spec chat(t(), String.t(), list(), String.t()) :: {:ok, map()} | {:error, any()}
  def chat(%__MODULE__{provider: provider, config: config}, system_prompt, history, prompt) do
    Provider.chat(provider, config, system_prompt, history, prompt)
  end

  @doc """
  Generates a completion using the configured provider.
  """
  @spec complete(t(), String.t(), map()) :: {:ok, map()} | {:error, any()}
  def complete(%__MODULE__{provider: provider, config: config}, prompt, options) do
    Provider.complete(provider, config, prompt, options)
  end

  @doc """
  Streams a chat message using the configured provider.
  """
  @spec stream(t(), String.t()) :: {:ok, Enumerable.t()} | {:error, any()}
  def stream(%__MODULE__{provider: provider, config: config}, message) do
    Provider.stream(provider, config, message)
  end

  @doc """
  Streams a chat message with system prompt, history, and user prompt using the configured provider.
  """
  @spec stream(t(), String.t(), list(), String.t()) :: {:ok, Enumerable.t()} | {:error, any()}
  def stream(%__MODULE__{provider: provider, config: config}, system_prompt, history, prompt) do
    Provider.stream(provider, config, system_prompt, history, prompt)
  end

  @doc """
  Generates an embedding for a single text.
  """
  @spec embed(t(), String.t(), map()) :: {:ok, map()} | {:error, any()}
  def embed(%__MODULE__{provider: provider, config: config}, text, opts \\ %{}) do
    Provider.embed(provider, config, text, opts)
  end

  @doc """
  Generates embeddings for multiple texts in a single call.
  """
  @spec batch_embed(t(), list(), map()) :: {:ok, map()} | {:error, any()}
  def batch_embed(%__MODULE__{provider: provider, config: config}, texts, opts \\ %{}) do
    Provider.batch_embed(provider, config, texts, opts)
  end

  @doc """
  Streams a chat message with system prompt, history, user prompt, and files using the configured provider.
  Files should be a list of maps with :mime_type and :file_uri keys.
  """
  @spec stream(t(), String.t(), list(), String.t(), list()) ::
          {:ok, Enumerable.t()} | {:error, any()}
  def stream(
        %__MODULE__{provider: provider, config: config},
        system_prompt,
        history,
        prompt,
        files
      ) do
    Provider.stream(provider, config, system_prompt, history, prompt, files)
  end
end
