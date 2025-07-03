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
end
