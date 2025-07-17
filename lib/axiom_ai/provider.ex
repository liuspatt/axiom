defmodule AxiomAi.Provider do
  @moduledoc """
  Provider behavior and dispatcher for different AI providers.
  """

  @callback chat(config :: map(), message :: String.t()) :: {:ok, map()} | {:error, any()}
  @callback chat(
              config :: map(),
              system_prompt :: String.t(),
              history :: list(),
              prompt :: String.t()
            ) :: {:ok, map()} | {:error, any()}
  @callback complete(config :: map(), prompt :: String.t(), options :: map()) ::
              {:ok, map()} | {:error, any()}
  @callback stream(config :: map(), message :: String.t()) :: {:ok, any()} | {:error, any()}
  @callback stream(
              config :: map(),
              system_prompt :: String.t(),
              history :: list(),
              prompt :: String.t()
            ) :: {:ok, any()} | {:error, any()}

  @doc """
  Dispatches chat requests to the appropriate provider.
  """
  @spec chat(atom(), map(), String.t()) :: {:ok, map()} | {:error, any()}
  def chat(provider, config, message) do
    provider_module = get_provider_module(provider)
    provider_module.chat(config, message)
  end

  @doc """
  Dispatches chat requests with system prompt, history, and user prompt to the appropriate provider.
  """
  @spec chat(atom(), map(), String.t(), list(), String.t()) :: {:ok, map()} | {:error, any()}
  def chat(provider, config, system_prompt, history, prompt) do
    provider_module = get_provider_module(provider)
    provider_module.chat(config, system_prompt, history, prompt)
  end

  @doc """
  Dispatches completion requests to the appropriate provider.
  """
  @spec complete(atom(), map(), String.t(), map()) :: {:ok, map()} | {:error, any()}
  def complete(provider, config, prompt, options) do
    provider_module = get_provider_module(provider)
    provider_module.complete(config, prompt, options)
  end

  @doc """
  Dispatches streaming chat requests to the appropriate provider.
  """
  @spec stream(atom(), map(), String.t()) :: {:ok, any()} | {:error, any()}
  def stream(provider, config, message) do
    provider_module = get_provider_module(provider)
    provider_module.stream(config, message)
  end

  @doc """
  Dispatches streaming chat requests with system prompt, history, and user prompt to the appropriate provider.
  """
  @spec stream(atom(), map(), String.t(), list(), String.t()) :: {:ok, any()} | {:error, any()}
  def stream(provider, config, system_prompt, history, prompt) do
    provider_module = get_provider_module(provider)
    provider_module.stream(config, system_prompt, history, prompt)
  end

  defp get_provider_module(:vertex_ai), do: AxiomAi.Provider.VertexAi
  defp get_provider_module(:openai), do: AxiomAi.Provider.OpenAi
  defp get_provider_module(:anthropic), do: AxiomAi.Provider.Anthropic
  defp get_provider_module(:deepseek), do: AxiomAi.Provider.DeepSeek
  defp get_provider_module(:bedrock), do: AxiomAi.Provider.Bedrock
  defp get_provider_module(:local), do: AxiomAi.Provider.Local

  defp get_provider_module(provider) do
    raise ArgumentError, "Unsupported provider: #{inspect(provider)}"
  end
end
