defmodule AxiomAi.Provider do
  @moduledoc """
  Provider behavior and dispatcher for different AI providers.
  """

  @callback chat(config :: map(), message :: String.t()) :: {:ok, map()} | {:error, any()}
  @callback complete(config :: map(), prompt :: String.t(), options :: map()) ::
              {:ok, map()} | {:error, any()}

  @doc """
  Dispatches chat requests to the appropriate provider.
  """
  @spec chat(atom(), map(), String.t()) :: {:ok, map()} | {:error, any()}
  def chat(provider, config, message) do
    provider_module = get_provider_module(provider)
    provider_module.chat(config, message)
  end

  @doc """
  Dispatches completion requests to the appropriate provider.
  """
  @spec complete(atom(), map(), String.t(), map()) :: {:ok, map()} | {:error, any()}
  def complete(provider, config, prompt, options) do
    provider_module = get_provider_module(provider)
    provider_module.complete(config, prompt, options)
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
