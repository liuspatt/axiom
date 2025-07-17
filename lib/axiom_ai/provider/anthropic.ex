defmodule AxiomAi.Provider.Anthropic do
  @moduledoc """
  Anthropic Claude provider implementation.
  """

  @behaviour AxiomAi.Provider

  alias AxiomAi.Http

  @impl true
  def chat(config, message) do
    %{api_key: api_key, model: model} = config
    base_url = Map.get(config, :base_url, "https://api.anthropic.com")

    endpoint = "#{base_url}/v1/messages"

    payload = %{
      model: model,
      max_tokens: Map.get(config, :max_tokens, 1024),
      messages: [
        %{
          role: "user",
          content: message
        }
      ]
    }

    headers = [
      {"x-api-key", api_key},
      {"Content-Type", "application/json"},
      {"anthropic-version", "2023-06-01"}
    ]

    case Http.post(endpoint, payload, headers) do
      {:ok, %{status_code: 200, body: body}} ->
        parse_response(body)

      {:ok, %{status_code: status_code, body: body}} ->
        {:error, %{status_code: status_code, message: body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @impl true
  def chat(_config, _system_prompt, _history, _prompt) do
    {:error, :not_implemented}
  end

  @impl true
  def complete(config, prompt, options) do
    %{api_key: api_key, model: model} = config
    base_url = Map.get(config, :base_url, "https://api.anthropic.com")

    endpoint = "#{base_url}/v1/messages"

    payload = %{
      model: model,
      max_tokens: Map.get(options, :max_tokens, Map.get(config, :max_tokens, 1024)),
      messages: [
        %{
          role: "user",
          content: prompt
        }
      ]
    }

    headers = [
      {"x-api-key", api_key},
      {"Content-Type", "application/json"},
      {"anthropic-version", "2023-06-01"}
    ]

    case Http.post(endpoint, payload, headers) do
      {:ok, %{status_code: 200, body: body}} ->
        parse_completion_response(body)

      {:ok, %{status_code: status_code, body: body}} ->
        {:error, %{status_code: status_code, message: body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp parse_response(body) do
    case Jason.decode(body) do
      {:ok, %{"content" => [%{"text" => text} | _]}} ->
        {:ok, %{response: text}}

      {:ok, %{"error" => error}} ->
        {:error, error}

      {:ok, response} ->
        {:error, %{message: "Unexpected response format", response: response}}

      {:error, reason} ->
        {:error, %{message: "JSON decode error", reason: reason}}
    end
  end

  defp parse_completion_response(body) do
    case Jason.decode(body) do
      {:ok, %{"content" => [%{"text" => text} | _]}} ->
        {:ok, %{completion: text}}

      {:ok, %{"error" => error}} ->
        {:error, error}

      {:ok, response} ->
        {:error, %{message: "Unexpected response format", response: response}}

      {:error, reason} ->
        {:error, %{message: "JSON decode error", reason: reason}}
    end
  end

  @impl true
  def stream(_config, _message) do
    {:error, :not_implemented}
  end

  @impl true
  def stream(_config, _system_prompt, _history, _prompt) do
    {:error, :not_implemented}
  end
end
