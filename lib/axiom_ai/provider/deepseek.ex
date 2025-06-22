defmodule AxiomAi.Provider.DeepSeek do
  @moduledoc """
  DeepSeek AI provider implementation.
  """

  @behaviour AxiomAi.Provider

  alias AxiomAi.Http

  @impl true
  def chat(config, message) do
    %{api_key: api_key, model: model} = config
    base_url = Map.get(config, :base_url, "https://api.deepseek.com")

    endpoint = "#{base_url}/v1/chat/completions"

    payload = %{
      model: model,
      messages: [
        %{
          role: "user",
          content: message
        }
      ],
      temperature: Map.get(config, :temperature, 0.7),
      max_tokens: Map.get(config, :max_tokens, 1024),
      stream: false
    }

    headers = [
      {"Authorization", "Bearer #{api_key}"},
      {"Content-Type", "application/json"}
    ]

    case Http.post(endpoint, payload, headers) do
      {:ok, %{status_code: 200, body: body}} ->
        parse_chat_response(body)

      {:ok, %{status_code: status_code, body: body}} ->
        {:error, %{status_code: status_code, message: body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @impl true
  def complete(config, prompt, options) do
    %{api_key: api_key, model: model} = config
    base_url = Map.get(config, :base_url, "https://api.deepseek.com")

    endpoint = "#{base_url}/v1/chat/completions"

    payload = %{
      model: model,
      messages: [
        %{
          role: "user",
          content: prompt
        }
      ],
      temperature: Map.get(options, :temperature, Map.get(config, :temperature, 0.7)),
      max_tokens: Map.get(options, :max_tokens, Map.get(config, :max_tokens, 1024)),
      stream: false
    }

    headers = [
      {"Authorization", "Bearer #{api_key}"},
      {"Content-Type", "application/json"}
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

  defp parse_chat_response(body) do
    case Jason.decode(body) do
      {:ok, %{"choices" => [%{"message" => %{"content" => content}} | _]}} ->
        {:ok, %{response: content}}

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
      {:ok, %{"choices" => [%{"message" => %{"content" => content}} | _]}} ->
        {:ok, %{completion: content}}

      {:ok, %{"error" => error}} ->
        {:error, error}

      {:ok, response} ->
        {:error, %{message: "Unexpected response format", response: response}}

      {:error, reason} ->
        {:error, %{message: "JSON decode error", reason: reason}}
    end
  end
end