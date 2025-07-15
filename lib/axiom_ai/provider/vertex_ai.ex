defmodule AxiomAi.Provider.VertexAi do
  @moduledoc """
  Google Vertex AI provider implementation.
  """

  @behaviour AxiomAi.Provider

  alias AxiomAi.Http

  @base_url "https://us-central1-aiplatform.googleapis.com/v1"

  @impl true
  def chat(config, message) do
    %{project_id: project_id, region: region, model: model} = config

    endpoint = build_endpoint(project_id, region, model, "generateContent")

    payload = %{
      contents: [
        %{
          role: "user",
          parts: [%{text: message}]
        }
      ],
      generationConfig: %{
        temperature: Map.get(config, :temperature, 0.7),
        maxOutputTokens: Map.get(config, :max_tokens, 65536),
        topK: Map.get(config, :top_k, 40),
        topP: Map.get(config, :top_p, 0.95)
      }
    }

    headers = build_headers(config)

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
  def chat(config, system_prompt, history, prompt) do
    %{project_id: project_id, region: region, model: model} = config

    endpoint = build_endpoint(project_id, region, model, "generateContent")

    contents = build_contents(system_prompt, history, prompt)

    payload = %{
      contents: contents,
      generationConfig: %{
        temperature: Map.get(config, :temperature, 0.7),
        maxOutputTokens: Map.get(config, :max_tokens, 65536),
        topK: Map.get(config, :top_k, 40),
        topP: Map.get(config, :top_p, 0.95)
      }
    }

    headers = build_headers(config)

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
  def complete(config, prompt, options) do
    %{project_id: project_id, region: region, model: model} = config

    endpoint = build_endpoint(project_id, region, model, "generateContent")

    payload = %{
      contents: [
        %{
          role: "user",
          parts: [%{text: prompt}]
        }
      ],
      generationConfig: %{
        temperature: Map.get(options, :temperature, Map.get(config, :temperature, 0.7)),
        maxOutputTokens: Map.get(options, :max_tokens, Map.get(config, :max_tokens, 1024)),
        topK: Map.get(options, :top_k, Map.get(config, :top_k, 40)),
        topP: Map.get(options, :top_p, Map.get(config, :top_p, 0.95))
      }
    }

    headers = build_headers(config)

    case Http.post(endpoint, payload, headers) do
      {:ok, %{status_code: 200, body: body}} ->
        parse_completion_response(body)

      {:ok, %{status_code: status_code, body: body}} ->
        {:error, %{status_code: status_code, message: body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @impl true
  def stream(config, message) do
    %{project_id: project_id, region: region, model: model} = config

    endpoint = build_endpoint(project_id, region, model, "streamGenerateContent")

    payload = %{
      contents: [
        %{
          role: "user",
          parts: [%{text: message}]
        }
      ],
      generationConfig: %{
        temperature: Map.get(config, :temperature, 0.7),
        maxOutputTokens: Map.get(config, :max_tokens, 65536),
        topK: Map.get(config, :top_k, 40),
        topP: Map.get(config, :top_p, 0.95)
      }
    }

    headers = build_headers(config)

    case Http.post_stream(endpoint, payload, headers) do
      {:ok, response} ->
        {:ok, response}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @impl true
  def stream(config, system_prompt, history, prompt) do
    %{project_id: project_id, region: region, model: model} = config

    endpoint = build_endpoint(project_id, region, model, "streamGenerateContent")

    contents = build_contents(system_prompt, history, prompt)

    payload = %{
      contents: contents,
      generationConfig: %{
        temperature: Map.get(config, :temperature, 0.7),
        maxOutputTokens: Map.get(config, :max_tokens, 65536),
        topK: Map.get(config, :top_k, 40),
        topP: Map.get(config, :top_p, 0.95)
      }
    }

    headers = build_headers(config)

    case Http.post_stream(endpoint, payload, headers) do
      {:ok, response} ->
        {:ok, response}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp build_endpoint(project_id, region, model, action) do
    base_region_url = String.replace(@base_url, "us-central1", region)

    "#{base_region_url}/projects/#{project_id}/locations/#{region}/publishers/google/models/#{model}:#{action}"
  end

  defp build_headers(config) do
    token = get_access_token(config)

    [
      {"Authorization", "Bearer #{token}"},
      {"Content-Type", "application/json"}
    ]
  end

  defp get_access_token(config) do
    case AxiomAi.Auth.get_gcp_token(config) do
      {:ok, token} -> token
      {:error, reason} -> raise "Failed to get access token: #{inspect(reason)}"
    end
  end

  defp parse_response(body) do
    case Jason.decode(body) do
      {:ok, %{"candidates" => [%{"content" => %{"parts" => [%{"text" => text}]}} | _]}} ->
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
      {:ok, %{"candidates" => [%{"content" => %{"parts" => [%{"text" => text}]}} | _]}} ->
        {:ok, %{completion: text}}

      {:ok, %{"error" => error}} ->
        {:error, error}

      {:ok, response} ->
        {:error, %{message: "Unexpected response format", response: response}}

      {:error, reason} ->
        {:error, %{message: "JSON decode error", reason: reason}}
    end
  end

  defp build_contents(system_prompt, history, prompt) do
    system_message = %{
      role: "user",
      parts: [%{text: system_prompt}]
    }

    model_system_response = %{
      role: "model",
      parts: [%{text: "I understand. I'll follow your instructions."}]
    }

    history_messages =
      Enum.map(history, fn
        %{role: role, content: content} ->
          vertex_role = if role == "assistant", do: "model", else: "user"
          %{role: vertex_role, parts: [%{text: content}]}

        %{"role" => role, "content" => content} ->
          vertex_role = if role == "assistant", do: "model", else: "user"
          %{role: vertex_role, parts: [%{text: content}]}

        message when is_binary(message) ->
          %{role: "user", parts: [%{text: message}]}
      end)

    user_message = %{
      role: "user",
      parts: [%{text: prompt}]
    }

    [system_message, model_system_response] ++ history_messages ++ [user_message]
  end
end
