defmodule AxiomAi.Provider.VertexAi do
  @moduledoc """
  Google Vertex AI provider implementation.
  """

  @behaviour AxiomAi.Provider

  require Logger

  alias AxiomAi.Http


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
    http_opts = build_http_opts(config)

    case Http.post(endpoint, payload, headers, http_opts) do
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

    Logger.info("🤖 [AXIOM VERTEX] chat/4 called - project: #{project_id}, region: #{region}, model: #{model}")
    Logger.info("🤖 [AXIOM VERTEX] system_prompt length: #{String.length(system_prompt || "")}")
    Logger.info("🤖 [AXIOM VERTEX] history length: #{length(history)}")
    Logger.info("🤖 [AXIOM VERTEX] prompt length: #{String.length(prompt || "")}")

    endpoint = build_endpoint(project_id, region, model, "generateContent")
    Logger.info("🤖 [AXIOM VERTEX] endpoint: #{endpoint}")

    contents = build_contents(system_prompt, history, prompt)

    payload =
      %{
        contents: contents,
        generationConfig: build_generation_config(config)
      }
      |> maybe_put_tools(config)

    headers = build_headers(config)
    http_opts = build_http_opts(config)
    Logger.info("🤖 [AXIOM VERTEX] http_opts: #{inspect(http_opts)}")

    Logger.info("🤖 [AXIOM VERTEX] Making HTTP POST request...")
    case Http.post(endpoint, payload, headers, http_opts) do
      {:ok, %{status_code: 200, body: body}} ->
        Logger.info("🤖 [AXIOM VERTEX] ✅ HTTP 200 OK - body length: #{String.length(body || "")}")
        parse_response(body)

      {:ok, %{status_code: status_code, body: body}} ->
        Logger.error("🤖 [AXIOM VERTEX] ❌ HTTP #{status_code} - body: #{String.slice(body || "", 0, 500)}")
        {:error, %{status_code: status_code, message: body}}

      {:error, reason} ->
        Logger.error("🤖 [AXIOM VERTEX] ❌ HTTP Error: #{inspect(reason)}")
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
    http_opts = build_http_opts(config)

    case Http.post(endpoint, payload, headers, http_opts) do
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
    http_opts = build_http_opts(config)

    case Http.post_stream(endpoint, payload, headers, http_opts) do
      {:ok, response} ->
        {:ok, response}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @impl true
  def stream(config, system_prompt, history, prompt) do
    stream(config, system_prompt, history, prompt, [])
  end

  def stream(config, system_prompt, history, prompt, files) do
    %{project_id: project_id, region: region, model: model} = config

    Logger.info("🌊 [AXIOM VERTEX STREAM] stream/5 called - project: #{project_id}, region: #{region}, model: #{model}")
    Logger.info("🌊 [AXIOM VERTEX STREAM] system_prompt length: #{String.length(system_prompt || "")}")
    Logger.info("🌊 [AXIOM VERTEX STREAM] history length: #{length(history)}")
    Logger.info("🌊 [AXIOM VERTEX STREAM] prompt length: #{String.length(prompt || "")}")
    Logger.info("🌊 [AXIOM VERTEX STREAM] files count: #{length(files)}")

    endpoint = build_endpoint(project_id, region, model, "streamGenerateContent")
    Logger.info("🌊 [AXIOM VERTEX STREAM] endpoint: #{endpoint}")

    contents = build_contents(system_prompt, history, prompt, files)

    payload =
      %{
        contents: contents,
        generationConfig: build_generation_config(config)
      }
      |> maybe_put_tools(config)

    headers = build_headers(config)
    http_opts = build_http_opts(config)
    Logger.info("🌊 [AXIOM VERTEX STREAM] http_opts: #{inspect(http_opts)}")

    Logger.info("🌊 [AXIOM VERTEX STREAM] Making HTTP POST stream request...")
    case Http.post_stream(endpoint, payload, headers, http_opts) do
      {:ok, response} ->
        Logger.info("🌊 [AXIOM VERTEX STREAM] ✅ Stream started successfully")
        {:ok, response}

      {:error, reason} ->
        Logger.error("🌊 [AXIOM VERTEX STREAM] ❌ Stream Error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Generate an embedding for a single text using Vertex AI.

  ## Options
    - `:model` - Embedding model (default: "text-embedding-005")
    - `:task_type` - One of "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY", etc.
    - `:dimensions` - Output dimensionality (default: 768)

  Returns `{:ok, %{embedding: [float()]}}` or `{:error, reason}`.
  """
  @impl true
  def embed(config, text, opts \\ %{}) do
    %{project_id: project_id, region: region} = config
    model = Map.get(opts, :model, Map.get(config, :embedding_model, "text-embedding-005"))
    task_type = Map.get(opts, :task_type, "RETRIEVAL_DOCUMENT")
    dimensions = Map.get(opts, :dimensions, 768)

    endpoint = build_endpoint(project_id, region, model, "predict")
    headers = build_headers(config)
    http_opts = build_http_opts(config)

    payload = %{
      instances: [%{content: text, task_type: task_type}],
      parameters: %{outputDimensionality: dimensions}
    }

    case Http.post(endpoint, payload, headers, http_opts) do
      {:ok, %{status_code: 200, body: body}} ->
        parse_embedding_response(body)

      {:ok, %{status_code: status_code, body: body}} ->
        {:error, %{status_code: status_code, message: body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Generate embeddings for multiple texts in a single Vertex AI call.

  Same options as `embed/3`. Returns `{:ok, %{embeddings: [[float()]]}}`.
  """
  @impl true
  def batch_embed(config, texts, opts \\ %{}) when is_list(texts) do
    %{project_id: project_id, region: region} = config
    model = Map.get(opts, :model, Map.get(config, :embedding_model, "text-embedding-005"))
    task_type = Map.get(opts, :task_type, "RETRIEVAL_DOCUMENT")
    dimensions = Map.get(opts, :dimensions, 768)

    endpoint = build_endpoint(project_id, region, model, "predict")
    headers = build_headers(config)
    http_opts = build_http_opts(config)

    instances = Enum.map(texts, fn text -> %{content: text, task_type: task_type} end)

    payload = %{
      instances: instances,
      parameters: %{outputDimensionality: dimensions}
    }

    case Http.post(endpoint, payload, headers, http_opts) do
      {:ok, %{status_code: 200, body: body}} ->
        parse_batch_embedding_response(body)

      {:ok, %{status_code: status_code, body: body}} ->
        {:error, %{status_code: status_code, message: body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp parse_embedding_response(body) do
    case Jason.decode(body) do
      {:ok, %{"predictions" => [%{"embeddings" => %{"values" => values}} | _]}} ->
        {:ok, %{embedding: values}}

      {:ok, %{"error" => error}} ->
        {:error, error}

      {:ok, response} ->
        {:error, %{message: "Unexpected embedding response", response: response}}

      {:error, reason} ->
        {:error, %{message: "JSON decode error", reason: reason}}
    end
  end

  defp parse_batch_embedding_response(body) do
    case Jason.decode(body) do
      {:ok, %{"predictions" => predictions}} when is_list(predictions) ->
        embeddings =
          Enum.map(predictions, fn
            %{"embeddings" => %{"values" => values}} -> values
            _ -> nil
          end)

        {:ok, %{embeddings: embeddings}}

      {:ok, %{"error" => error}} ->
        {:error, error}

      {:ok, response} ->
        {:error, %{message: "Unexpected batch embedding response", response: response}}

      {:error, reason} ->
        {:error, %{message: "JSON decode error", reason: reason}}
    end
  end

  defp build_endpoint(project_id, region, model, action) do
    # El endpoint global usa host `aiplatform.googleapis.com` (sin prefijo de
    # región); el regional usa `<region>-aiplatform.googleapis.com`. Los modelos
    # nuevos (Gemini 3.x) son global-only, así que esto debe ser correcto.
    host =
      if region == "global",
        do: "aiplatform.googleapis.com",
        else: "#{region}-aiplatform.googleapis.com"

    "https://#{host}/v1/projects/#{project_id}/locations/#{region}/publishers/google/models/#{model}:#{action}"
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

  defp build_http_opts(config) do
    [
      timeout: Map.get(config, :timeout, 30_000),
      recv_timeout: Map.get(config, :recv_timeout, 30_000)
    ]
  end

  # Adjunta `tools` al payload cuando el config las trae (p. ej. Google Search
  # grounding: `[%{googleSearch: %{}}]`). Si no hay tools, el payload no cambia.
  defp maybe_put_tools(payload, config) do
    case Map.get(config, :tools) do
      tools when is_list(tools) and tools != [] -> Map.put(payload, :tools, tools)
      _ -> payload
    end
  end

  # generationConfig base + overrides opcionales desde el config:
  #   :response_mime_type -> responseMimeType (p. ej. "application/json")
  #   :response_schema    -> responseSchema  (fuerza salida estructurada)
  # Útil para clasificadores que deben devolver JSON confiable.
  defp build_generation_config(config) do
    %{
      temperature: Map.get(config, :temperature, 0.7),
      maxOutputTokens: Map.get(config, :max_tokens, 65536),
      topK: Map.get(config, :top_k, 40),
      topP: Map.get(config, :top_p, 0.95)
    }
    |> maybe_put(:responseMimeType, Map.get(config, :response_mime_type))
    |> maybe_put(:responseSchema, Map.get(config, :response_schema))
  end

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)

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
    build_contents(system_prompt, history, prompt, [])
  end

  defp build_contents(system_prompt, history, prompt, files) do
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

    # Build parts for user message with files
    file_parts =
      Enum.map(files, fn file ->
        %{
          fileData: %{
            mimeType: file.mime_type,
            fileUri: file.file_uri
          }
        }
      end)

    text_part = %{text: prompt}
    user_parts = file_parts ++ [text_part]

    user_message = %{
      role: "user",
      parts: user_parts
    }

    [system_message, model_system_response] ++ history_messages ++ [user_message]
  end
end
