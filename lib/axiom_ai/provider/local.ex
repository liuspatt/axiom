defmodule AxiomAi.Provider.Local do
  @moduledoc """
  Local AI provider implementation for self-hosted models.

  Supports both HTTP endpoints (OpenAI-compatible APIs, Ollama, etc.)
  and direct Python script execution for local models.
  """

  @behaviour AxiomAi.Provider

  alias AxiomAi.Http
  alias AxiomAi.LocalModels

  # Track Python initialization state to avoid multiple init calls
  @python_init_key :axiom_ai_python_initialized
  @python_globals_key :axiom_ai_python_globals

  @impl true
  def chat(config, message) do
    case determine_execution_type(config) do
      :predefined ->
        execute_predefined_model(config, message)

      :python ->
        execute_python_model(config, message)

      :http ->
        execute_http_request(config, message, :chat)

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
    merged_config = Map.merge(config, options)

    case determine_execution_type(merged_config) do
      :predefined ->
        execute_predefined_model(merged_config, prompt, :complete)

      :python ->
        execute_python_model(merged_config, prompt, :complete)

      :http ->
        execute_http_request(merged_config, prompt, :complete)

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Determines how to execute the model based on configuration
  defp determine_execution_type(config) do
    cond do
      Map.has_key?(config, :predefined_model) ->
        :predefined

      Map.has_key?(config, :python_script) or Map.has_key?(config, :python_code) ->
        :python

      Map.has_key?(config, :endpoint) ->
        :http

      true ->
        {:error, :missing_execution_config}
    end
  end

  # Execute predefined model
  defp execute_predefined_model(config, message, mode \\ :chat) do
    model_key = config.predefined_model

    case LocalModels.get_model_config(model_key) do
      {:ok, model_config} ->
        merged_config = Map.merge(config, model_config)

        case model_config.type do
          :python ->
            execute_python_model(merged_config, message, mode)

          :pythonx ->
            execute_pythonx_model(merged_config, message, mode)

          :http ->
            execute_http_request(merged_config, message, mode)
        end

      {:error, :not_found} ->
        {:error, {:predefined_model_not_found, model_key}}
    end
  end

  # Execute Python model directly
  defp execute_python_model(config, message, mode \\ :chat) do
    script_content = Map.get(config, :python_script, "")
    model_path = Map.get(config, :model_path, "")

    if script_content == "" or model_path == "" do
      {:error, :missing_python_config}
    else
      execute_python_script(script_content, model_path, message, config, mode)
    end
  end

  # Execute Pythonx model directly
  defp execute_pythonx_model(config, message, mode) do
    python_deps = Map.get(config, :python_deps, "")
    python_code = Map.get(config, :python_code, "")
    model_path = Map.get(config, :model_path, "")

    if python_deps == "" or python_code == "" or model_path == "" do
      {:error, :missing_pythonx_config}
    else
      execute_with_pythonx(python_deps, python_code, model_path, message, config, mode)
    end
  end

  # Execute HTTP request (original functionality)
  defp execute_http_request(config, message, mode) do
    case mode do
      :chat -> execute_chat_request(config, message)
      :complete -> execute_completion_request(config, message)
    end
  end

  defp execute_chat_request(config, message) do
    endpoint = Map.get(config, :endpoint, "http://localhost:8000")
    model = Map.get(config, :model, "default")
    api_format = Map.get(config, :api_format, :openai)

    url = build_url(endpoint, "chat/completions", api_format)

    payload = build_chat_payload(model, message, config, api_format)
    headers = build_headers(config)

    case Http.post(url, payload, headers) do
      {:ok, %{status_code: 200, body: body}} ->
        parse_chat_response(body, api_format)

      {:ok, %{status_code: status_code, body: body}} ->
        {:error, %{status_code: status_code, message: body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp execute_completion_request(config, prompt) do
    endpoint = Map.get(config, :endpoint, "http://localhost:8000")
    model = Map.get(config, :model, "default")
    api_format = Map.get(config, :api_format, :openai)

    url = build_url(endpoint, "completions", api_format)

    payload = build_completion_payload(model, prompt, config, api_format)
    headers = build_headers(config)

    case Http.post(url, payload, headers) do
      {:ok, %{status_code: 200, body: body}} ->
        parse_completion_response(body, api_format)

      {:ok, %{status_code: status_code, body: body}} ->
        {:error, %{status_code: status_code, message: body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Execute Python script
  defp execute_python_script(script_content, model_path, message, config, mode) do
    script_path = nil

    try do
      # Create temporary script file
      script_path = create_temp_script(script_content)

      input_data = %{
        model_path: model_path,
        prompt: message,
        max_tokens: Map.get(config, :max_tokens, 1024),
        temperature: Map.get(config, :temperature, 0.7)
      }

      json_input = Jason.encode!(input_data)

      # Execute Python script
      result =
        case System.cmd("python3", [script_path, json_input]) do
          {output, 0} ->
            case Jason.decode(output) do
              {:ok, %{"response" => response}} ->
                case mode do
                  :chat -> {:ok, %{response: response}}
                  :complete -> {:ok, %{completion: response}}
                end

              {:ok, %{"error" => error}} ->
                {:error, {:python_execution_error, error}}

              {:error, reason} ->
                {:error, {:json_decode_error, reason}}
            end

          {error_output, exit_code} ->
            {:error, {:python_script_failed, exit_code, error_output}}
        end

      result
    rescue
      e ->
        {:error, {:script_execution_error, Exception.message(e)}}
    after
      # Clean up temporary script file
      if script_path && File.exists?(script_path) do
        File.rm(script_path)
      end
    end
  end

  # Execute with Pythonx
  # WARNING: Due to Python's GIL, this will block other Python executions
  # For concurrent scenarios, consider using System.cmd/3 instead
  defp execute_with_pythonx(python_deps, python_code, model_path, message, config, mode) do
    try do
      # Initialize Python environment with dependencies only once per process
      ensure_python_initialized(python_deps)

      # Prepare variables for Python execution
      max_tokens = Map.get(config, :max_tokens, 1024)
      temperature = Map.get(config, :temperature, 0.7)

      # Get or create process-specific globals to avoid concurrent access issues
      process_globals = get_process_globals()

      # Execute the Python code with the inference function
      # Using process-local globals to maintain model cache and avoid concurrency issues
      {result, updated_globals} =
        Pythonx.eval(
          """
          #{python_code}

          # Call the inference function with escaped strings to prevent injection
          response = generate_response("#{String.replace(model_path, "\"", "\\\"")}", "#{String.replace(message, "\"", "\\\"")}", #{max_tokens}, #{temperature})
          response
          """,
          process_globals
        )

      # Store updated globals back to process dictionary
      put_process_globals(updated_globals)

      # Decode the result
      response = Pythonx.decode(result)

      case mode do
        :chat -> {:ok, %{response: response}}
        :complete -> {:ok, %{completion: response}}
      end
    rescue
      e ->
        {:error, {:pythonx_execution_error, Exception.message(e)}}
    end
  end

  # Ensure Python is initialized only once per process
  defp ensure_python_initialized(python_deps) do
    case Process.get(@python_init_key) do
      nil ->
        try do
          Pythonx.uv_init(python_deps)
          Process.put(@python_init_key, true)
        rescue
          # If Python is already initialized at system level, catch any error and mark as initialized
          _error ->
            Process.put(@python_init_key, true)
            :ok
        end

      true ->
        # Already initialized, skip
        :ok
    end
  end

  # Get process-local Python globals to avoid concurrency issues
  defp get_process_globals do
    Process.get(@python_globals_key, %{})
  end

  # Store updated Python globals in process dictionary
  defp put_process_globals(globals) do
    Process.put(@python_globals_key, globals)
  end

  defp create_temp_script(script_content) do
    temp_dir = System.tmp_dir!()
    timestamp = :os.system_time(:millisecond)
    script_path = Path.join(temp_dir, "axiom_ai_script_#{timestamp}.py")

    File.write!(script_path, script_content)
    script_path
  end

  defp build_url(endpoint, path, api_format) do
    endpoint = String.trim_trailing(endpoint, "/")

    case api_format do
      :openai -> "#{endpoint}/v1/#{path}"
      :ollama -> "#{endpoint}/api/#{if path == "chat/completions", do: "chat", else: "generate"}"
      _ -> "#{endpoint}/v1/#{path}"
    end
  end

  defp build_chat_payload(model, message, config, api_format) do
    base_payload = %{
      model: model,
      temperature: Map.get(config, :temperature, 0.7),
      max_tokens: Map.get(config, :max_tokens, 1024)
    }

    case api_format do
      :openai ->
        Map.put(base_payload, :messages, [
          %{role: "user", content: message}
        ])

      :ollama ->
        Map.merge(base_payload, %{
          messages: [%{role: "user", content: message}],
          stream: false
        })

      _ ->
        Map.put(base_payload, :messages, [
          %{role: "user", content: message}
        ])
    end
  end

  defp build_completion_payload(model, prompt, config, api_format) do
    base_payload = %{
      model: model,
      prompt: prompt,
      temperature: Map.get(config, :temperature, 0.7),
      max_tokens: Map.get(config, :max_tokens, 1024)
    }

    case api_format do
      :ollama ->
        Map.merge(base_payload, %{stream: false})

      _ ->
        base_payload
    end
  end

  defp build_headers(config) do
    headers = [{"Content-Type", "application/json"}]

    case Map.get(config, :api_key) do
      nil -> headers
      api_key -> [{"Authorization", "Bearer #{api_key}"} | headers]
    end
  end

  defp parse_chat_response(body, api_format) do
    case Jason.decode(body) do
      {:ok, response} ->
        case api_format do
          :openai ->
            case response do
              %{"choices" => [%{"message" => %{"content" => content}} | _]} ->
                {:ok, %{response: content}}

              %{"error" => error} ->
                {:error, error}

              _ ->
                {:error, %{message: "Unexpected response format", response: response}}
            end

          :ollama ->
            case response do
              %{"message" => %{"content" => content}} ->
                {:ok, %{response: content}}

              %{"error" => error} ->
                {:error, error}

              _ ->
                {:error, %{message: "Unexpected response format", response: response}}
            end

          _ ->
            {:error, %{message: "Unsupported API format", api_format: api_format}}
        end

      {:error, reason} ->
        {:error, %{message: "JSON decode error", reason: reason}}
    end
  end

  defp parse_completion_response(body, api_format) do
    case Jason.decode(body) do
      {:ok, response} ->
        case api_format do
          :openai ->
            case response do
              %{"choices" => [%{"text" => text} | _]} ->
                {:ok, %{completion: text}}

              %{"error" => error} ->
                {:error, error}

              _ ->
                {:error, %{message: "Unexpected response format", response: response}}
            end

          :ollama ->
            case response do
              %{"response" => text} ->
                {:ok, %{completion: text}}

              %{"error" => error} ->
                {:error, error}

              _ ->
                {:error, %{message: "Unexpected response format", response: response}}
            end

          _ ->
            {:error, %{message: "Unsupported API format", api_format: api_format}}
        end

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
