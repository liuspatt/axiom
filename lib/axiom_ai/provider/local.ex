defmodule AxiomAi.Provider.Local do
  @moduledoc """
  Local AI provider implementation for self-hosted models.

  Supports both HTTP endpoints (OpenAI-compatible APIs, Ollama, etc.)
  and direct Python script execution for local models.
  """

  @behaviour AxiomAi.Provider

  alias AxiomAi.Http
  alias AxiomAi.LocalModels
  alias AxiomAi.PythonInterface

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

          :python_interface ->
            execute_python_interface_model(merged_config, message, mode)

          :http ->
            execute_http_request(merged_config, message, mode)
        end

      {:error, :not_found} ->
        {:error, {:predefined_model_not_found, model_key}}
    end
  end

  # Execute Python model directly
  defp execute_python_model(config, message, mode \\ :chat) do
    cond do
      Map.has_key?(config, :python_code) ->
        execute_python_interface_model(config, message, mode)

      Map.has_key?(config, :python_script) ->
        script_content = Map.get(config, :python_script, "")
        model_path = Map.get(config, :model_path, "")

        if script_content == "" or model_path == "" do
          {:error, :missing_python_config}
        else
          execute_python_script(script_content, model_path, message, config, mode)
        end

      true ->
        {:error, :missing_python_config}
    end
  end

  # Execute Python model directly
  defp execute_python_interface_model(config, message, mode) do
    python_deps = Map.get(config, :python_deps, "")
    python_code = Map.get(config, :python_code, "")
    model_path = Map.get(config, :model_path, "")
    category = Map.get(config, :category, :text_generation)

    if python_deps == "" or python_code == "" or model_path == "" do
      {:error, :missing_python_interface_config}
    else
      execute_with_python_interface(
        python_deps,
        python_code,
        model_path,
        message,
        config,
        mode,
        category
      )
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

      # Clean up temporary script file
      if File.exists?(script_path) do
        File.rm(script_path)
      end

      result
    rescue
      e ->
        {:error, {:script_execution_error, Exception.message(e)}}
    end
  end

  # Execute with PythonInterface - each client handles its own initialization
  # WARNING: Due to Python's GIL, this will block other Python executions
  # For concurrent scenarios, consider using System.cmd/3 instead
  defp execute_with_python_interface(
         python_deps,
         python_code,
         model_path,
         message,
         config,
         mode,
         category
       ) do
    # Get python_version and env_name from config
    python_version = Map.get(config, :python_version, ">=3.9")
    python_env_name = Map.get(config, :python_env_name, "default_env")

    # Ensure PythonInterface is properly initialized for this client
    case ensure_python_interface_ready(python_deps, category, python_version, python_env_name) do
      :ok ->
        # Execute inference using the python_interface
        case PythonInterface.execute_inference(model_path, message, python_code, config, category) do
          {:ok, response} ->
            case mode do
              :chat -> {:ok, %{response: response}}
              :complete -> {:ok, %{completion: response}}
            end

          {:error, reason} ->
            {:error, {:python_interface_execution_error, reason}}
        end

      {:error, reason} ->
        {:error, {:python_interface_execution_error, reason}}
    end
  end

  # Ensure PythonInterface is ready with proper initialization for each client
  defp ensure_python_interface_ready(python_deps, category, python_version, python_env_name) do
    # PythonInterface.Supervisor should be started by AxiomAi.Application
    supervisor_pid = Process.whereis(Elixir.PythonInterface.Supervisor)
    IO.puts("PythonInterface.Supervisor pid: #{inspect(supervisor_pid)}")

    case supervisor_pid do
      nil ->
        # Try to start the supervisor if it's not running
        IO.puts("PythonInterface.Supervisor not found, trying to start it...")
        case Supervisor.start_link([Elixir.PythonInterface.Janitor], strategy: :one_for_one, name: Elixir.PythonInterface.Supervisor) do
          {:ok, _pid} ->
            IO.puts("✅ PythonInterface.Supervisor started successfully")
            initialize_or_switch_python_environment(python_deps, category, python_version, python_env_name)
          {:error, {:already_started, _pid}} ->
            IO.puts("✅ PythonInterface.Supervisor already started")
            initialize_or_switch_python_environment(python_deps, category, python_version, python_env_name)
          {:error, reason} ->
            IO.puts("❌ Failed to start PythonInterface.Supervisor: #{inspect(reason)}")
            {:error, {:supervisor_start_failed, reason}}
        end
      _pid ->
        # Supervisor is running, initialize the environment
        IO.puts("PythonInterface.Supervisor is running")
        initialize_or_switch_python_environment(python_deps, category, python_version, python_env_name)
    end
  end

  defp initialize_or_switch_python_environment(python_deps, category, python_version, python_env_name) do
    # Check if Python interpreter is already initialized
    init_key = String.to_atom("python_initialized_#{python_env_name}")
    
    case Process.get(init_key) do
      true ->
        # Environment already set up, just switch to it
        IO.puts("Switching to existing Python environment: #{python_env_name}")
        switch_python_environment(python_env_name, category)
        
      _ ->
        # First time initialization
        IO.puts("Initializing new Python environment: #{python_env_name}")
        case initialize_python_environment(python_deps, category, python_version, python_env_name) do
          :ok ->
            # Mark this environment as initialized
            Process.put(init_key, true)
            # Store environment info for later switching including the actual TOML used
            toml_used = generate_toml_config(python_deps, python_version, python_env_name)
            env_info = %{
              python_deps: python_deps,
              python_version: python_version,
              python_env_name: python_env_name,
              category: category,
              toml_content: toml_used
            }
            Process.put(String.to_atom("env_info_#{python_env_name}"), env_info)
            :ok
          {:error, reason} ->
            {:error, reason}
        end
    end
  end

  defp switch_python_environment(python_env_name, _category) do
    # Get the stored environment info
    env_info_key = String.to_atom("env_info_#{python_env_name}")
    env_info = Process.get(env_info_key)
    
    if env_info do
      IO.puts("Switching Python sys.path to environment: #{python_env_name}")
      
      # Use the stored TOML content to ensure exact same cache_id calculation
      toml_content = Map.get(env_info, :toml_content) || generate_toml_config(env_info.python_deps, env_info.python_version, env_info.python_env_name)
      cache_id = 
        toml_content
        |> :erlang.md5()
        |> Base.encode32(case: :lower, padding: false)
      
      IO.puts("Environment switching debug info:")
      IO.puts("  Environment: #{python_env_name}")
      IO.puts("  Cache ID: #{cache_id}")
      IO.puts("  Dependencies: #{inspect(env_info.python_deps)}")
      
      # Use the same cache directory logic as PythonInterface.Uv
      cache_dir = get_proper_cache_dir()
      project_dir = Path.join([cache_dir, "projects", cache_id])
      venv_packages_path = Path.join([project_dir, ".venv", "lib", "python*", "site-packages"])
      
      IO.puts("  Project dir: #{project_dir}")
      IO.puts("  Looking for: #{venv_packages_path}")
      
      # Find the actual Python version directory
      actual_venv_path = case Path.wildcard(venv_packages_path) do
        [path | _] -> path
        [] -> nil
      end
      
      if actual_venv_path do
        # Switch sys.path to use this environment's packages and clear module cache
        switch_code = """
        import sys
        import importlib
        
        # Clear only safe modules to avoid torch docstring conflicts
        # Focus on document processing modules which are safer to reload
        modules_to_clear = []
        for module_name in list(sys.modules.keys()):
            # Only clear document processing modules that are safe to reload
            if any(pkg == module_name or module_name.startswith(pkg + '.') 
                   for pkg in ['docx', 'pptx', 'fitz']):
                modules_to_clear.append(module_name)
        
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]
                print(f"Cleared module: {module_name}")
        
        print("Note: Relying on sys.path switching for torch/ML modules to avoid reload conflicts")

        # Remove old venv paths
        old_paths = [p for p in sys.path if p.endswith('site-packages') and '.venv' in p]
        for old_path in old_paths:
            if old_path in sys.path:
                sys.path.remove(old_path)
                print(f"Removed old path: {old_path}")

        # Add new venv path at the beginning for priority
        new_path = "#{actual_venv_path}"
        if new_path not in sys.path:
            sys.path.insert(0, new_path)
            print(f"Added new path: {new_path}")
        
        # Verify path switching worked
        print(f"Current sys.path entries with site-packages:")
        for i, path in enumerate(sys.path):
            if 'site-packages' in path:
                print(f"  {i}: {path}")
        
        # Test if we can import a key module from this environment
        import importlib.util
        try:
            if "#{python_env_name}" == "whisper_env":
                spec = importlib.util.find_spec("torch")
                if spec:
                    print("✅ torch module found in whisper environment")
                else:
                    print("❌ torch module NOT found in whisper environment")
            elif "#{python_env_name}" == "magic_doc_env":
                spec = importlib.util.find_spec("docx")
                if spec:
                    print("✅ docx module found in magic_doc environment")
                else:
                    print("❌ docx module NOT found in magic_doc environment")
        except Exception as e:
            print(f"Module verification error: {e}")
            
        print(f"✅ Switched to environment: #{python_env_name}")
        "success"
        """
        
        # Execute the path switching code using PythonInterface.eval
        try do
          {_result, _updated_globals} = Elixir.PythonInterface.eval(switch_code, %{})
          IO.puts("✅ Successfully switched Python environment to: #{python_env_name}")
          :ok
        rescue
          e ->
            IO.puts("❌ Failed to switch Python environment: #{inspect(e)}")
            :ok  # Don't fail, just log
        end
      else
        IO.puts("⚠️ Could not find venv path for environment: #{python_env_name}")
        :ok
      end
    else
      IO.puts("⚠️ No environment info found for: #{python_env_name}")
      :ok
    end
  end

  defp get_proper_cache_dir() do
    # Use the same logic as PythonInterface.Uv.cache_dir/0
    base_dir =
      if dir = System.get_env("PYTHON_CACHE_DIR") do
        Path.expand(dir)
      else
        :filename.basedir(:user_cache, "python_interface")
      end

    version = Application.spec(:axiom_ai, :vsn) |> to_string()
    Path.join([base_dir, version, "uv", "0.5.21"])
  end

  defp initialize_python_environment(python_deps, category, python_version, python_env_name) do
    IO.puts("Initializing Python environment for category: #{category}")
    IO.puts("Python dependencies: #{inspect(python_deps)}")
    IO.puts("Python version: #{python_version}")
    IO.puts("Python environment name: #{python_env_name}")

    # Check if this is the first Python environment initialization
    global_init_key = :python_interpreter_initialized
    
    case Process.get(global_init_key) do
      nil ->
        # First environment - do full initialization
        IO.puts("First Python environment initialization")
        case initialize_with_uv_init(python_deps, python_version, python_env_name) do
          :ok ->
            Process.put(global_init_key, python_env_name)
            IO.puts("✅ Python environment initialized successfully")
            
            # Store environment info for switching including the actual TOML used
            toml_used = generate_toml_config(python_deps, python_version, python_env_name)
            env_info = %{
              python_deps: python_deps,
              python_version: python_version,
              python_env_name: python_env_name,
              category: category,
              toml_content: toml_used
            }
            Process.put(String.to_atom("env_info_#{python_env_name}"), env_info)
            
            # No need to switch - we're already in the right environment
            :ok
          {:error, reason} ->
            IO.puts("❌ uv_init failed, trying PythonInterface.init_environment...")
            
            # Fallback to PythonInterface.init_environment
            case PythonInterface.init_environment(python_deps, category) do
              :ok ->
                Process.put(global_init_key, python_env_name)
                IO.puts("✅ PythonInterface.init_environment succeeded")
                
                # Store environment info for switching including the actual TOML used
                toml_used = generate_toml_config(python_deps, python_version, python_env_name)
                env_info = %{
                  python_deps: python_deps,
                  python_version: python_version,
                  python_env_name: python_env_name,
                  category: category,
                  toml_content: toml_used
                }
                Process.put(String.to_atom("env_info_#{python_env_name}"), env_info)
                
                :ok

              {:error, fallback_reason} ->
                IO.puts("❌ Both initialization methods failed")
                IO.puts("uv_init error: #{inspect(reason)}")
                IO.puts("init_environment error: #{inspect(fallback_reason)}")
                {:error, {:initialization_failed, reason, fallback_reason}}
            end
        end
      
      first_env_name ->
        # Subsequent environment - just set up dependencies, don't reinitialize interpreter
        IO.puts("Setting up additional Python environment (interpreter already initialized with: #{first_env_name})")
        
        case setup_additional_environment(python_deps, python_version, python_env_name) do
          :ok ->
            # Store environment info for switching even for additional environments including the actual TOML used
            toml_used = generate_toml_config(python_deps, python_version, python_env_name)
            env_info = %{
              python_deps: python_deps,
              python_version: python_version,
              python_env_name: python_env_name,
              category: category,
              toml_content: toml_used
            }
            Process.put(String.to_atom("env_info_#{python_env_name}"), env_info)
            
            # Immediately switch to this environment after setup
            IO.puts("🔄 Switching to newly created environment: #{python_env_name}")
            switch_python_environment(python_env_name, category)
            
            IO.puts("✅ Additional Python environment set up successfully")
            :ok
          {:error, reason} ->
            IO.puts("⚠️ Failed to set up additional environment, but continuing: #{inspect(reason)}")
            :ok  # Don't fail the whole process
        end
    end
  end

  defp setup_additional_environment(python_deps, python_version, python_env_name) do
    IO.puts("Setting up additional environment without reinitializing interpreter")
    
    # Use uv to prepare the dependencies but don't initialize the interpreter
    try do
      # Call uv_init which will create the virtual environment and dependencies
      # but won't reinitialize the Python interpreter since it's already running
      Elixir.PythonInterface.uv_init(generate_toml_config(python_deps, python_version, python_env_name), [])
      :ok
    rescue
      e ->
        error_msg = Exception.message(e)
        IO.puts("⚠️ Additional environment setup warning: #{error_msg}")
        
        if String.contains?(error_msg, "already been initialized") do
          IO.puts("✅ Dependencies prepared, interpreter already initialized")
          :ok
        else
          {:error, {:additional_env_setup_failed, error_msg}}
        end
    end
  end

  defp generate_toml_config(python_deps, python_version, python_env_name) when is_list(python_deps) do
    deps_string =
      python_deps
      |> Enum.map(fn dep -> "    \"#{dep}\"" end)
      |> Enum.join(",\n")

    # Generate clean TOML without extra metadata to ensure consistent cache IDs
    """
    [project]
    name = "#{python_env_name}"
    version = "0.1.0"
    requires-python = "#{python_version}"
    dependencies = [
    #{deps_string}
    ]

    [build-system]
    requires = ["setuptools", "wheel"]
    build-backend = "setuptools.build_meta"
    """
  end


  defp initialize_with_uv_init(python_deps, python_version, python_env_name) when is_list(python_deps) do
    # Convert list of dependencies to TOML format - fix TOML syntax
    deps_string =
      python_deps
      |> Enum.map(fn dep -> "    \"#{dep}\"" end)  # Remove trailing comma
      |> Enum.join(",\n")

    toml_config = """
    [project]
    name = "#{python_env_name}"
    version = "0.1.0"
    requires-python = "#{python_version}"
    dependencies = [
    #{deps_string}
    ]

    [build-system]
    requires = ["setuptools", "wheel"]
    build-backend = "setuptools.build_meta"
    """

    IO.puts("Initializing Python with TOML config for environment '#{python_env_name}':")
    IO.puts(toml_config)

    try do
      # Use default uv_init with environment name embedded in TOML project name for isolation
      Elixir.PythonInterface.uv_init(toml_config, [])
      IO.puts("✅ Python dependencies initialized successfully for environment '#{python_env_name}'")
      :ok
    rescue
      e ->
        error_msg = Exception.message(e)
        IO.puts("❌ Python initialization failed for environment '#{python_env_name}': #{error_msg}")

        if String.contains?(error_msg, "already been initialized") do
          IO.puts("✅ Python already initialized for environment '#{python_env_name}', switching to it...")
          
          # Store environment info for switching including the actual TOML used
          toml_used = generate_toml_config(python_deps, python_version, python_env_name)
          env_info = %{
            python_deps: python_deps,
            python_version: python_version,
            python_env_name: python_env_name,
            category: :text_generation,  # Default category since we don't have it here
            toml_content: toml_used
          }
          Process.put(String.to_atom("env_info_#{python_env_name}"), env_info)
          
          # Switch to this environment
          switch_python_environment(python_env_name, :text_generation)
          :ok
        else
          {:error, {:uv_init_failed, error_msg}}
        end
    end
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
