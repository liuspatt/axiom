defmodule AxiomAi.Config do
  @moduledoc """
  Configuration validation and management for different AI providers.
  """

  @doc """
  Validates provider-specific configuration.
  """
  @spec validate(atom(), map()) :: map()
  def validate(:vertex_ai, config) do
    config
    |> validate_required([:project_id])
    |> validate_auth_method()
    |> Map.put_new(:region, "us-central1")
    |> Map.put_new(:model, "gemini-1.5-pro")
  end

  def validate(:openai, config) do
    config
    |> validate_required([:api_key])
    |> Map.put_new(:model, "gpt-4")
    |> Map.put_new(:base_url, "https://api.openai.com/v1")
  end

  def validate(:anthropic, config) do
    config
    |> validate_required([:api_key])
    |> Map.put_new(:model, "claude-3-sonnet-20240229")
    |> Map.put_new(:base_url, "https://api.anthropic.com")
  end

  def validate(:local, config) do
    config
    |> validate_local_execution_type()
    |> Map.put_new(:model, "default")
  end

  def validate(:deepseek, config) do
    config
    |> validate_required([:api_key])
    |> Map.put_new(:model, "deepseek-chat")
    |> Map.put_new(:base_url, "https://api.deepseek.com")
  end

  def validate(:bedrock, config) do
    config
    |> validate_required([:model])
    |> validate_aws_auth()
    |> Map.put_new(:region, "us-east-1")
  end

  def validate(provider, _config) do
    raise ArgumentError, "Unsupported provider: #{inspect(provider)}"
  end

  defp validate_required(config, required_keys) do
    missing_keys = required_keys -- Map.keys(config)

    unless Enum.empty?(missing_keys) do
      raise ArgumentError, "Missing required configuration keys: #{inspect(missing_keys)}"
    end

    config
  end

  defp validate_auth_method(config) do
    auth_methods = [:access_token, :service_account_key, :service_account_path]
    has_auth = Enum.any?(auth_methods, &Map.has_key?(config, &1))

    unless has_auth do
      # Allow ADC (Application Default Credentials) as fallback
      config
    else
      # Validate service account configurations
      cond do
        Map.has_key?(config, :service_account_path) ->
          validate_service_account_path(config)

        Map.has_key?(config, :service_account_key) ->
          validate_service_account_key(config)

        true ->
          config
      end
    end
  end

  defp validate_service_account_path(config) do
    path = config.service_account_path

    unless File.exists?(path) do
      raise ArgumentError, "Service account file not found: #{path}"
    end

    config
  end

  defp validate_service_account_key(config) do
    key = config.service_account_key

    required_fields = ["client_email", "private_key", "type"]
    missing_fields = required_fields -- Map.keys(key)

    unless Enum.empty?(missing_fields) do
      raise ArgumentError,
            "Service account key missing required fields: #{inspect(missing_fields)}"
    end

    unless key["type"] == "service_account" do
      raise ArgumentError, "Invalid service account key type: #{key["type"]}"
    end

    config
  end

  defp validate_aws_auth(config) do
    # AWS auth is optional - can use environment variables, profiles, or IAM roles
    # Only validate if explicit keys are provided
    if Map.has_key?(config, :access_key) or Map.has_key?(config, :secret_key) do
      validate_required(config, [:access_key, :secret_key])
    else
      config
    end
  end

  defp validate_local_execution_type(config) do
    has_predefined = Map.has_key?(config, :predefined_model)
    has_python = Map.has_key?(config, :python_script) and Map.has_key?(config, :model_path)

    has_python_interface =
      Map.has_key?(config, :python_code) and Map.has_key?(config, :python_deps) and
        Map.has_key?(config, :model_path)

    has_endpoint = Map.has_key?(config, :endpoint)

    cond do
      has_predefined ->
        # Predefined model configuration
        config

      has_python ->
        # Python script execution
        validate_python_config(config)

      has_python_interface ->
        # Python execution
        validate_python_interface_config(config)

      has_endpoint ->
        # HTTP endpoint (original behavior)
        validate_required(config, [:endpoint])

      true ->
        raise ArgumentError,
              "Local provider requires one of: :predefined_model, :python_script + :model_path, :python_code + :python_deps + :model_path, or :endpoint"
    end
  end

  defp validate_python_config(config) do
    config
    |> validate_required([:python_script, :model_path])
    |> validate_python_version()
    |> validate_python_env_name()
    |> validate_model_path()
    |> validate_temperature()
    |> validate_max_tokens()
  end

  defp validate_python_interface_config(config) do
    config
    |> validate_required([:python_code, :python_deps, :model_path])
    |> validate_python_version()
    |> validate_python_env_name()
    |> validate_python_deps()
    |> validate_python_code()
    |> validate_model_path()
    |> validate_temperature()
    |> validate_max_tokens()
  end

  defp validate_python_version(config) do
    case Map.get(config, :python_version) do
      nil ->
        Map.put(config, :python_version, ">=3.9")

      version when is_binary(version) ->
        if String.match?(version, ~r/^(>=|==|>|<|<=|!=)\d+\.\d+(\.\d+)?$/) do
          config
        else
          raise ArgumentError,
                "Invalid python_version format: #{version}. Expected format: '>=3.9', '==3.10', etc."
        end

      _ ->
        raise ArgumentError,
              "python_version must be a string with version constraint (e.g., '>=3.9')"
    end
  end

  defp validate_python_env_name(config) do
    case Map.get(config, :python_env_name) do
      nil ->
        Map.put(config, :python_env_name, "default_env")

      env_name when is_binary(env_name) ->
        if String.match?(env_name, ~r/^[a-zA-Z][a-zA-Z0-9_]*$/) do
          config
        else
          raise ArgumentError,
                "Invalid python_env_name: #{env_name}. Must start with a letter and contain only letters, numbers, and underscores"
        end

      _ ->
        raise ArgumentError, "python_env_name must be a string"
    end
  end

  defp validate_python_deps(config) do
    case Map.get(config, :python_deps) do
      deps when is_list(deps) ->
        if Enum.all?(deps, &is_binary/1) do
          validate_python_deps_content(deps)
          config
        else
          raise ArgumentError, "All python_deps must be strings"
        end

      deps when is_binary(deps) ->
        # Handle TOML string format
        config

      _ ->
        raise ArgumentError, "python_deps must be a list of strings or a TOML string"
    end
  end

  defp validate_python_deps_content(deps) do
    Enum.each(deps, fn dep ->
      cond do
        # Allow pip index URLs with -i flag
        String.starts_with?(dep, "-i ") or dep == "-i" ->
          validate_pip_index_url(dep)

        # Allow other pip flags
        String.starts_with?(dep, "--") ->
          validate_pip_flag(dep)

        # Allow standard package specifications
        String.contains?(dep, "==") or String.contains?(dep, ">=") or String.contains?(dep, "<=") or
          String.contains?(dep, ">") or String.contains?(dep, "<") or String.contains?(dep, "~=") ->
          validate_package_spec(dep)

        # Allow simple package names (including dots)
        String.match?(dep, ~r/^[a-zA-Z0-9_.-]+$/) ->
          :ok

        # Allow package names with extras (e.g., "package[extra]")
        String.match?(dep, ~r/^[a-zA-Z0-9_.-]+\[[a-zA-Z0-9_,-]+\]/) ->
          :ok

        true ->
          raise ArgumentError,
                "Invalid python dependency format: #{dep}. Expected package name, version specification, or pip flag."
      end
    end)
  end

  defp validate_pip_index_url(dep) do
    # Extract URL from "-i URL" format
    case String.split(dep, " ", parts: 2) do
      ["-i", url] when byte_size(url) > 0 ->
        if String.starts_with?(url, "http://") or String.starts_with?(url, "https://") do
          :ok
        else
          raise ArgumentError, "Invalid index URL: #{url}. Must start with http:// or https://"
        end

      ["-i"] ->
        raise ArgumentError, "Invalid -i flag format: #{dep}. Expected '-i URL' (URL is missing)"

      _ ->
        raise ArgumentError, "Invalid -i flag format: #{dep}. Expected '-i URL'"
    end
  end

  defp validate_pip_flag(dep) do
    # Allow common pip flags
    allowed_flags = [
      "--extra-index-url",
      "--trusted-host",
      "--find-links",
      "--no-deps",
      "--no-cache-dir",
      "--upgrade",
      "--force-reinstall",
      "--no-binary",
      "--only-binary"
    ]

    flag = dep |> String.split(" ") |> List.first()

    unless flag in allowed_flags do
      raise ArgumentError,
            "Unsupported pip flag: #{flag}. Allowed flags: #{Enum.join(allowed_flags, ", ")}"
    end
  end

  defp validate_package_spec(dep) do
    # Basic validation for package specifications
    # Allow common patterns like "package==1.0.0", "package >= 1.0.0", etc.
    # Allow spaces around operators
    unless String.match?(
             dep,
             ~r/^[a-zA-Z0-9_.-]+(\[[a-zA-Z0-9_,-]+\])?\s*(==|>=|<=|>|<|~=|!=)\s*.+$/
           ) do
      raise ArgumentError,
            "Invalid package specification: #{dep}. Expected format: 'package>=version'"
    end
  end

  defp validate_python_code(config) do
    case Map.get(config, :python_code) do
      code when is_binary(code) ->
        if String.contains?(code, "def generate_response") do
          config
        else
          raise ArgumentError, "python_code must contain a 'generate_response' function"
        end

      _ ->
        raise ArgumentError, "python_code must be a string containing Python code"
    end
  end

  defp validate_model_path(config) do
    case Map.get(config, :model_path) do
      path when is_binary(path) ->
        if String.length(path) > 0 do
          config
        else
          raise ArgumentError, "model_path cannot be empty"
        end

      _ ->
        raise ArgumentError, "model_path must be a string"
    end
  end

  defp validate_temperature(config) do
    case Map.get(config, :temperature) do
      nil ->
        config

      temp when is_number(temp) ->
        if temp >= 0.0 and temp <= 2.0 do
          config
        else
          raise ArgumentError, "temperature must be between 0.0 and 2.0, got: #{temp}"
        end

      _ ->
        raise ArgumentError, "temperature must be a number"
    end
  end

  defp validate_max_tokens(config) do
    case Map.get(config, :max_tokens) do
      nil ->
        config

      tokens when is_integer(tokens) ->
        if tokens > 0 and tokens <= 4096 do
          config
        else
          raise ArgumentError, "max_tokens must be between 1 and 4096, got: #{tokens}"
        end

      _ ->
        raise ArgumentError, "max_tokens must be a positive integer"
    end
  end
end
