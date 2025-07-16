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
    validate_required(config, [:python_script, :model_path])
  end

  defp validate_python_interface_config(config) do
    validate_required(config, [:python_code, :python_deps, :model_path])
  end
end
