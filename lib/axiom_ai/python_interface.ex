defmodule AxiomAi.PythonInterface do
  @moduledoc """
  Python Interface - Interface module for embedded python functionality.

  This module provides a clean API for interacting with the embedded python library
  within the axiom project. It handles Python environment management, code execution,
  and provides utilities for AI model inference.
  """

  @doc """
  Initialize Python environment with specified dependencies.

  ## Parameters
  - `python_deps`: List of Python dependencies to install
  - `category`: Category identifier for environment isolation

  ## Returns
  - `:ok` on success
  - `{:error, reason}` on failure
  """
  def init_environment(python_deps, category \\ :default) do
    init_key = String.to_atom("axiom_ai_python_initialized_#{category}")

    case Process.get(init_key) do
      nil ->
        try do
          # Create a TOML configuration for the dependencies
          toml_config = create_toml_config(python_deps)
          PythonInterface.uv_init(toml_config)
          Process.put(init_key, true)
          :ok
        rescue
          e -> {:error, {:python_init_error, Exception.message(e)}}
        end

      _already_initialized ->
        :ok
    end
  end

  @doc """
  Execute Python code with python_interface.

  ## Parameters
  - `code`: Python code string to execute
  - `globals`: Global variables dictionary (optional)
  - `category`: Category for environment isolation

  ## Returns
  - `{:ok, result}` with decoded Python result
  - `{:error, reason}` on failure
  """
  def execute_python(code, globals \\ %{}, category \\ :default) do
    try do
      process_globals = get_process_globals(category, globals)

      {result, updated_globals} = PythonInterface.eval(code, process_globals)

      put_process_globals(updated_globals, category)

      decoded_result = PythonInterface.decode(result)
      {:ok, decoded_result}
    rescue
      e -> {:error, {:python_execution_error, Exception.message(e)}}
    end
  end

  @doc """
  Execute AI model inference using Python.

  ## Parameters
  - `model_path`: Path to the AI model
  - `message`: Input message for the model
  - `python_code`: Python code template for inference
  - `config`: Configuration map with model parameters
  - `category`: Category for environment isolation

  ## Returns
  - `{:ok, response}` with model response
  - `{:error, reason}` on failure
  """
  def execute_inference(model_path, message, python_code, config, category \\ :default) do
    try do
      max_tokens = Map.get(config, :max_tokens, 1024)
      temperature = Map.get(config, :temperature, 0.7)

      process_globals = get_process_globals(category)

      inference_code = """
      #{python_code}

      # Call the inference function with escaped strings to prevent injection
      response = generate_response("#{String.replace(model_path, "\"", "\\\"")}", "#{String.replace(message, "\"", "\\\"")}", #{max_tokens}, #{temperature})
      response
      """

      {result, updated_globals} = PythonInterface.eval(inference_code, process_globals)

      put_process_globals(updated_globals, category)

      response = PythonInterface.decode(result)
      {:ok, response}
    rescue
      e -> {:error, {:inference_error, Exception.message(e)}}
    end
  end

  @doc """
  Execute streaming inference for AI models.

  ## Parameters
  - `model_path`: Path to the AI model
  - `message`: Input message for the model
  - `python_code`: Python code template for streaming inference
  - `config`: Configuration map with model parameters
  - `category`: Category for environment isolation

  ## Returns
  - Stream of responses or error
  """
  def execute_streaming_inference(model_path, message, python_code, config, category \\ :default) do
    # This is a placeholder for streaming functionality
    # In a real implementation, this would handle streaming responses
    case execute_inference(model_path, message, python_code, config, category) do
      {:ok, response} ->
        Stream.iterate(response, fn _ -> :halt end) |> Stream.take(1)

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Get available Python packages in the environment.

  ## Parameters
  - `category`: Category for environment isolation

  ## Returns
  - `{:ok, packages}` list of installed packages
  - `{:error, reason}` on failure
  """
  def get_installed_packages(category \\ :default) do
    code = """
    import pkg_resources
    installed_packages = [str(d) for d in pkg_resources.working_set]
    installed_packages
    """

    execute_python(code, %{}, category)
  end

  @doc """
  Clean up Python environment for a specific category.

  ## Parameters
  - `category`: Category to clean up

  ## Returns
  - `:ok`
  """
  def cleanup_environment(category \\ :default) do
    init_key = String.to_atom("axiom_ai_python_initialized_#{category}")
    globals_key = String.to_atom("axiom_ai_python_globals_#{category}")

    Process.delete(init_key)
    Process.delete(globals_key)

    :ok
  end

  # Private helper functions

  defp create_toml_config(python_deps) do
    # Convert list of dependencies to TOML format
    deps_string =
      case python_deps do
        [] ->
          ""

        deps when is_list(deps) ->
          deps
          |> Enum.map(&format_dependency/1)
          |> Enum.join("\n")

        deps when is_binary(deps) ->
          deps
      end

    """
    [project]
    name = "axiom-ai-python"
    version = "0.1.0"
    description = "Python environment for AxiomAI"
    dependencies = [
    #{deps_string}
    ]

    [tool.uv]
    dev-dependencies = []
    """
  end

  defp format_dependency(dep) when is_binary(dep) do
    "    \"#{dep}\","
  end

  defp format_dependency({package, version}) when is_binary(package) and is_binary(version) do
    "    \"#{package}>#{version}\","
  end

  defp format_dependency(_), do: ""

  defp get_process_globals(category, default \\ %{}) do
    globals_key = String.to_atom("axiom_ai_python_globals_#{category}")
    Process.get(globals_key, default)
  end

  defp put_process_globals(globals, category) do
    globals_key = String.to_atom("axiom_ai_python_globals_#{category}")
    Process.put(globals_key, globals)
  end
end
