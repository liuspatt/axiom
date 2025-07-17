defmodule AxiomAi.EnvironmentManager do
  @moduledoc """
  Environment Manager for handling different Python environments for various AI models.

  This module provides functionality to create, manage, and switch between different
  Python environments tailored for specific AI model categories.
  """

  alias AxiomAi.PythonInterface

  @environments_dir "environments"
  @default_python_version "3.10"

  @doc """
  Load environment dependencies for a specific model category.

  ## Parameters
  - `category`: Model category (e.g., :qwen, :llama, :whisper, :vision, :embeddings)
  - `force_reload`: Whether to force reload the environment (default: false)

  ## Returns
  - `{:ok, deps}` with list of dependencies
  - `{:error, reason}` on failure
  """
  def load_environment(category, force_reload \\ false) do
    category_str = Atom.to_string(category)
    requirements_file = Path.join([@environments_dir, category_str, "requirements.txt"])

    case File.exists?(requirements_file) do
      true ->
        case File.read(requirements_file) do
          {:ok, content} ->
            deps = parse_requirements(content)

            if force_reload do
              PythonInterface.cleanup_environment(category)
            end

            case PythonInterface.init_environment(deps, category) do
              :ok -> {:ok, deps}
              {:error, reason} -> {:error, reason}
            end

          {:error, reason} ->
            {:error, {:file_read_error, reason}}
        end

      false ->
        {:error, {:missing_requirements_file, requirements_file}}
    end
  end

  @doc """
  Get the list of available model environments.

  ## Returns
  - List of available environment categories
  """
  def list_environments do
    case File.ls(@environments_dir) do
      {:ok, dirs} ->
        dirs
        |> Enum.filter(fn dir ->
          requirements_file = Path.join([@environments_dir, dir, "requirements.txt"])
          File.exists?(requirements_file)
        end)
        |> Enum.map(&String.to_atom/1)

      {:error, _reason} ->
        []
    end
  end

  @doc """
  Get environment information for a specific category.

  ## Parameters
  - `category`: Model category

  ## Returns
  - `{:ok, info}` with environment information
  - `{:error, reason}` on failure
  """
  def get_environment_info(category) do
    category_str = Atom.to_string(category)
    requirements_file = Path.join([@environments_dir, category_str, "requirements.txt"])

    case File.exists?(requirements_file) do
      true ->
        case File.read(requirements_file) do
          {:ok, content} ->
            deps = parse_requirements(content)

            info = %{
              category: category,
              requirements_file: requirements_file,
              dependencies: deps,
              dependency_count: length(deps),
              python_version: @default_python_version
            }

            {:ok, info}

          {:error, reason} ->
            {:error, {:file_read_error, reason}}
        end

      false ->
        {:error, {:missing_requirements_file, requirements_file}}
    end
  end

  @doc """
  Create a new environment configuration.

  ## Parameters
  - `category`: Model category
  - `dependencies`: List of Python dependencies
  - `python_version`: Python version (optional)

  ## Returns
  - `:ok` on success
  - `{:error, reason}` on failure
  """
  def create_environment(category, dependencies, python_version \\ @default_python_version) do
    category_str = Atom.to_string(category)
    env_dir = Path.join([@environments_dir, category_str])
    requirements_file = Path.join([env_dir, "requirements.txt"])

    # Create environment directory
    case File.mkdir_p(env_dir) do
      :ok ->
        # Create requirements.txt
        requirements_content = format_requirements(dependencies)

        case File.write(requirements_file, requirements_content) do
          :ok ->
            # Create environment info file
            info_content = """
            # Environment: #{category}
            # Python Version: #{python_version}
            # Created: #{DateTime.utc_now() |> DateTime.to_string()}
            # Dependencies: #{length(dependencies)}
            """

            info_file = Path.join([env_dir, "environment_info.txt"])
            File.write(info_file, info_content)

            :ok

          {:error, reason} ->
            {:error, {:file_write_error, reason}}
        end

      {:error, reason} ->
        {:error, {:directory_creation_error, reason}}
    end
  end

  @doc """
  Remove an environment configuration.

  ## Parameters
  - `category`: Model category to remove

  ## Returns
  - `:ok` on success
  - `{:error, reason}` on failure
  """
  def remove_environment(category) do
    category_str = Atom.to_string(category)
    env_dir = Path.join([@environments_dir, category_str])

    # Clean up the runtime environment
    PythonInterface.cleanup_environment(category)

    # Remove the directory and its contents
    case File.rm_rf(env_dir) do
      {:ok, _files} -> :ok
      {:error, reason, _path} -> {:error, {:directory_removal_error, reason}}
    end
  end

  @doc """
  Execute code in a specific environment.

  ## Parameters
  - `category`: Model category
  - `code`: Python code to execute
  - `auto_load`: Whether to auto-load environment if not loaded (default: true)

  ## Returns
  - `{:ok, result}` on success
  - `{:error, reason}` on failure
  """
  def execute_in_environment(category, code, auto_load \\ true) do
    if auto_load do
      case load_environment(category) do
        {:ok, _deps} ->
          PythonInterface.execute_python(code, %{}, category)

        {:error, reason} ->
          {:error, reason}
      end
    else
      PythonInterface.execute_python(code, %{}, category)
    end
  end

  @doc """
  Check if an environment is loaded and ready.

  ## Parameters
  - `category`: Model category

  ## Returns
  - `true` if environment is loaded
  - `false` if not loaded
  """
  def environment_loaded?(category) do
    init_key = String.to_atom("axiom_ai_python_initialized_#{category}")

    case Process.get(init_key) do
      nil -> false
      true -> true
      _ -> false
    end
  end

  # Private helper functions

  defp parse_requirements(content) do
    content
    |> String.split("\n")
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == "" or String.starts_with?(&1, "#")))
    |> Enum.map(&parse_requirement_line/1)
    |> Enum.reject(&is_nil/1)
  end

  defp parse_requirement_line(line) do
    # Handle different requirement formats
    cond do
      String.contains?(line, ">=") ->
        [package, version] = String.split(line, ">=", parts: 2)
        "#{String.trim(package)}>#{String.trim(version)}"

      String.contains?(line, "==") ->
        [package, version] = String.split(line, "==", parts: 2)
        "#{String.trim(package)}=#{String.trim(version)}"

      String.contains?(line, "~=") ->
        [package, version] = String.split(line, "~=", parts: 2)
        "#{String.trim(package)}~#{String.trim(version)}"

      true ->
        String.trim(line)
    end
  end

  defp format_requirements(dependencies) do
    dependencies
    |> Enum.map(&format_dependency/1)
    |> Enum.join("\n")
  end

  defp format_dependency(dep) when is_binary(dep), do: dep
  defp format_dependency({package, version}), do: "#{package}>#{version}"
  defp format_dependency(_), do: ""
end
