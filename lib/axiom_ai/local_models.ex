defmodule AxiomAi.LocalModels do
  @moduledoc """
  Modular configuration system for local AI models.
  Supports multiple model categories with extensible templates and configurations.
  """

  alias AxiomAi.LocalModels.{Registry, Templates, Categories}

  @doc """
  Returns all available model configurations organized by category.
  """
  @spec get_predefined_models() :: %{String.t() => map()}
  def get_predefined_models do
    Registry.get_all_models()
  end

  @doc """
  Gets a specific model configuration by key.
  """
  @spec get_model_config(String.t()) :: {:ok, map()} | {:error, :not_found}
  def get_model_config(model_key) do
    Registry.get_model(model_key)
  end

  @doc """
  Lists all available predefined model keys.
  """
  @spec list_models() :: [String.t()]
  def list_models do
    Registry.list_all_models()
  end

  @doc """
  Lists models by category.
  """
  @spec list_models_by_category(atom()) :: [String.t()]
  def list_models_by_category(category) do
    Registry.list_models_by_category(category)
  end

  @doc """
  Gets all available categories.
  """
  @spec list_categories() :: [atom()]
  def list_categories do
    Categories.list_all()
  end

  @doc """
  Creates a custom model configuration using a template.
  """
  @spec create_custom_config(atom(), map()) :: map()
  def create_custom_config(template_type, overrides \\ %{}) do
    Templates.create_from_template(template_type, overrides)
  end

  @doc """
  Registers a new model configuration.
  """
  @spec register_model(String.t(), atom(), map()) :: :ok | {:error, term()}
  def register_model(model_key, category, config) do
    Registry.add_model(model_key, category, config)
  end
end
