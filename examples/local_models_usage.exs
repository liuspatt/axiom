# Example usage of the new modular local models system

alias AxiomAi.LocalModels

# List all available models
IO.puts("=== All Available Models ===")
LocalModels.list_models()
|> Enum.each(fn model ->
  {:ok, config} = LocalModels.get_model_config(model)
  IO.puts("#{model}: #{config.name} (#{config.category})")
end)

# List models by category
IO.puts("\n=== Text Generation Models ===")
LocalModels.list_models_by_category(:text_generation)
|> Enum.each(&IO.puts/1)

IO.puts("\n=== Code Generation Models ===")
LocalModels.list_models_by_category(:code_generation)
|> Enum.each(&IO.puts/1)

IO.puts("\n=== HTTP Endpoint Models ===")
LocalModels.list_models_by_category(:http_endpoints)
|> Enum.each(&IO.puts/1)

# Create a custom model using templates
IO.puts("\n=== Creating Custom Models ===")

# Custom text generation model
custom_text_model = LocalModels.create_custom_config(:python_interface_text, %{
  name: "Custom Qwen Model",
  model_path: "Qwen/Custom-Model-Path",
  description: "My custom text generation model"
})

IO.puts("Custom text model: #{inspect(custom_text_model)}")

# Custom HTTP endpoint model
custom_http_model = LocalModels.create_custom_config(:http_openai, %{
  name: "Custom API Model",
  endpoint: "http://my-server:8080",
  model: "my-custom-model",
  description: "Custom API-based model"
})

IO.puts("Custom HTTP model: #{inspect(custom_http_model)}")

# Register a new model at runtime
IO.puts("\n=== Registering New Model ===")
new_model_config = %{
  name: "My Runtime Model",
  type: :http,
  endpoint: "http://localhost:9000",
  model: "runtime-model",
  api_format: :openai,
  description: "Model added at runtime"
}

LocalModels.register_model("my-runtime-model", :text_generation, new_model_config)

# Verify the new model was added
{:ok, runtime_model} = LocalModels.get_model_config("my-runtime-model")
IO.puts("Runtime model registered: #{runtime_model.name}")

# List all categories
IO.puts("\n=== Available Categories ===")
LocalModels.list_categories()
|> Enum.each(fn category ->
  description = AxiomAi.LocalModels.Categories.get_description(category)
  IO.puts("#{category}: #{description}")
end)