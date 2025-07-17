#!/usr/bin/env elixir

# Environment Setup Script for AxiomAI
# This script helps set up and manage Python environments for different AI models

defmodule EnvironmentSetup do
  alias AxiomAi.EnvironmentManager

  def main(args) do
    case args do
      [] ->
        show_help()
      
      ["list"] ->
        list_environments()
      
      ["create", category] ->
        create_environment(String.to_atom(category))
      
      ["remove", category] ->
        remove_environment(String.to_atom(category))
      
      ["info", category] ->
        show_environment_info(String.to_atom(category))
      
      ["load", category] ->
        load_environment(String.to_atom(category))
      
      ["status"] ->
        show_status()
      
      _ ->
        show_help()
    end
  end

  defp show_help do
    IO.puts """
    AxiomAI Environment Setup Script
    
    Usage: elixir scripts/environment_setup.exs <command> [options]
    
    Commands:
      list                    List all available environments
      create <category>       Create a new environment for a category
      remove <category>       Remove an environment
      info <category>         Show information about an environment
      load <category>         Load an environment
      status                  Show status of all environments
    
    Available Categories:
      qwen, llama, mistral, whisper, vision, embeddings
    
    Examples:
      elixir scripts/environment_setup.exs list
      elixir scripts/environment_setup.exs info qwen
      elixir scripts/environment_setup.exs load whisper
    """
  end

  defp list_environments do
    IO.puts "Available Environments:"
    IO.puts "======================"
    
    environments = EnvironmentManager.list_environments()
    
    if environments == [] do
      IO.puts "No environments found."
    else
      Enum.each(environments, fn env ->
        case EnvironmentManager.get_environment_info(env) do
          {:ok, info} ->
            status = if EnvironmentManager.environment_loaded?(env), do: "✓ Loaded", else: "○ Not loaded"
            IO.puts "#{env}: #{info.dependency_count} dependencies [#{status}]"
          
          {:error, reason} ->
            IO.puts "#{env}: Error - #{inspect(reason)}"
        end
      end)
    end
  end

  defp create_environment(category) do
    IO.puts "Creating environment for category: #{category}"
    
    # Default dependencies based on category
    dependencies = case category do
      :qwen -> [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
        "numpy>=1.24.0",
        "huggingface_hub>=0.15.0"
      ]
      :llama -> [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
        "numpy>=1.24.0",
        "huggingface_hub>=0.15.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0"
      ]
      :whisper -> [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.9.0",
        "numpy>=1.24.0",
        "huggingface_hub>=0.15.0"
      ]
      :vision -> [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "pillow>=9.0.0",
        "timm>=0.6.0",
        "opencv-python>=4.5.0",
        "numpy>=1.24.0",
        "huggingface_hub>=0.15.0"
      ]
      :embeddings -> [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.0.0",
        "huggingface_hub>=0.15.0"
      ]
      _ -> [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0"
      ]
    end
    
    case EnvironmentManager.create_environment(category, dependencies) do
      :ok ->
        IO.puts "✓ Environment created successfully!"
        IO.puts "  Category: #{category}"
        IO.puts "  Dependencies: #{length(dependencies)}"
      
      {:error, reason} ->
        IO.puts "✗ Failed to create environment: #{inspect(reason)}"
    end
  end

  defp remove_environment(category) do
    IO.puts "Removing environment for category: #{category}"
    
    case EnvironmentManager.remove_environment(category) do
      :ok ->
        IO.puts "✓ Environment removed successfully!"
      
      {:error, reason} ->
        IO.puts "✗ Failed to remove environment: #{inspect(reason)}"
    end
  end

  defp show_environment_info(category) do
    case EnvironmentManager.get_environment_info(category) do
      {:ok, info} ->
        IO.puts "Environment Information: #{category}"
        IO.puts "==============================="
        IO.puts "Category: #{info.category}"
        IO.puts "Requirements File: #{info.requirements_file}"
        IO.puts "Python Version: #{info.python_version}"
        IO.puts "Dependencies: #{info.dependency_count}"
        IO.puts "Loaded: #{if EnvironmentManager.environment_loaded?(category), do: "Yes", else: "No"}"
        IO.puts ""
        IO.puts "Dependencies:"
        Enum.each(info.dependencies, fn dep ->
          IO.puts "  - #{dep}"
        end)
      
      {:error, reason} ->
        IO.puts "✗ Error getting environment info: #{inspect(reason)}"
    end
  end

  defp load_environment(category) do
    IO.puts "Loading environment: #{category}"
    
    case EnvironmentManager.load_environment(category) do
      {:ok, deps} ->
        IO.puts "✓ Environment loaded successfully!"
        IO.puts "  Dependencies loaded: #{length(deps)}"
      
      {:error, reason} ->
        IO.puts "✗ Failed to load environment: #{inspect(reason)}"
    end
  end

  defp show_status do
    IO.puts "Environment Status"
    IO.puts "=================="
    
    environments = EnvironmentManager.list_environments()
    
    if environments == [] do
      IO.puts "No environments configured."
    else
      Enum.each(environments, fn env ->
        loaded = EnvironmentManager.environment_loaded?(env)
        status_icon = if loaded, do: "✓", else: "○"
        status_text = if loaded, do: "Loaded", else: "Not loaded"
        
        case EnvironmentManager.get_environment_info(env) do
          {:ok, info} ->
            IO.puts "#{status_icon} #{env}: #{info.dependency_count} deps [#{status_text}]"
          
          {:error, _reason} ->
            IO.puts "✗ #{env}: Error loading info"
        end
      end)
    end
  end
end

# Start the application if we're in a Mix project
if Code.ensure_loaded?(Mix) do
  Mix.install([])
end

# Run the main function
EnvironmentSetup.main(System.argv())