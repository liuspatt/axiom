defmodule AxiomAi.PythonInterfaceTest do
  use ExUnit.Case, async: false

  alias AxiomAi.PythonInterface

  describe "PythonInterface integration tests" do
    test "init_environment initializes Python with dependencies" do
      deps = ["numpy", "requests"]
      
      assert :ok = PythonInterface.init_environment(deps, :test_category)
      
      # Should not fail if called again
      assert :ok = PythonInterface.init_environment(deps, :test_category)
      
      # Clean up
      PythonInterface.cleanup_environment(:test_category)
    end

    test "execute_python runs simple Python code" do
      # Initialize environment first
      :ok = PythonInterface.init_environment([], :test_basic_python)
      
      # Test basic Python execution
      code = """
      result = 2 + 2
      result
      """
      
      assert {:ok, 4} = PythonInterface.execute_python(code, %{}, :test_basic_python)
      
      # Clean up
      PythonInterface.cleanup_environment(:test_basic_python)
    end

    test "execute_python maintains globals between calls" do
      # Initialize environment first
      :ok = PythonInterface.init_environment([], :test_globals)
      
      # First call sets a variable
      code1 = """
      x = 10
      x
      """
      
      assert {:ok, 10} = PythonInterface.execute_python(code1, %{}, :test_globals)
      
      # Second call uses the variable
      code2 = """
      y = x * 2
      y
      """
      
      assert {:ok, 20} = PythonInterface.execute_python(code2, %{}, :test_globals)
      
      # Clean up
      PythonInterface.cleanup_environment(:test_globals)
    end

    test "execute_inference runs AI model inference simulation" do
      # Initialize environment first
      :ok = PythonInterface.init_environment([], :test_inference)
      
      model_path = "test/model/path"
      message = "Hello, world!"
      
      python_code = """
      def generate_response(model_path, message, max_tokens, temperature):
          # Simple mock response for testing
          return f"Model at {model_path} says: {message} (max_tokens: {max_tokens}, temp: {temperature})"
      """
      
      config = %{max_tokens: 100, temperature: 0.7}
      
      assert {:ok, response} = PythonInterface.execute_inference(
        model_path, 
        message, 
        python_code, 
        config, 
        :test_inference
      )
      
      assert response =~ "Model at test/model/path says: Hello, world!"
      assert response =~ "max_tokens: 100"
      assert response =~ "temp: 0.7"
      
      # Clean up
      PythonInterface.cleanup_environment(:test_inference)
    end

    test "get_installed_packages returns package list" do
      # Initialize environment with some packages
      :ok = PythonInterface.init_environment(["requests"], :test_packages)
      
      assert {:ok, packages} = PythonInterface.get_installed_packages(:test_packages)
      assert is_list(packages)
      
      # Clean up
      PythonInterface.cleanup_environment(:test_packages)
    end

    test "cleanup_environment cleans up properly" do
      # Initialize environment
      :ok = PythonInterface.init_environment([], :test_cleanup)
      
      # Set some variables
      code = """
      cleanup_test_var = 42
      cleanup_test_var
      """
      
      assert {:ok, 42} = PythonInterface.execute_python(code, %{}, :test_cleanup)
      
      # Clean up
      assert :ok = PythonInterface.cleanup_environment(:test_cleanup)
      
      # After cleanup, environment should be reinitialized
      assert :ok = PythonInterface.init_environment([], :test_cleanup)
      
      # Previous variables should not exist
      code2 = """
      try:
          cleanup_test_var
      except NameError:
          "variable_not_found"
      """
      
      assert {:ok, "variable_not_found"} = PythonInterface.execute_python(code2, %{}, :test_cleanup)
      
      # Clean up
      PythonInterface.cleanup_environment(:test_cleanup)
    end

    test "different categories maintain separate environments" do
      # Initialize two different categories
      :ok = PythonInterface.init_environment([], :cat1)
      :ok = PythonInterface.init_environment([], :cat2)
      
      # Set different variables in each category
      code1 = """
      category_var = "category1"
      category_var
      """
      
      code2 = """
      category_var = "category2"
      category_var
      """
      
      assert {:ok, "category1"} = PythonInterface.execute_python(code1, %{}, :cat1)
      assert {:ok, "category2"} = PythonInterface.execute_python(code2, %{}, :cat2)
      
      # Verify they remain separate
      code_check = """
      category_var
      """
      
      assert {:ok, "category1"} = PythonInterface.execute_python(code_check, %{}, :cat1)
      assert {:ok, "category2"} = PythonInterface.execute_python(code_check, %{}, :cat2)
      
      # Clean up
      PythonInterface.cleanup_environment(:cat1)
      PythonInterface.cleanup_environment(:cat2)
    end

    test "execute_streaming_inference returns a stream" do
      # Initialize environment first
      :ok = PythonInterface.init_environment([], :test_streaming)
      
      model_path = "test/streaming/model"
      message = "Stream this!"
      
      python_code = """
      def generate_response(model_path, message, max_tokens, temperature):
          return f"Streaming from {model_path}: {message}"
      """
      
      config = %{max_tokens: 50, temperature: 0.5}
      
      stream = PythonInterface.execute_streaming_inference(
        model_path, 
        message, 
        python_code, 
        config, 
        :test_streaming
      )
      
      # Should be a stream
      assert is_function(stream)
      
      # Take first response
      responses = Enum.take(stream, 1)
      assert [response] = responses
      assert response =~ "Streaming from test/streaming/model: Stream this!"
      
      # Clean up
      PythonInterface.cleanup_environment(:test_streaming)
    end

    test "handles Python execution errors gracefully" do
      # Initialize environment first
      :ok = PythonInterface.init_environment([], :test_errors)
      
      # Code that will cause an error
      code = """
      raise ValueError("Test error")
      """
      
      assert {:error, {:python_execution_error, _message}} = 
        PythonInterface.execute_python(code, %{}, :test_errors)
      
      # Clean up
      PythonInterface.cleanup_environment(:test_errors)
    end

    test "handles inference errors gracefully" do
      # Initialize environment first
      :ok = PythonInterface.init_environment([], :test_inference_errors)
      
      model_path = "test/model"
      message = "Test message"
      
      # Python code that will cause an error
      python_code = """
      def generate_response(model_path, message, max_tokens, temperature):
          raise RuntimeError("Model loading failed")
      """
      
      config = %{max_tokens: 100, temperature: 0.7}
      
      assert {:error, {:inference_error, _message}} = 
        PythonInterface.execute_inference(
          model_path, 
          message, 
          python_code, 
          config, 
          :test_inference_errors
        )
      
      # Clean up
      PythonInterface.cleanup_environment(:test_inference_errors)
    end
  end
end