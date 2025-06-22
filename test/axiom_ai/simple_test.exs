defmodule AxiomAi.SimpleTest do
  use ExUnit.Case
  alias AxiomAi.{Config, Client}

  describe "basic functionality" do
    test "creates a client with vertex ai config" do
      config = %{project_id: "test-project"}

      client = AxiomAi.new(:vertex_ai, config)

      assert %Client{} = client
      assert client.provider == :vertex_ai
      assert client.config.project_id == "test-project"
      assert client.config.region == "us-central1"
      assert client.config.model == "gemini-1.5-pro"
    end

    test "validates configuration correctly" do
      # Valid config
      config = %{project_id: "test-project"}
      result = Config.validate(:vertex_ai, config)

      assert result.project_id == "test-project"
      assert result.region == "us-central1"
      assert result.model == "gemini-1.5-pro"
    end

    test "validates service account configuration" do
      service_account_key = %{
        "type" => "service_account",
        "client_email" => "test@test-project.iam.gserviceaccount.com",
        "private_key" => "-----BEGIN PRIVATE KEY-----\ntest-key\n-----END PRIVATE KEY-----\n"
      }

      config = %{
        project_id: "test-project",
        service_account_key: service_account_key
      }

      result = Config.validate(:vertex_ai, config)

      assert result.project_id == "test-project"
      assert result.service_account_key == service_account_key
    end

    test "validates access token configuration" do
      config = %{
        project_id: "test-project",
        access_token: "test-token"
      }

      result = Config.validate(:vertex_ai, config)

      assert result.project_id == "test-project"
      assert result.access_token == "test-token"
    end

    test "raises error for missing project_id" do
      config = %{}

      assert_raise ArgumentError, ~r/Missing required configuration keys: \[:project_id\]/, fn ->
        Config.validate(:vertex_ai, config)
      end
    end

    test "raises error for invalid service account key" do
      service_account_key = %{
        "type" => "invalid_type",
        "client_email" => "test@test-project.iam.gserviceaccount.com",
        "private_key" => "-----BEGIN PRIVATE KEY-----\ntest-key\n-----END PRIVATE KEY-----\n"
      }

      config = %{
        project_id: "test-project",
        service_account_key: service_account_key
      }

      assert_raise ArgumentError, ~r/Invalid service account key type: invalid_type/, fn ->
        Config.validate(:vertex_ai, config)
      end
    end

    test "raises error for unsupported provider" do
      assert_raise ArgumentError, ~r/Unsupported provider: :invalid/, fn ->
        Config.validate(:invalid, %{})
      end
    end
  end

  describe "auth module" do
    test "returns direct access token when provided" do
      config = %{access_token: "direct-token"}

      assert {:ok, "direct-token"} = AxiomAi.Auth.get_gcp_token(config)
    end

    test "returns error for invalid service account key" do
      config = %{service_account_key: "invalid-key"}

      assert {:error, :invalid_service_account_key} = AxiomAi.Auth.get_gcp_token(config)
    end

    test "returns error for missing private key" do
      service_account_key = %{
        "type" => "service_account",
        "client_email" => "test@test-project.iam.gserviceaccount.com"
        # Missing private_key
      }

      config = %{service_account_key: service_account_key}

      assert {:error, :missing_private_key} = AxiomAi.Auth.get_gcp_token(config)
    end

    test "returns error for non-existent file" do
      config = %{service_account_path: "/non/existent/path.json"}

      assert {:error, {:file_read_error, :enoent}} = AxiomAi.Auth.get_gcp_token(config)
    end
  end
end
