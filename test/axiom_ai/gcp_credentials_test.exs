defmodule AxiomAi.GcpCredentialsTest do
  use ExUnit.Case
  # import Mock

  @moduletag :gcp_credentials

  describe "GCP Vertex AI with real credentials" do
    test "chat with real GCP credentials" do

      service_account_path = "test/fixtures/credentials.json"
      {:ok, creds} = File.read(service_account_path)
      {:ok, creds_map} = Jason.decode(creds)
      project_id = creds_map["project_id"]

      if project_id && service_account_path do
        config = %{
          project_id: project_id,
          service_account_path: service_account_path,
          model: "gemini-2.0-flash"
        }

        client = AxiomAi.new(:vertex_ai, config)

        assert {:ok, response} = AxiomAi.chat(client, "Hello! Please respond with just 'Hi there!'")
        assert is_binary(response.response)
        assert String.length(response.response) > 0
      else
        IO.puts("Skipping GCP credentials test - missing environment variables:")
        IO.puts("  GCP_PROJECT_ID: #{project_id}")
        IO.puts("  GCP_SERVICE_ACCOUNT_PATH: #{service_account_path}")
      end
    end
  end
end
