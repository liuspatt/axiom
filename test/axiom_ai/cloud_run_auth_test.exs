defmodule AxiomAi.CloudRunAuthTest do
  use ExUnit.Case
  import Mock

  alias AxiomAi.Auth

  @moduletag :cloud_run_auth

  describe "Cloud Run metadata service authentication" do
    test "get_gcp_token uses metadata service in Cloud Run environment" do
      # Mock successful metadata service response
      mock_response = %{
        status_code: 200,
        body: Jason.encode!(%{"access_token" => "test-metadata-token", "expires_in" => 3600})
      }

      with_mock HTTPoison, [
        get: fn(url, headers, _opts) ->
          assert url == "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
          assert {"Metadata-Flavor", "Google"} in headers
          {:ok, mock_response}
        end
      ] do
        # Test with empty config (should trigger ADC/metadata service)
        config = %{project_id: "test-project"}
        
        assert {:ok, "test-metadata-token"} = Auth.get_gcp_token(config)
        
        # Verify the metadata service was called
        assert called(HTTPoison.get(:_, :_, :_))
      end
    end

    test "falls back to gcloud when metadata service is not available" do
      # Mock metadata service failure (like in local development)
      with_mocks([
        {HTTPoison, [], [
          get: fn(_url, _headers, _opts) ->
            {:error, %HTTPoison.Error{reason: :nxdomain}}
          end
        ]},
        {System, [], [
          cmd: fn("gcloud", ["auth", "application-default", "print-access-token"]) ->
            {"test-gcloud-token\n", 0}
          end
        ]}
      ]) do
        config = %{project_id: "test-project"}
        
        assert {:ok, "test-gcloud-token"} = Auth.get_gcp_token(config)
        
        # Verify both metadata service and gcloud were called
        assert called(HTTPoison.get(:_, :_, :_))
        assert called(System.cmd("gcloud", ["auth", "application-default", "print-access-token"]))
      end
    end

    test "handles metadata service errors gracefully" do
      # Mock metadata service HTTP error
      mock_response = %{
        status_code: 404,
        body: "Not found"
      }

      with_mocks([
        {HTTPoison, [], [
          get: fn(_url, _headers, _opts) ->
            {:ok, mock_response}
          end
        ]},
        {System, [], [
          cmd: fn("gcloud", ["auth", "application-default", "print-access-token"]) ->
            {"fallback-token\n", 0}
          end
        ]}
      ]) do
        config = %{project_id: "test-project"}
        
        assert {:ok, "fallback-token"} = Auth.get_gcp_token(config)
      end
    end

    test "prioritizes explicit credentials over metadata service" do
      config = %{
        project_id: "test-project",
        access_token: "explicit-token"
      }
      
      # Should use explicit token without calling metadata service
      assert {:ok, "explicit-token"} = Auth.get_gcp_token(config)
    end
  end
end