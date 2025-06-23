defmodule AxiomAi.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/liuspatt/axiom-ai"

  def project do
    [
      app: :axiom_ai,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package(),
      docs: docs(),
      name: "AxiomAI",
      source_url: @source_url
    ]
  end

  def application do
    [
      extra_applications: [:logger, :httpoison]
    ]
  end

  defp deps do
    [
      {:httpoison, "~> 2.0"},
      {:jason, "~> 1.4"},
      {:joken, "~> 2.6"},
      {:ex_aws, "~> 2.5"},
      {:hackney, "~> 1.9"},
      {:sweet_xml, "~> 0.7"},
      {:pythonx, "~> 0.4.0"},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false},
      {:mox, "~> 1.0", only: :test},
      {:mock, "~> 0.3.0", only: :test}
    ]
  end

  defp description do
    """
    A unified Elixir client for multiple AI providers including Vertex AI, OpenAI,
    Anthropic, AWS Bedrock, and local PyTorch models.
    """
  end

  defp package do
    [
      name: "axiom_ai",
      files: ~w(lib .formatter.exs mix.exs README* LICENSE*),
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      maintainers: ["Luis Patt"]
    ]
  end

  defp docs do
    [
      main: "readme",
      source_ref: "v#{@version}",
      source_url: @source_url,
      extras: ["README.md"]
    ]
  end
end
