defmodule AxiomAi.Application do
  use Application

  def start(_type, _args) do
    enable_sigchld()

    IO.puts("Starting AxiomAi.Application...")

    children = [
      # Start PythonInterface.Janitor under PythonInterface.Supervisor to match expected structure
      %{
        id: Elixir.PythonInterface.Supervisor,
        start: {Supervisor, :start_link, [[Elixir.PythonInterface.Janitor], [strategy: :one_for_one, name: Elixir.PythonInterface.Supervisor]]},
        type: :supervisor
      }
    ]

    opts = [strategy: :one_for_one, name: AxiomAi.Supervisor]

    with {:ok, result} <- Supervisor.start_link(children, opts) do
      IO.puts("AxiomAi.Application started successfully")
      
      # Check if PythonInterface.Supervisor is running
      supervisor_pid = Process.whereis(Elixir.PythonInterface.Supervisor)
      IO.puts("PythonInterface.Supervisor pid after startup: #{inspect(supervisor_pid)}")
      
      maybe_uv_init()

      {:ok, result}
    end
  end

  defp maybe_uv_init(), do: :noop

  defp enable_sigchld() do
    # Some APIs in Python, such as subprocess.run, wait for a child
    # OS process to finish. On Unix, this relies on `waitpid` C API,
    # which does not work properly if SIGCHLD is ignored, resulting
    # in infinite waiting. ERTS ignores the signal by default, so we
    # explicitly restore the default handler.
    case :os.type() do
      {:win32, _osname} -> :ok
      {:unix, _osname} -> :os.set_signal(:sigchld, :default)
    end
  end
end
