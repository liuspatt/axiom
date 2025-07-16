defmodule AxiomAi.Application do
  use Application

  def start(_type, _args) do
    enable_sigchld()

    children = [
      PythonInterface.Janitor
    ]

    opts = [strategy: :one_for_one, name: AxiomAi.Supervisor]

    with {:ok, result} <- Supervisor.start_link(children, opts) do
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
