defmodule PythonInterface.Error do
  @moduledoc """
  An exception raised when Python raises an exception.
  """

  defexception [:type, :value, :traceback]

  @type t :: %{
          type: PythonInterface.Object.t(),
          value: PythonInterface.Object.t(),
          traceback: PythonInterface.Object.t()
        }

  @impl true
  def message(error) do
    lines = PythonInterface.NIF.format_exception(error)

    lines =
      Enum.map(lines, fn line ->
        ["        ", line]
      end)

    IO.iodata_to_binary(["Python exception raised\n\n", lines])
  end
end
