defmodule PythonInterface.Object do
  @moduledoc """
  A struct holding a Python object.

  This is an opaque struct used to pass Python objects around in
  Elixir code.
  """

  defstruct [:resource]

  @type t :: %__MODULE__{resource: reference()}
end

defimpl Inspect, for: PythonInterface.Object do
  import Inspect.Algebra

  alias PythonInterface.Object

  def inspect(%Object{} = object, _opts) do
    repr_string =
      object
      |> PythonInterface.NIF.object_repr()
      |> PythonInterface.NIF.unicode_to_string()

    repr_lines = String.split(repr_string, "\n")
    inner = Enum.map_intersperse(repr_lines, line(), &string/1)

    force_unfit(
      concat([
        "#PythonInterface.Object<",
        nest(concat([line() | inner]), 2),
        line(),
        ">"
      ])
    )
  end
end
