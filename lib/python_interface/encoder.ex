defprotocol PythonInterface.Encoder do
  @moduledoc ~S'''
  A protocol for converting data structures to Python objects.

  The protocol has implementation for Elixir built-in data types.

  In order to define it for other data structures, you can use `PythonInterface.eval/2`
  and pass necessary information as built-in types. For example, imagine
  we have the following struct representing a complex number:

      defmodule Complex do
        defstruct [:re, :im]
      end

  The protocol implementation could look like this:

      defimpl PythonInterface.Encoder, for: Complex do
        def encode(complex, _encoder) do
          {result, %{}} =
            PythonInterface.eval(
              """
              complex(re, im)
              """,
              %{"re" => complex.re, "im" => complex.im}
            )

          result
        end
      end

      PythonInterface.encode!(%Complex{re: 1, im: -1})
      #=> #PythonInterface.Object<
      #=>   (1-1j)
      #=> >

  When dealing with more complex data structures, you will want to
  return an object from a Python package. In that case, it is a good
  idea to raise a clear error if the package is not installed. For
  example, here is one possible implementation for `Explorer.DataFrame`:

      defimpl PythonInterface.Encoder, for: Explorer.DataFrame do
        def encode(df, _encoder) do
          {result, %{}} =
            PythonInterface.eval(
              """
              try:
                import polars
                result = polars.read_ipc(ipc)
              except ModuleNotFoundError:
                result = None

              result
              """,
              %{"ipc" => Explorer.DataFrame.dump_ipc!(df)}
            )

          case PythonInterface.decode(result) do
            %PythonInterface.Object{} ->
              result

            nil ->
              raise Protocol.UndefinedError,
                protocol: @protocol,
                value: df,
                description:
                  "cannot encode Explorer.DataFrame, because the polars Python package is not installed"
          end
        end
      end

  '''

  @doc """
  A function invoked to encode the given term to `PythonInterface.Object`.
  """
  @spec encode(term :: term(), PythonInterface.encoder()) :: PythonInterface.Object.t()
  def encode(term, encoder)
end

defimpl PythonInterface.Encoder, for: PythonInterface.Object do
  def encode(object, _encoder) do
    object
  end
end

defimpl PythonInterface.Encoder, for: Atom do
  def encode(nil, _encoder) do
    PythonInterface.NIF.none_new()
  end

  def encode(false, _encoder) do
    PythonInterface.NIF.false_new()
  end

  def encode(true, _encoder) do
    PythonInterface.NIF.true_new()
  end

  def encode(term, _encoder) do
    term
    |> Atom.to_string()
    |> PythonInterface.NIF.unicode_from_string()
  end
end

defimpl PythonInterface.Encoder, for: Float do
  def encode(term, _encoder) do
    PythonInterface.NIF.float_new(term)
  end
end

defimpl PythonInterface.Encoder, for: Integer do
  @max_int64 2 ** 63 - 1
  @min_int64 Kernel.-(2 ** 63)

  def encode(term, _encoder) when @min_int64 <= term and term <= @max_int64 do
    PythonInterface.NIF.long_from_int64(term)
  end

  def encode(term, _encoder) do
    # Technically we could do an object call on Python int.from_bytes,
    # however given that this is a rare path (integers over 64 bits)
    # and that Python C API has a specific function to create integer
    # from string, we pick this most straightforward option.
    PythonInterface.NIF.long_from_string(Integer.to_string(term, 36), 36)
  end
end

defimpl PythonInterface.Encoder, for: BitString do
  def encode(term, _encoder) when is_binary(term) do
    PythonInterface.NIF.bytes_from_binary(term)
  end

  def encode(term, _encoder) do
    raise Protocol.UndefinedError,
      protocol: @protocol,
      value: term,
      description: "cannot encode a bitstring as a Python object"
  end
end

defimpl PythonInterface.Encoder, for: Map do
  def encode(term, encoder) do
    dict = PythonInterface.NIF.dict_new()

    for {key, value} <- term do
      PythonInterface.NIF.dict_set_item(dict, encoder.(key, encoder), encoder.(value, encoder))
    end

    dict
  end
end

defimpl PythonInterface.Encoder, for: Tuple do
  def encode(term, encoder) do
    size = tuple_size(term)

    tuple = PythonInterface.NIF.tuple_new(size)

    for index <- 0..(size - 1)//1 do
      value = encoder.(elem(term, index), encoder)
      PythonInterface.NIF.tuple_set_item(tuple, index, value)
    end

    tuple
  end
end

defimpl PythonInterface.Encoder, for: List do
  def encode(term, encoder) do
    # Note that to compute length we need to traverse the list, but
    # otherwise we cannot preallocate the Python list and we would
    # need to use append (which could result in many reallocations).
    size = length(term)

    list = PythonInterface.NIF.list_new(size)

    Enum.with_index(term, fn item, index ->
      value = encoder.(item, encoder)
      PythonInterface.NIF.list_set_item(list, index, value)
    end)

    list
  end
end

defimpl PythonInterface.Encoder, for: MapSet do
  def encode(term, encoder) do
    set = PythonInterface.NIF.set_new()

    for item <- term do
      key = encoder.(item, encoder)
      PythonInterface.NIF.set_add(set, key)
    end

    set
  end
end
