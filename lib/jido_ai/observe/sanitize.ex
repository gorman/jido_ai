defmodule Jido.AI.Observe.Sanitize do
  @moduledoc """
  Sanitizer boundary for AI telemetry and public/tool transport payloads.

  The telemetry profile keeps metadata low-cardinality by redacting sensitive
  keys, bounding nested values, and summarizing payload-shaped fields.

  The transport profile preserves more payload detail while converting arbitrary
  terms into bounded JSON-safe data.

  Both profiles are bounded twice: by `:max_depth` for one path down the payload,
  and by `:max_bytes` for the payload as a whole. The walk counts the bytes it
  emits, so a payload built from shared subterms stops before it is expanded.
  """

  @sensitive_exact_keys MapSet.new([
                          "api_key",
                          "apikey",
                          "password",
                          "secret",
                          "token",
                          "auth_token",
                          "authtoken",
                          "private_key",
                          "privatekey",
                          "access_key",
                          "accesskey",
                          "bearer",
                          "api_secret",
                          "apisecret",
                          "client_secret",
                          "clientsecret"
                        ])

  @sensitive_contains ["secret_"]
  @sensitive_suffixes ["_secret", "_key", "_token", "_password"]

  @redacted "[REDACTED]"
  @truncated_key :__jido_ai_truncated__

  # Approximate JSON cost of each emitted node, charged while the walk runs.
  # Exact byte counts are not necessary: the budget only has to stop growth
  # before the expanded tree is built.
  @scalar_cost 8
  @quote_cost 2
  @key_cost 4
  @container_cost 2
  @summary_cost 64

  @telemetry_payload_summary_keys MapSet.new([
                                    :content,
                                    "content",
                                    :messages,
                                    "messages",
                                    :output,
                                    "output",
                                    :raw,
                                    "raw",
                                    :raw_result,
                                    "raw_result",
                                    :raw_response,
                                    "raw_response",
                                    :response,
                                    "response",
                                    :result,
                                    "result"
                                  ])

  @telemetry_defaults %{
    max_depth: 4,
    max_list_items: 10,
    max_map_entries: 32,
    max_string_chars: 512,
    max_inspect_chars: 512,
    max_bytes: 256_000
  }

  @transport_defaults %{
    max_depth: 8,
    max_list_items: 100,
    max_map_entries: 100,
    max_string_chars: 16_384,
    max_inspect_chars: 2_048,
    max_bytes: 2_000_000
  }

  @type profile :: :telemetry | :transport

  @doc """
  Redacts sensitive keys recursively without changing payload shape otherwise.
  """
  @spec sensitive(term()) :: term()
  def sensitive(payload) when is_map(payload) do
    Map.new(payload, fn {key, value} ->
      if sensitive_key?(key) do
        {key, @redacted}
      else
        {key, sensitive(value)}
      end
    end)
  end

  def sensitive(payload) when is_list(payload) do
    if proper_list?(payload), do: Enum.map(payload, &sensitive/1), else: inspect_limited(payload, 512)
  end

  def sensitive(payload), do: payload

  @doc """
  Sanitizes arbitrary payloads for a specific boundary profile.
  """
  @spec sanitize(term(), profile(), keyword()) :: term()
  def sanitize(payload, profile, opts \\ [])

  def sanitize(payload, :telemetry, opts) when is_list(opts) do
    run(payload, :telemetry, sanitize_opts(@telemetry_defaults, opts))
  end

  def sanitize(payload, :transport, opts) when is_list(opts) do
    run(payload, :transport, sanitize_opts(@transport_defaults, opts))
  end

  @doc """
  Sanitizes event metadata before it crosses the telemetry boundary.
  """
  @spec telemetry_metadata(term(), keyword()) :: term()
  def telemetry_metadata(metadata, opts \\ []), do: sanitize(metadata, :telemetry, opts)

  @doc """
  Sanitizes public/tool payloads into bounded JSON-safe data.

  The depth limit keeps nested fields, and the `:max_bytes` budget keeps the
  whole payload small. When the budget runs out, the remaining values become
  the same summaries that the depth limit makes. Sensitive keys are redacted at
  every depth.
  """
  @spec transport_payload(term(), keyword()) :: term()
  def transport_payload(payload, opts \\ []), do: sanitize(payload, :transport, opts)

  defp sanitize_opts(defaults, opts) do
    Enum.reduce(opts, defaults, fn
      {key, value}, acc when is_map_key(acc, key) and is_integer(value) and value > 0 -> Map.put(acc, key, value)
      _other, acc -> acc
    end)
  end

  defp run(payload, profile, opts) do
    {sanitized, _spent} = sanitize_value(payload, profile, opts, 0, 0)
    sanitized
  end

  # Each clause returns the sanitized value and the bytes spent so far. The
  # budget is checked before the walk goes into a container, so a payload that
  # is too large is never expanded.
  defp sanitize_value(value, _profile, _opts, _depth, spent)
       when is_nil(value) or is_boolean(value) or is_number(value),
       do: {value, spent + @scalar_cost}

  defp sanitize_value(value, _profile, _opts, _depth, spent) when is_atom(value),
    do: {value, spent + atom_cost(value)}

  defp sanitize_value(value, profile, opts, _depth, spent) when spent >= opts.max_bytes,
    do: {summarize_value(value, profile, opts), spent + @summary_cost}

  defp sanitize_value(value, _profile, opts, _depth, spent) when is_binary(value) do
    sanitized = sanitize_binary(value, opts.max_string_chars)
    {sanitized, spent + byte_size(sanitized) + @quote_cost}
  end

  defp sanitize_value(value, profile, opts, depth, spent) when depth >= opts.max_depth,
    do: {summarize_value(value, profile, opts), spent + @summary_cost}

  defp sanitize_value(%module{} = value, profile, opts, depth, spent) do
    cond do
      is_exception(value) ->
        message = sanitize_binary(Exception.message(value), opts.max_string_chars)
        {%{type: module_label(module), message: message}, spent + byte_size(message) + @summary_cost}

      profile == :transport ->
        value
        |> Map.from_struct()
        |> Map.put(:__struct__, module_label(module))
        |> sanitize_value(profile, opts, depth + 1, spent)

      true ->
        {summarize_value(value, profile, opts), spent + @summary_cost}
    end
  end

  defp sanitize_value(value, profile, opts, depth, spent) when is_map(value) do
    {entries, omitted_count} =
      value
      |> Enum.take(opts.max_map_entries + 1)
      |> then(fn entries ->
        if length(entries) > opts.max_map_entries do
          {Enum.take(entries, opts.max_map_entries), map_size(value) - opts.max_map_entries}
        else
          {entries, 0}
        end
      end)

    {pairs, spent} =
      Enum.map_reduce(entries, spent + @container_cost, fn {key, entry_value}, spent ->
        sanitized_key = sanitize_key(key)
        spent = spent + key_cost(sanitized_key)

        cond do
          sensitive_key?(key) ->
            {{sanitized_key, @redacted}, spent + byte_size(@redacted)}

          profile == :telemetry and telemetry_payload_summary_key?(key) ->
            {{sanitized_key, summarize_value(entry_value, profile, opts)}, spent + @summary_cost}

          true ->
            {sanitized_entry, spent} = sanitize_value(entry_value, profile, opts, depth + 1, spent)
            {{sanitized_key, sanitized_entry}, spent}
        end
      end)

    {pairs |> Map.new() |> maybe_put_truncated(omitted_count), spent}
  end

  defp sanitize_value(value, profile, opts, depth, spent) when is_list(value) do
    if proper_list?(value) do
      {items, omitted_count} = take_with_omitted_count(value, opts.max_list_items)

      {sanitized_items, spent} =
        Enum.map_reduce(items, spent + @container_cost, fn item, spent ->
          sanitize_value(item, profile, opts, depth + 1, spent)
        end)

      {maybe_append_truncated(sanitized_items, omitted_count), spent}
    else
      {summarize_value(value, profile, opts), spent + @summary_cost}
    end
  end

  defp sanitize_value(value, profile, opts, depth, spent) when is_tuple(value) do
    if profile == :transport do
      {items, spent} =
        value
        |> Tuple.to_list()
        |> Enum.map_reduce(spent + @container_cost, fn item, spent ->
          sanitize_value(item, profile, opts, depth + 1, spent)
        end)

      {%{type: :tuple, items: items, size: tuple_size(value)}, spent}
    else
      {summarize_value(value, profile, opts), spent + @summary_cost}
    end
  end

  defp sanitize_value(value, profile, opts, _depth, spent),
    do: {summarize_value(value, profile, opts), spent + @summary_cost}

  defp atom_cost(value), do: byte_size(Atom.to_string(value)) + @quote_cost

  defp key_cost(key) when is_atom(key), do: atom_cost(key) + @key_cost
  defp key_cost(key) when is_binary(key), do: byte_size(key) + @key_cost + @quote_cost

  defp summarize_value(value, profile, opts) do
    value
    |> summary_base()
    |> maybe_add_summary_details(value, profile, opts)
  end

  defp summary_base(value) when is_binary(value), do: %{type: :string, bytes: byte_size(value)}
  defp summary_base(value) when is_list(value), do: %{type: :list, length: safe_length(value)}
  defp summary_base(value) when is_map(value), do: %{type: :map, size: map_size(value), keys: summary_keys(value)}
  defp summary_base(value) when is_tuple(value), do: %{type: :tuple, size: tuple_size(value)}
  defp summary_base(value) when is_function(value), do: %{type: :function}
  defp summary_base(value) when is_pid(value), do: %{type: :pid}
  defp summary_base(value) when is_port(value), do: %{type: :port}
  defp summary_base(value) when is_reference(value), do: %{type: :reference}
  defp summary_base(value) when is_atom(value), do: %{type: :atom, value: value}
  defp summary_base(value) when is_number(value), do: %{type: :number, value: value}
  defp summary_base(_value), do: %{type: :term}

  defp maybe_add_summary_details(summary, {:ok, payload, effects}, profile, opts) when is_list(effects) do
    Map.merge(summary, %{
      status: :ok,
      value: summarize_value(payload, profile, opts),
      effects_count: length(effects)
    })
  end

  defp maybe_add_summary_details(summary, {:error, error, effects}, profile, opts) when is_list(effects) do
    Map.merge(summary, %{
      status: :error,
      error: summarize_error(error, profile, opts),
      effects_count: length(effects)
    })
  end

  defp maybe_add_summary_details(summary, %{} = value, profile, opts) do
    cond do
      Map.has_key?(value, :type) or Map.has_key?(value, "type") or Map.has_key?(value, :message) or
          Map.has_key?(value, "message") ->
        Map.put(summary, :error, summarize_error(value, profile, opts))

      true ->
        summary
    end
  end

  defp maybe_add_summary_details(summary, value, _profile, opts) when is_binary(value) do
    Map.put(summary, :truncated?, String.valid?(value) and String.length(value) > opts.max_string_chars)
  end

  defp maybe_add_summary_details(summary, value, :transport, opts) do
    if json_scalar?(value) do
      summary
    else
      Map.put(summary, :inspect, inspect_limited(value, opts.max_inspect_chars))
    end
  end

  defp maybe_add_summary_details(summary, _value, _profile, _opts), do: summary

  defp summarize_error(%{} = error, profile, opts) do
    %{}
    |> maybe_put_summary_field(:type, Map.get(error, :type, Map.get(error, "type")))
    |> maybe_put_summary_field(:code, Map.get(error, :code, Map.get(error, "code")))
    |> maybe_put_summary_field(
      :message,
      sanitize_summary_message(Map.get(error, :message, Map.get(error, "message")), opts)
    )
    |> maybe_put_summary_field(:retryable?, Map.get(error, :retryable?, Map.get(error, "retryable?")))
    |> case do
      empty when empty == %{} -> summarize_value(Map.drop(error, [:details, "details"]), profile, opts)
      summary -> summary
    end
  end

  defp summarize_error(error, _profile, opts) when is_exception(error) do
    %{type: module_label(error.__struct__), message: sanitize_binary(Exception.message(error), opts.max_string_chars)}
  end

  defp summarize_error(error, profile, opts), do: summarize_value(error, profile, opts)

  defp maybe_put_summary_field(map, _key, nil), do: map
  defp maybe_put_summary_field(map, key, value), do: Map.put(map, key, value)

  defp sanitize_summary_message(nil, _opts), do: nil

  defp sanitize_summary_message(message, opts) when is_binary(message),
    do: sanitize_binary(message, opts.max_string_chars)

  defp sanitize_summary_message(message, _opts) when is_atom(message), do: message
  defp sanitize_summary_message(message, opts), do: inspect_limited(message, opts.max_inspect_chars)

  defp sanitize_binary(value, max_chars) do
    cond do
      not String.valid?(value) ->
        "[BINARY #{byte_size(value)} bytes]"

      String.length(value) > max_chars ->
        String.slice(value, 0, max_chars) <> "...[truncated]"

      true ->
        value
    end
  end

  defp sanitize_key(key) when is_binary(key) or is_atom(key), do: key
  defp sanitize_key(key), do: inspect_limited(key, @telemetry_defaults.max_inspect_chars)

  defp telemetry_payload_summary_key?(key), do: MapSet.member?(@telemetry_payload_summary_keys, key)

  defp take_with_omitted_count(list, max_items) do
    items = Enum.take(list, max_items + 1)

    if length(items) > max_items do
      {Enum.take(items, max_items), length(list) - max_items}
    else
      {items, 0}
    end
  end

  defp maybe_put_truncated(map, omitted_count) when omitted_count > 0 do
    Map.put(map, @truncated_key, %{omitted_entries: omitted_count})
  end

  defp maybe_put_truncated(map, _omitted_count), do: map

  defp maybe_append_truncated(list, omitted_count) when omitted_count > 0 do
    list ++ [%{@truncated_key => %{omitted_items: omitted_count}}]
  end

  defp maybe_append_truncated(list, _omitted_count), do: list

  defp summary_keys(map) do
    map
    |> Map.keys()
    |> Enum.take(10)
    |> Enum.map(&sanitize_key/1)
  end

  defp safe_length(list) do
    if proper_list?(list), do: length(list), else: :improper
  end

  defp proper_list?([]), do: true
  defp proper_list?([_head | tail]), do: proper_list?(tail)
  defp proper_list?(_), do: false

  defp sensitive_key?(key) when is_atom(key), do: key |> Atom.to_string() |> sensitive_key?()

  defp sensitive_key?(key) when is_binary(key) do
    key = String.downcase(key)

    MapSet.member?(@sensitive_exact_keys, key) or
      Enum.any?(@sensitive_contains, &String.contains?(key, &1)) or
      Enum.any?(@sensitive_suffixes, &String.ends_with?(key, &1))
  end

  defp sensitive_key?(_key), do: false

  defp json_scalar?(value),
    do: is_nil(value) or is_boolean(value) or is_binary(value) or is_number(value) or is_atom(value)

  defp inspect_limited(value, max_chars), do: inspect(value, limit: 20, printable_limit: max_chars)

  defp module_label(module) when is_atom(module), do: module |> Atom.to_string() |> String.replace_prefix("Elixir.", "")
end
