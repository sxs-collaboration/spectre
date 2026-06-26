# Distributed under the MIT License.
# See LICENSE.txt for details.

import importlib.util
import unittest

from spectre.__main__ import cli
from spectre.support import Mcp


def _commands(expose="all", **kwargs):
    """Return a {tool_name: DiscoveredCommand} map for the spectre CLI."""
    return {
        entry.tool_name: entry
        for entry in Mcp.discover_commands(cli, expose=expose, **kwargs)
    }


class TestMcp(unittest.TestCase):
    def test_discover_safe_vs_all(self):
        safe = _commands(expose="safe")
        all_commands = _commands(expose="all")
        # 'all' is a superset of 'safe' and includes the unsafe commands.
        self.assertTrue(set(safe).issubset(set(all_commands)))
        self.assertGreater(len(all_commands), len(safe))
        # A representative sample of analysis/extraction commands is exposed in
        # the safe set.
        for tool in ["extract_dat", "generate_xdmf", "validate"]:
            self.assertIn(tool, safe)

    def test_unsafe_commands_gating(self):
        safe = _commands(expose="safe")
        all_commands = _commands(expose="all")
        # Cluster-job / auto-resubmit / destructive commands are hidden by
        # default. 'schedule' submits jobs; 'delete_subfiles' and 'clean_output'
        # irreversibly delete data.
        for tool in [
            "resubmit",
            "bbh_generate_id",
            "schedule",
            "delete_subfiles",
            "clean_output",
        ]:
            self.assertNotIn(tool, safe)
            self.assertIn(tool, all_commands)
        # 'transform-volume-data' runs arbitrary user kernels and rewrites H5
        # files in place, so it is gated like the destructive commands. It is
        # an optional command, so only assert the gating when it is built.
        if "transform_volume_data" in all_commands:
            self.assertNotIn("transform_volume_data", safe)
        # ... but '--include' re-enables a specific one without exposing all.
        with_included = _commands(expose="safe", include=["bbh-generate-id"])
        self.assertIn("bbh_generate_id", with_included)
        self.assertNotIn("resubmit", with_included)
        # '--exclude' removes a command even in 'all' mode.
        without_status = _commands(expose="all", exclude=["status"])
        self.assertNotIn("status", without_status)
        # '--exclude' wins over '--include' for the same command.
        excluded_and_included = _commands(
            expose="safe",
            include=["delete-subfiles"],
            exclude=["delete-subfiles"],
        )
        self.assertNotIn("delete_subfiles", excluded_and_included)

    def test_groups_and_self_excluded(self):
        names = set(_commands(expose="all"))
        # Groups are recursed into, not exposed as tools, and the server does
        # not expose itself.
        self.assertNotIn("mcp", names)
        self.assertNotIn("bbh", names)
        self.assertNotIn("plot", names)
        # Nested leaf commands are exposed with a path-joined name.
        self.assertIn("bbh_find_horizon", names)
        self.assertIn("plot_dat", names)

    def test_schema_clean_output(self):
        schema = Mcp.command_input_schema(_commands()["clean_output"].command)
        properties = schema["properties"]
        # Required positional argument maps to a required string property.
        self.assertEqual(properties["input_file"]["type"], "string")
        self.assertIn("input_file", schema["required"])
        # '--output-dir' is a required string option.
        self.assertEqual(properties["output_dir"]["type"], "string")
        # '--force' is a boolean flag with a default.
        self.assertEqual(properties["force"]["type"], "boolean")
        self.assertEqual(properties["force"]["default"], False)
        self.assertEqual(schema["additionalProperties"], False)

    def test_schema_enum_and_array(self):
        # 'click.Choice' maps to a string enum.
        schedule = Mcp.command_input_schema(_commands()["schedule"].command)
        profile_with = schedule["properties"]["profile_with"]
        self.assertEqual(profile_with["type"], "string")
        self.assertEqual(profile_with["enum"], ["hpctoolkit"])
        # A 'multiple=True' option maps to an array of the base type.
        extract = Mcp.command_input_schema(_commands()["extract_dat"].command)
        subfiles = extract["properties"]["subfiles"]
        self.assertEqual(subfiles["type"], "array")
        self.assertEqual(subfiles["items"]["type"], "string")
        # An optional argument with no real default does not advertise one.
        self.assertNotIn("default", extract["properties"]["out_dir"])

    def test_schema_range_and_nargs(self):
        commands = _commands(expose="all")
        if "bbh_generate_id" not in commands:
            self.skipTest("bbh generate-id is not available in this build")
        properties = Mcp.command_input_schema(
            commands["bbh_generate_id"].command
        )["properties"]
        # 'click.FloatRange(1.0, None)' maps to a number with a minimum.
        if "mass_ratio" in properties:
            self.assertEqual(properties["mass_ratio"]["type"], "number")
            self.assertEqual(properties["mass_ratio"]["minimum"], 1.0)
        # A fixed 'nargs=3' option (e.g. a spin vector) maps to a 3-item array.
        fixed_arrays = [
            value
            for value in properties.values()
            if value.get("type") == "array" and value.get("minItems") == 3
        ]
        self.assertTrue(
            fixed_arrays, "expected a 3-element array option (spin vector)"
        )
        self.assertEqual(fixed_arrays[0]["maxItems"], 3)

    def test_arguments_to_argv_roundtrip(self):
        command = _commands()["clean_output"].command
        argv = Mcp.arguments_to_argv(
            command,
            {"input_file": "/in.yaml", "output_dir": "/out", "force": True},
        )
        # Positional argument comes last; options precede it.
        self.assertEqual(argv[-1], "/in.yaml")
        self.assertIn("--force", argv)
        self.assertIn("--output-dir", argv)
        self.assertEqual(argv[argv.index("--output-dir") + 1], "/out")
        # Repeatable options are repeated.
        extract = _commands()["extract_dat"].command
        argv = Mcp.arguments_to_argv(
            extract, {"filename": "/f.h5", "subfiles": ["/a.dat", "/b.dat"]}
        )
        self.assertEqual(argv.count("--subfile"), 2)
        self.assertEqual(argv[-1], "/f.h5")

    def test_schema_and_argv_repeatable_fixed_group(self):
        # A 'multiple=True' option that also takes a fixed 'nargs > 1' group is
        # a list of fixed-length tuples: the schema must be an array of
        # fixed-length arrays, and the argv must repeat the option once per
        # group. No current command uses this combination, so build a synthetic
        # one (regression guard for treating it as a single flat group).
        import click

        @click.command()
        @click.option("--pair", nargs=2, multiple=True, type=float)
        def cmd(pair):
            pass

        prop = Mcp.command_input_schema(cmd)["properties"]["pair"]
        self.assertEqual(prop["type"], "array")
        self.assertEqual(prop["items"]["type"], "array")
        self.assertEqual(prop["items"]["minItems"], 2)
        self.assertEqual(prop["items"]["maxItems"], 2)
        argv = Mcp.arguments_to_argv(cmd, {"pair": [[1.0, 2.0], [3.0, 4.0]]})
        self.assertEqual(argv, ["--pair", "1.0", "2.0", "--pair", "3.0", "4.0"])

    def test_argv_boolean_flag_with_secondary(self):
        schedule = _commands()["schedule"].command
        # '--validate/--no-validate' uses the primary opt when True ...
        argv_true = Mcp.arguments_to_argv(
            schedule, {"input_file_template": "/x.yaml", "validate": True}
        )
        self.assertIn("--validate", argv_true)
        self.assertNotIn("--no-validate", argv_true)
        # ... and the secondary opt when False.
        argv_false = Mcp.arguments_to_argv(
            schedule, {"input_file_template": "/x.yaml", "validate": False}
        )
        self.assertIn("--no-validate", argv_false)

    def test_run_tool_in_process(self):
        # The in-process runner returns the exit code and captured output. Use
        # '--help' so the command is hermetic and deterministic.
        text = Mcp.run_tool_in_process(["clean-output"], ["--help"])
        self.assertIn("exit code 0", text)
        self.assertIn("Usage:", text)

    def test_run_tool_subprocess(self):
        # The subprocess runner returns the exit code and captured output. Use
        # '--help' so the command is hermetic, deterministic, and fast.
        text = Mcp.run_tool_subprocess(["clean-output"], ["--help"], timeout=60)
        self.assertIn("exit code 0", text)
        self.assertIn("Usage:", text)

    def test_format_result_no_stderr_duplication(self):
        # stdout and stderr are formatted into independent sections, so a
        # message that appears only on stderr must not leak into the stdout
        # section (regression guard for using 'result.output', the combined
        # stream, instead of 'result.stdout').
        import click
        from click.testing import CliRunner

        @click.command()
        def emit():
            click.echo("hello-stdout")
            click.echo("oops-stderr", err=True)

        # Older 'click' (< 8.2) mixes stderr into stdout by default and raises
        # 'ValueError' on 'result.stderr'; 'mix_stderr=False' captures them
        # separately. Click >= 8.2 removed the parameter and always captures
        # stderr separately, so fall back to the default runner there.
        try:
            runner = CliRunner(mix_stderr=False)
        except TypeError:
            runner = CliRunner()
        result = runner.invoke(emit, [], input="")
        text = Mcp._format_result(
            result.exit_code, result.stdout, result.stderr
        )
        self.assertEqual(text.count("oops-stderr"), 1)
        self.assertEqual(text.count("hello-stdout"), 1)

    def test_format_result_truncates_large_output(self):
        # A stream larger than the cap is truncated with an explicit marker,
        # and the result length stays bounded.
        big = "x" * (Mcp.MAX_OUTPUT_CHARS + 5000)
        text = Mcp._format_result(0, big, "")
        self.assertIn("output truncated", text)
        self.assertLess(len(text), Mcp.MAX_OUTPUT_CHARS + 500)
        # Output at or below the cap is returned verbatim.
        small = "y" * 100
        self.assertNotIn("output truncated", Mcp._format_result(0, small, ""))

    @unittest.skipUnless(
        importlib.util.find_spec("mcp"), "the 'mcp' package is not installed"
    )
    def test_generated_schemas_are_valid_tools(self):
        # Every generated tool must be accepted by the MCP 'Tool' model, which
        # validates the name, description, and input schema.
        import mcp.types as types

        for entry in Mcp.discover_commands(cli, expose="all"):
            tool = types.Tool(
                name=entry.tool_name,
                description=Mcp.tool_description(entry.command),
                inputSchema=Mcp.command_input_schema(entry.command),
            )
            self.assertEqual(tool.name, entry.tool_name)
            self.assertTrue(tool.description)


if __name__ == "__main__":
    unittest.main(verbosity=2)
