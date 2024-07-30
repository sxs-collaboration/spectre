# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import stat
import unittest
from pathlib import Path

import yaml
from click.testing import CliRunner

from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.support.Logging import configure_logging
from spectre.support.Resubmit import resubmit, resubmit_command
from spectre.support.Schedule import (
    Checkpoint,
    Segment,
    list_checkpoints,
    list_segments,
    schedule,
    schedule_command,
)


class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(unit_test_build_path(), "Schedule").resolve()
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.spectre_cli = (
            Path(unit_test_build_path()).parent.parent / "bin/spectre"
        )

        # Create an executable that just outputs all arguments passed to it
        self.executable = self.test_dir / "TestExec"
        self.executable.write_text("""#!/bin/bash
echo $@ > out.txt
""")
        self.executable.chmod(self.executable.stat().st_mode | stat.S_IEXEC)

        # Create an input file template
        self.input_file_template = self.test_dir / "InputFile.yaml"
        self.input_file_template.write_text("""
Executable: TestExec
MetadataOption: {{ metadata_option }}
---
Option: {{ extra_option }}
""")

        # Create a submit script template
        (self.test_dir / "SubmitTemplateBase.sh").write_text("""\
SPECTRE_EXECUTABLE={{ executable }}
SPECTRE_CLI={{ spectre_cli }}
{% block derived %}
{% endblock %}
""")
        self.submit_script_template = self.test_dir / "SubmitTemplate.sh"
        self.submit_script_template.write_text("""\
{% extends "SubmitTemplateBase.sh" %}
{% block derived %}
NUM_NODES={{ num_nodes | default(1) }}
{% endblock %}
""")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_run(self):
        # Run executable once
        proc = schedule(
            input_file_template=self.input_file_template,
            scheduler=None,
            executable=self.executable,
            run_dir=self.test_dir / "Run",
            # Passing one extra param in a dict and another as a keyword
            # argument to test that both work
            extra_params=dict(extra_option="TestOpt"),
            metadata_option="MetaOpt",
        )
        self.assertEqual(proc.returncode, 0)
        with open(self.test_dir / "Run/InputFile.yaml", "r") as open_input_file:
            rendered_metadata, rendered_input_file = yaml.safe_load_all(
                open_input_file
            )
            self.assertEqual(rendered_input_file["Option"], "TestOpt")
            self.assertEqual(rendered_metadata["MetadataOption"], "MetaOpt")
        args = (self.test_dir / "Run/out.txt").read_text().split()
        self.assertEqual(
            args,
            [
                "--input-file",
                str(self.test_dir / "Run/InputFile.yaml"),
                "+auto-provision",
            ],
        )

        # Run executable multiple times (like a convergence test)
        schedule(
            input_file_template=self.input_file_template,
            scheduler=None,
            executable=self.executable,
            run_dir=self.test_dir / "Lev{{ lev }}",
            lev=range(1, 3),
            extra_option="TestOpt",
            metadata_option="MetaOpt",
        )
        self.assertEqual(
            sorted(self.test_dir.glob("Lev*")),
            [self.test_dir / "Lev1", self.test_dir / "Lev2"],
        )
        self.assertEqual(
            sorted(self.test_dir.glob("Lev*/InputFile.yaml")),
            [
                self.test_dir / "Lev1/InputFile.yaml",
                self.test_dir / "Lev2/InputFile.yaml",
            ],
        )

        # Create first of a series of segments
        schedule(
            input_file_template=self.input_file_template,
            scheduler=None,
            executable=self.executable,
            segments_dir=self.test_dir,
            extra_option="TestOpt",
            metadata_option="MetaOpt",
        )
        self.assertEqual(
            sorted(self.test_dir.glob("Segment*")),
            [self.test_dir / "Segment_0000"],
        )
        self.assertEqual(
            sorted(self.test_dir.glob("Segment*/InputFile.yaml")),
            [self.test_dir / "Segment_0000/InputFile.yaml"],
        )

        # Create next segment
        # - Can't continue without a previous checkpoint
        with self.assertRaisesRegex(AssertionError, "continue from the last"):
            schedule(
                input_file_template=self.input_file_template,
                scheduler=None,
                executable=self.executable,
                segments_dir=self.test_dir,
                extra_option="TestOpt",
                metadata_option="MetaOpt",
            )
        # - Can't continue from an earlier checkpoint than the last
        earlier_checkpoint = Checkpoint.match(
            self.test_dir / "Segment_0000/Checkpoints/Checkpoint_0000"
        )
        earlier_checkpoint.path.mkdir(parents=True)
        last_checkpoint = Checkpoint.match(
            self.test_dir / "Segment_0000/Checkpoints/Checkpoint_0001"
        )
        last_checkpoint.path.mkdir(parents=True)
        with self.assertRaisesRegex(AssertionError, "continue from the last"):
            schedule(
                input_file_template=self.input_file_template,
                scheduler=None,
                executable=self.executable,
                segments_dir=self.test_dir,
                from_checkpoint=earlier_checkpoint,
                extra_option="TestOpt",
                metadata_option="MetaOpt",
            )
        # - Continue from last checkpoint
        schedule(
            input_file_template=self.input_file_template,
            scheduler=None,
            executable=self.executable,
            segments_dir=self.test_dir,
            from_checkpoint=last_checkpoint,
            extra_option="TestOpt",
            metadata_option="MetaOpt",
        )

    def test_schedule(self):
        # Submit a batch job to create the first segment
        schedule(
            input_file_template=self.input_file_template,
            scheduler=["echo", "Submitted batch job 000"],
            submit_script_template=self.submit_script_template,
            executable=self.executable,
            segments_dir=self.test_dir / "Segments",
            extra_option="TestOpt",
            metadata_option="MetaOpt",
            submit=True,
        )
        self.assertEqual(
            sorted(self.test_dir.glob("Segments/**/*")),
            [
                self.test_dir / "Segments/Segment_0000",
                self.test_dir / "Segments/Segment_0000/InputFile.yaml",
                self.test_dir / "Segments/Segment_0000/SchedulerContext.yaml",
                self.test_dir / "Segments/Segment_0000/Submit.sh",
                self.test_dir / "Segments/Segment_0000/jobid.txt",
                self.test_dir / "Segments/Segment_0000/out.txt",
                self.test_dir / "Segments/SubmitTemplate.sh",
                self.test_dir / "Segments/SubmitTemplateBase.sh",
                self.test_dir / "Segments/TestExec",
            ],
        )
        self.assertEqual(
            (self.test_dir / "Segments/Segment_0000/jobid.txt").read_text(),
            "000",
        )
        self.assertEqual(
            (self.test_dir / "Segments/Segment_0000/out.txt")
            .read_text()
            .split(),
            [
                "--input-file",
                str(self.test_dir / "Segments/Segment_0000/InputFile.yaml"),
                "--check-options",
            ],
        )
        with open(
            self.test_dir / "Segments/Segment_0000/SchedulerContext.yaml", "r"
        ) as open_context_file:
            context = yaml.safe_load(open_context_file)
        self.assertDictEqual(
            context,
            dict(
                clean_output=False,
                context_file_name="SchedulerContext.yaml",
                copy_executable=None,
                executable=str(self.test_dir / "Segments/TestExec"),
                executable_name="TestExec",
                extra_option="TestOpt",
                metadata_option="MetaOpt",
                force=False,
                no_validate=False,
                input_file="InputFile.yaml",
                input_file_name="InputFile.yaml",
                input_file_template=str(self.test_dir / "InputFile.yaml"),
                job_name="TestExec",
                out_file=str(
                    self.test_dir / "Segments/Segment_0000/spectre.out"
                ),
                out_file_name="spectre.out",
                run_dir=str(self.test_dir / "Segments/Segment_0000"),
                scheduler=["echo", "Submitted batch job 000"],
                segments_dir=str(self.test_dir / "Segments"),
                spectre_cli=str(self.spectre_cli),
                submit=True,
                submit_script_name="Submit.sh",
                submit_script_template=str(
                    self.test_dir / "Segments/SubmitTemplate.sh"
                ),
            ),
        )
        self.assertEqual(
            (self.test_dir / "Segments/Segment_0000/Submit.sh").read_text(),
            """\
SPECTRE_EXECUTABLE={executable}
SPECTRE_CLI={spectre_cli}
NUM_NODES=1
""".format(
                executable=self.test_dir / "Segments/TestExec",
                spectre_cli=self.spectre_cli,
            ),
        )

        # Resubmit from the first segment using `schedule`
        checkpoint = Checkpoint.match(
            self.test_dir / "Segments/Segment_0000/Checkpoints/Checkpoint_0000"
        )
        checkpoint.path.mkdir(parents=True)
        schedule(
            input_file_template=self.test_dir
            / "Segments/Segment_0000/InputFile.yaml",
            scheduler=["echo", "Submitted batch job 001"],
            submit_script_template=self.test_dir / "Segments/SubmitTemplate.sh",
            executable=self.test_dir / "Segments/TestExec",
            segments_dir=self.test_dir / "Segments",
            from_checkpoint=checkpoint,
            extra_option="TestOpt",
            metadata_option="MetaOpt",
            submit=True,
        )
        self.assertEqual(
            (self.test_dir / "Segments/Segment_0001/jobid.txt").read_text(),
            "001",
        )

        # Resubmit from the second segment using `resubmit`
        checkpoint = Checkpoint.match(
            self.test_dir / "Segments/Segment_0001/Checkpoints/Checkpoint_0000"
        )
        checkpoint.path.mkdir(parents=True)
        resubmit(self.test_dir / "Segments", submit=True)
        self.assertEqual(
            (self.test_dir / "Segments/Segment_0002/jobid.txt").read_text(),
            "001",
        )

    def test_cli(self):
        runner = CliRunner()
        runner.invoke(
            schedule_command,
            [
                str(self.input_file_template),
                "-E",
                str(self.executable),
                "-o",
                str(self.test_dir / "Run"),
                "-p",
                "extra_option=TestOpt",
                "-p",
                "metadata_option=MetaOpt",
            ],
            catch_exceptions=False,
        )
        runner.invoke(
            schedule_command,
            [
                str(self.input_file_template),
                "-E",
                str(self.executable),
                "-O",
                str(self.test_dir / "Segments"),
                "-p",
                "extra_option=TestOpt",
                "-p",
                "metadata_option=MetaOpt",
                "--scheduler",
                "cat",
                "--submit-script-template",
                str(self.submit_script_template),
                "--submit",
            ],
            catch_exceptions=False,
        )
        checkpoint = Checkpoint.match(
            self.test_dir / "Segments/Segment_0000/Checkpoints/Checkpoint_0000"
        )
        checkpoint.path.mkdir(parents=True)
        runner.invoke(
            resubmit_command,
            [
                str(self.test_dir / "Segments"),
            ],
            catch_exceptions=False,
        )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
