#!/usr/bin/env wolframscript
(* Distributed under the MIT License. *)
(* See LICENSE.txt for details. *)

(* Tests the Mathematica interface to the SpECTRE exporter library. Usage:
   wolframscript -file Test_BundledExporter.wls SPECTRE_ROOT VOLUME_FILE \
     SUBFILE_NAME STEP FIELD X Y Z EXPECTED *)

args = Rest[$ScriptCommandLine];
If[Length[args] != 9,
  Print["Usage: Test_BundledExporter.wls SPECTRE_ROOT VOLUME_FILE \
SUBFILE_NAME STEP FIELD X Y Z EXPECTED"];
  Exit[1]
];
{spectreRoot, volumeFile, subfileName, step, field, x, y, z, expected} = args;
{step, x, y, z, expected} = ToExpression /@ {step, x, y, z, expected};

Get[FileNameJoin[{spectreRoot, "include", "spectre", "SpectreExporter.wl"}]];

(* Interpolate data *)
result = SpectreInterpolate[{volumeFile}, subfileName, step, {field},
  {{x}, {y}, {z}}];
If[FailureQ[result],
  Print["FAILURE: ", result["Message"]];
  Exit[1]
];
(* Check that errors are returned to the caller *)
If[Not[FailureQ[SpectreInterpolate[{volumeFile}, "nonexistent", step, {field},
    {{x}, {y}, {z}}]]],
  Print["FAILURE: Expected an error for a nonexistent subfile."];
  Exit[1]
];
(* Check that points outside the domain are Indeterminate *)
resultWithMissingPoint = SpectreInterpolate[{volumeFile}, subfileName, step,
  {field}, {{x, 1.*^10}, {y, 0.}, {z, 0.}}];
If[resultWithMissingPoint[[1, 1]] != result[[1, 1]]
    || resultWithMissingPoint[[1, 2]] =!= Indeterminate,
  Print["FAILURE: Expected Indeterminate for a point outside the domain."];
  Exit[1]
];
(* Check result *)
If[Abs[result[[1, 1]] - expected] < 1.*^-10,
  Print["SUCCESS"],
  Print["FAILURE. Result is: ", result[[1, 1]]];
  Exit[1]
];
