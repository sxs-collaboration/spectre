(* Distributed under the MIT License. *)
(* See LICENSE.txt for details. *)

(* Mathematica interface to the SpECTRE exporter library (see
   spectre/Exporter.h). Requires ForeignFunctionLoad (tested with Mathematica
   14.3). Load the package from the SpECTRE build directory:

     Get["/path/to/spectre/build/include/spectre/SpectreExporter.wl"]

   It finds the BundledExporter library in the build directory. To use a
   different library, set $SpectreExporterLibrary before loading the package. *)

BeginPackage["SpectreExporter`"];

$SpectreExporterLibrary::usage =
  "$SpectreExporterLibrary is the path to the SpECTRE BundledExporter library.";

SpectreInterpolate::usage =
  "SpectreInterpolate[volumeFiles, subfileName, observationStep, \
tensorComponents, targetPoints] interpolates the tensorComponents in the \
subfileName of the H5 volumeFiles to the targetPoints. The observationStep is \
the index of the observation in the volume files (0 is the first observation \
and -1 is the last). The targetPoints have shape {numDimensions, numPoints}. \
Returns the interpolated data with shape {numTensorComponents, numPoints}, or \
a Failure. Points outside the domain are Indeterminate unless the option \
\"ErrorOnMissingPoints\" is True. Further options are \
\"ExtrapolateIntoExcisions\" and \"NumThreads\".";

Begin["`Private`"];

If[Not[ValueQ[$SpectreExporterLibrary]],
  $SpectreExporterLibrary = FileNameJoin[{
    ParentDirectory[DirectoryName[$InputFileName], 2], "lib",
    "libBundledExporter." <> Switch[$OperatingSystem,
      "MacOSX", "dylib", "Windows", "dll", _, "so"]}]
];

(* Load the C functions once per library *)
cFunctions[library_String] := cFunctions[library] = <|
  "Interpolate" -> ForeignFunctionLoad[library,
    "spectre_interpolate_to_points", {
      "RawPointer"::["RawPointer"::["CDouble"]], (* out *)
      "RawPointer"::["RawPointer"::["UnsignedInteger8"]], (* volume_files *)
      "CSizeT", (* num_volume_files *)
      "RawPointer"::["UnsignedInteger8"], (* subfile_name *)
      "CInt", (* observation_step *)
      "RawPointer"::["RawPointer"::["UnsignedInteger8"]],
        (* tensor_components *)
      "CSizeT", (* num_tensor_components *)
      "RawPointer"::["RawPointer"::["CDouble"]], (* target_points *)
      "CSizeT", (* num_dimensions *)
      "CSizeT", (* num_points *)
      "CUnsignedChar", (* extrapolate_into_excisions *)
      "CUnsignedChar", (* error_on_missing_points *)
      "CInt" (* num_threads *)
    } -> "CInt"],
  "LastError" -> ForeignFunctionLoad[library, "spectre_exporter_last_error",
    {} -> "RawPointer"::["UnsignedInteger8"]]
|>;

Options[SpectreInterpolate] = {
  "ExtrapolateIntoExcisions" -> False,
  "ErrorOnMissingPoints" -> False,
  "NumThreads" -> Automatic
};

SpectreInterpolate[volumeFiles : {__String}, subfileName_String,
    observationStep_Integer, tensorComponents : {__String},
    targetPoints_?(MatrixQ[#, NumericQ] &), OptionsPattern[]] :=
  Module[{functions, numDimensions, numPoints, volumeFilePtrs,
      tensorComponentPtrs, targetPointPtrs, outPtrs, status},
    functions = cFunctions[$SpectreExporterLibrary];
    {numDimensions, numPoints} = Dimensions[targetPoints];
    (* The C function reads from and writes to these buffers directly. Keep
       references to all of them until the call returns, because the memory
       is freed once they are no longer referenced. *)
    volumeFilePtrs = RawMemoryExport /@ volumeFiles;
    tensorComponentPtrs = RawMemoryExport /@ tensorComponents;
    targetPointPtrs = RawMemoryExport[#, "CDouble"] & /@ N[targetPoints];
    outPtrs = Table[RawMemoryAllocate["CDouble", numPoints],
      Length[tensorComponents]];
    status = functions["Interpolate"][
      RawMemoryExport[outPtrs, "RawPointer"::["CDouble"]],
      RawMemoryExport[volumeFilePtrs, "RawPointer"::["UnsignedInteger8"]],
      Length[volumeFiles],
      subfileName,
      observationStep,
      RawMemoryExport[tensorComponentPtrs,
        "RawPointer"::["UnsignedInteger8"]],
      Length[tensorComponents],
      RawMemoryExport[targetPointPtrs, "RawPointer"::["CDouble"]],
      numDimensions,
      numPoints,
      Boole[TrueQ[OptionValue["ExtrapolateIntoExcisions"]]],
      Boole[TrueQ[OptionValue["ErrorOnMissingPoints"]]],
      Replace[OptionValue["NumThreads"], Automatic -> 0]
    ];
    If[status != 0,
      Return[Failure["SpectreExporterError", <|
        "MessageTemplate" -> RawMemoryImport[functions["LastError"][],
          "String"]|>]]
    ];
    Normal[RawMemoryImport[#, {"NumericArray", numPoints}]] & /@ outPtrs
  ];

End[];

EndPackage[];
