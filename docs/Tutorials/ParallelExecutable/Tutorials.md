\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Writing SpECTRE executables {#tutorials_parallel}

This set of tutorials will develop a parallel executable in stages,
beginning with the simplest possible executable, and then introducing
SpECTRE's parallel features one step at a time.  At each stage, the
tutorials will illustrate what code a user needs to add in order to
use a particular feature, and then, for those interested, an
explanation of the metaprogramming that turns the user-provided code
into an actual SpECTRE parallel executable.

Prior to going through these tutorials, you should have \ref
installation "installed SpECTRE" and built it successfully.

| Tutorial name | Concepts introduced |
|---------------|---------------------|
| \subpage tutorial_parallel_concepts "Parallelism in SpECTRE " | SpECTRE and Charm++ |
| \subpage tutorial_minimal_parallel_executable "Minimal executable " | Metavariables, Main component |

See also: \subpage ParallelInfoExecutablePage
