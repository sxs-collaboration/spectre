\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Developer Guides {#dev_guide}

### Developing and Improving Executables
- \subpage spectre_build_system "Build system" and how to add dependencies, unit
  tests, and executables.
- \subpage dev_guide_creating_executables "Executables and how to add them"
- \subpage tutorials_parallel - A series of tutorials demonstrating how to write
  a parallel executable, and explaining some of the metaprogramming that turns
  user-provided code into a SpECTRE parallel executable
- \subpage dev_guide_option_parsing "Option parsing" to get options from input
  files
- \subpage dev_guide_importing "Importing" data from files
- \subpage profiling_with_projections "Profiling With Charm++ Projections" and
  PAPI for optimizing performance
- \subpage spectre_writing_python_bindings "Writing Python Bindings" to use
  SpECTRE C++ classes and functions from within python.
- \subpage implementing_vectors "Implementing SpECTRE vectors" a quick how-to
  for making new generalizations of DataVectors
- \subpage compiler_and_linker_errors "How to parse linker and compiler errors"
- \subpage static_analysis_tools "Static analysis tools"
- \subpage build_profiling_and_optimization - Getting started with improving
  compilation time and memory use

### Having your Contributions Merged into SpECTRE
- \subpage writing_good_dox "Writing good documentation" is key for long term
  maintainability of the project.
- \subpage writing_unit_tests "Writing Unit Tests" to catch bugs and to make
  sure future changes don't cause your code to regress.
- \subpage travis_guide "Travis CI" is used to test every pull request.
- \subpage code_review_guide "Code review guidelines." All code merged into
  develop must follow these requirements.

### General SpECTRE Terminology
Terms with SpECTRE-specific meanings are defined here.
- \subpage domain_concepts "Domain Concepts" used throughout the code are
  defined here for reference.

### Charm++ Interface
- \subpage load_balancing_notes "Load Balancing Notes and Recommendations"

### Template Metaprogramming (TMP)
Explanations for TMP concepts and patterns known to the greater C++ community
can be found here.
- \subpage sfinae "SFINAE"

### Foundational Concepts in SpECTRE
Designed to give the reader an introduction to SpECTRE's most recurring concepts
and patterns.
- \subpage databox_foundations "Towards SpECTRE's DataBox"
- \subpage protocols "Protocols: metaprogramming interfaces"

### Technical Documentation for Fluent Developers
Assumes a thorough familiarity and fluency in SpECTRE's usage of TMP.
- \subpage DataBoxGroup "DataBox"
- \subpage observers_infrastructure_dev_guide "Observers infrastructure"
- \subpage dev_guide_parallelization_foundations - Parallelization
  infrastructure components and usage

### CoordinateMap Guide
Methods for creating custom coordinate maps are discussed here.
- \subpage redistributing_gridpoints "Methods for redistributing gridpoints"

### Continuous Integration
Explanations on our automated tests and deployments can be found here.

- \subpage dev_guide_automatic_versioning
