// Distributed under the MIT License.
// See LICENSE.txt for details.

/// [gnu_versions_example]
#ifdef COMPILATION_TEST_TEST_DIFFERENT_COMPILERS
// [[TAGS: unit, CompilationTest]]

// [[COMPILER: GNU:0.0.0 REGEX: assert with GCC 5]]
// [[COMPILER: GNU:6.0.0 REGEX: assert with GCC 6 or newer]]
// [[COMPILER: Clang REGEX: assert with Clang]]
// [[COMPILER: AppleClang REGEX: assert with AppleClang]]

#ifdef __APPLE__
static_assert(false, "assert with AppleClang");
#else  // __APPLE__
#ifdef __clang__
static_assert(false, "assert with Clang");
#else  // __clang__
#if __GNUC__ < 6
static_assert(false, "assert with GCC 5");
#else   // __GNUC__ < 6
static_assert(false, "assert with GCC 6 or newer");
#endif  // __GNUC__ < 6
#endif  // __clang__
#endif  // __APPLE__

int main() {}
#endif
/// [gnu_versions_example]

/// [compilation_test_example]
#ifdef COMPILATION_TEST_TEST_FRAMEWORK_WORKS
// [[TAGS: unit, CompilationTest]]

// [[COMPILER: all REGEX: Testing compilation failure tests]]

static_assert(false, "Testing compilation failure tests");

int main() {}

#endif

FILE_IS_COMPILATION_TEST
/// [compilation_test_example]
