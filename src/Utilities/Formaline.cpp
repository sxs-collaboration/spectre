// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/Formaline.hpp"

#include <cstdio>
#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

// NOTE: The definitions of the functions `formaline::get_archive`,
// `formaline::get_environment_variables`, `formaline::get_build_info`,
// and `formaline::get_paths` are generated at link time and defined in the
// script `tools/Formaline.sh` for non-macOS builds.
namespace formaline {
void write_to_file(const std::string& filename_without_extension) noexcept {
  std::FILE* outfile = nullptr;
  const auto archive = get_archive();

  const std::string filename = filename_without_extension + ".tar.gz";
  outfile = std::fopen(filename.c_str(), "w");
  if (outfile == nullptr) {
    ERROR("Failed to open file '" << filename << "' for Formaline output");
  }
  std::fwrite(archive.data(), sizeof(char), archive.size(), outfile);
  std::fclose(outfile);
}

#ifdef SPECTRE_NO_FORMALINE
std::vector<char> get_archive() noexcept {
  return {'N', 'o', 't', ' ', 's', 'u', 'p', 'p', 'o', 'r', 't', 'e', 'd'};
}

std::string get_environment_variables() noexcept {
  return "Not supported on macOS";
}

std::string get_build_info() noexcept { return "Not supported on macOS"; }

std::string get_paths() noexcept { return "Not supported on macOS"; }
#endif  // defined(__APPLE__)
}  // namespace formaline
