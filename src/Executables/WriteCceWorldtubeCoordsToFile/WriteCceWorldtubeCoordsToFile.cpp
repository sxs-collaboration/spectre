// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <boost/program_options.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <cstddef>
#include <exception>
#include <string>

#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/StrahlkorperCoordsToTextFile.hpp"
#include "Parallel/Printf/Printf.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

/*
 * This executable is used for writing the collocation points of a CCE
 * Strahlkorper to a text file. This is useful for other NR codes so they can
 * write data to certain points and we do the necessary transformations.
 */
int main(int argc, char** argv) {
  boost::program_options::options_description desc(
      "Program for writing the collocation points (coordinates) of a worldtube "
      "sphere that SpECTRE CCE is able to read and interpret to a "
      "file.\nDetails about the output file:\n"
      " * There are no header or comment lines\n"
      " * Each point is written to a new line of the output file\n"
      " * The delimiter for the x, y, z components of each point is a space\n"
      " * The points are written in scientific notation\n"
      " * The sphere is centered on the origin (0, 0, 0)");
  desc.add_options()("help,h", "show this help message")(
      "radius,r", boost::program_options::value<double>()->required(),
      "Radius of the worltube.")(
      "lmax,L", boost::program_options::value<size_t>()->required(),
      "The spherical harmonic L of the surface. The surface will have (L + 1) "
      "* (2L + 1) total points")(
      "output_file,o", boost::program_options::value<std::string>()->required(),
      "Output filename for the points. No extension will be added.")(
      "force,f", boost::program_options::bool_switch(),
      "Overwrite the output file if it already exists");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .options(desc)
          .run(),
      vars);

  if (vars.contains("help")) {
    Parallel::printf("%s\n", desc);
    return 0;
  }

  for (const auto& option : {"radius", "lmax", "output_file"}) {
    if (not vars.contains(option)) {
      Parallel::printf("Missing option: %s\n\n%s", option, desc);
      return 1;
    }
  }

  const double radius = vars["radius"].as<double>();
  const size_t l_max = vars["lmax"].as<size_t>();
  const std::array<double, 3> center{0.0, 0.0, 0.0};
  const std::string output_file = vars["output_file"].as<std::string>();
  const bool overwrite = vars["force"].as<bool>();

  try {
    ylm::write_strahlkorper_coords_to_text_file(
        radius, l_max, center, output_file, ylm::AngularOrdering::Cce,
        overwrite);
  } catch (const std::exception& exception) {
    Parallel::printf("%s\n", exception.what());
    return 1;
  }
}
