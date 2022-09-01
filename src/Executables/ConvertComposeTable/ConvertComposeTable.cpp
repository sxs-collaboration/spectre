// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>

#include <string>

#include "DataStructures/DataVector.hpp"
#include "IO/ComposeTable.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "Parallel/Printf.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

void convert_file(const std::string& compose_directory,
                  const std::string& spectre_eos_filename,
                  const std::string& spectre_eos_subfile) {
  const io::ComposeTable compose_table(compose_directory);
  h5::H5File<h5::AccessType::ReadWrite> spectre_file(spectre_eos_filename);
  auto& spectre_eos = spectre_file.insert<h5::EosTable>(
      spectre_eos_subfile,
      std::vector<std::string>{"number density", "temperature",
                               "electron fraction"},
      std::vector{compose_table.number_density_bounds(),
                  compose_table.temperature_bounds(),
                  compose_table.electron_fraction_bounds()},
      std::vector{compose_table.number_density_number_of_points(),
                  compose_table.temperature_number_of_points(),
                  compose_table.electron_fraction_number_of_points()},
      std::vector{compose_table.number_density_log_spacing(),
                  compose_table.temperature_log_spacing(),
                  compose_table.electron_fraction_log_spacing()},
      compose_table.beta_equilibrium());

  // Now dump data into the EosTable file
  for (const auto& [quantity_name, quantity_data] : compose_table.data()) {
    spectre_eos.write_quantity(quantity_name, quantity_data);
  }
}

int main(int argc, char** argv) {
  namespace bpo = boost::program_options;
  try {
    bpo::options_description command_line_options;

    // clang-format off
    command_line_options.add_options()
        ("help,h", "Describe program options.\nThis executable converts an "
         "ASCII formatted CompOSE 3d equation of state table into a "
         "SpECTRE-formatted HDF5 table. This reduces the file size by about a "
         "factor of 4. We don't use the CompOSE HDF5 tables since that "
         "requires an HDF5 that works with Fortran.\n"
         "Note: support for 1d and 2d tables can be added if the CompOSE ASCII "
         "reader is generalized to support them.")
        ("compose-directory", bpo::value<std::string>(),
         "The directory in which the CompOSE eos.quantities, eos.parameters, "
         "and eos.table files are.")
        ("eos-subfile", bpo::value<std::string>(),
         "Path of where to write the subfile SpECTRE EOS Table inside the "
         "HDF5 file.")
        ("output,o", bpo::value<std::string>(),
         "Path of the output HDF5 file to which the EOS subfile will be "
         "written")
        ;
    // clang-format on

    bpo::command_line_parser command_line_parser(argc, argv);
    command_line_parser.options(command_line_options);

    bpo::variables_map parsed_command_line_options;
    bpo::store(command_line_parser.run(), parsed_command_line_options);
    bpo::notify(parsed_command_line_options);

    if (parsed_command_line_options.count("help") != 0 or
        parsed_command_line_options.count("compose-directory") == 0 or
        parsed_command_line_options.count("output") == 0 or
        parsed_command_line_options.count("eos-subfile") == 0) {
      Parallel::printf("%s\n", command_line_options);
      return 1;
    }
    convert_file(
        parsed_command_line_options.at("compose-directory").as<std::string>(),
        parsed_command_line_options.at("output").as<std::string>(),
        parsed_command_line_options.at("eos-subfile").as<std::string>());
  } catch (const bpo::error& e) {
    ERROR(e.what());
  }
  return 0;
}
