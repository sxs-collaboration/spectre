// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "IO/ComposeTable.hpp"
#include "IO/ComposeTableDerivatives.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
std::vector<double> make_grid_1d(const std::array<double, 2>& bounds,
                                 const size_t npts, const bool log_spacing) {
  std::vector<double> grid(npts);
  if (npts == 0) {
    ERROR("Number of points is zero.");
  }
  if (npts == 1) {
    grid[0] = bounds[0];
    return grid;
  }
  if (log_spacing) {
    const double log_lo = std::log(bounds[0]);
    const double log_hi = std::log(bounds[1]);
    const double dlog = (log_hi - log_lo) / static_cast<double>(npts - 1);
    for (size_t i = 0; i < npts; ++i) {
      grid[i] = std::exp(log_lo + dlog * static_cast<double>(i));
    }
  } else {
    const double d = (bounds[1] - bounds[0]) / static_cast<double>(npts - 1);
    for (size_t i = 0; i < npts; ++i) {
      grid[i] = bounds[0] + d * static_cast<double>(i);
    }
  }
  return grid;
}

void convert_file(const std::string& compose_directory,
                  const std::string& spectre_eos_filename,
                  const std::string& spectre_eos_subfile) {
  const io::ComposeTable compose_table(compose_directory);
  h5::H5File<h5::AccessType::ReadWrite> spectre_file(spectre_eos_filename,
                                                     true);
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

  const auto& data = compose_table.data();

  // Required base quantities
  const DataVector& pressure = data.at("pressure");
  const DataVector& eps = data.at("specific internal energy");

  const size_t nN = compose_table.number_density_number_of_points();
  const size_t nT = compose_table.temperature_number_of_points();
  const size_t nYe = compose_table.electron_fraction_number_of_points();
  const size_t ntot = nN * nT * nYe;

  ASSERT(pressure.size() == ntot,
         "Pressure size does not match table dimensions.");
  ASSERT(eps.size() == pressure.size(),
         "Epsilon size does not match pressure size.");

  // Free-energy derivatives used to compute zeta analytically (CompOSE Table
  // 7.3 indices 3,4,5,8,9). They are inputs only.
  const std::array<std::string, 5> zeta_derivative_keys{
      {"d2 F / d T2", "d2 F / d T d n_b", "d2 F / d T d Y_e",
       "d2 F / d n_b d Y_e", "d F / d Y_e"}};
  const bool have_derivatives = std::all_of(
      zeta_derivative_keys.begin(), zeta_derivative_keys.end(),
      [&data](const std::string& key) { return data.contains(key); });

  // Write everything available from the CompOSE table except those derivatives.
  for (const auto& [quantity_name, quantity_data] : data) {
    if (std::find(zeta_derivative_keys.begin(), zeta_derivative_keys.end(),
                  quantity_name) != zeta_derivative_keys.end()) {
      continue;
    }
    spectre_eos.write_quantity(quantity_name, quantity_data);
  }

  // kappa = dp/dε (CompOSE Q11): if it wasn't requested in eos.quantities,
  // fall back to zeros so the H5 format stays uniform and Tabulated3D can
  // still read it.
  if (not data.contains("kappa")) {
    Parallel::printf(
        "WARNING: source CompOSE table does not contain kappa (Q11 dp/dε); "
        "writing kappa = 0. Regenerate with index 11 in eos.quantities to "
        "get the true value.\n");
    spectre_eos.write_quantity("kappa", DataVector(ntot, 0.0));
  }

  // zeta = ∂P/∂Ye is computed analytically from the CompOSE free-energy
  // derivatives (Table 7.3 indices 3,4,5,8,9). It is ill-defined for
  // beta-equilibrium tables (Ye is fixed by beta equilibrium rather than free).
  // In either the beta-equilibrium case or when those derivatives weren't
  // tabulated we cannot compute it, so — as with kappa — emit a warning and
  // write zeros to keep the H5 format uniform.
  if (compose_table.beta_equilibrium()) {
    Parallel::printf(
        "WARNING: source CompOSE table is flagged beta-equilibrium; writing "
        "zeta = 0 (∂P/∂Ye is undefined when Ye is fixed by beta "
        "equilibrium).\n");
    spectre_eos.write_quantity("zeta", DataVector(ntot, 0.0));
  } else if (not have_derivatives) {
    Parallel::printf(
        "WARNING: source CompOSE table does not contain the free-energy "
        "derivatives (Table 7.3 indices 3,4,5,8,9) needed for zeta = "
        "(∂P/∂Ye)_{rho,eps}; writing zeta = 0. Regenerate with derivative "
        "indices 3 4 5 8 9 in eos.quantities to get the true value.\n");
    spectre_eos.write_quantity("zeta", DataVector(ntot, 0.0));
  } else {
    const auto T_grid = make_grid_1d(compose_table.temperature_bounds(), nT,
                                     compose_table.temperature_log_spacing());
    const auto nb_grid =
        make_grid_1d(compose_table.number_density_bounds(), nN,
                     compose_table.number_density_log_spacing());
    spectre_eos.write_quantity(
        "zeta", io::compute_zeta_from_free_energy_derivatives(
                    data.at("d2 F / d T2"), data.at("d2 F / d T d n_b"),
                    data.at("d2 F / d T d Y_e"), data.at("d2 F / d n_b d Y_e"),
                    data.at("d F / d Y_e"), nb_grid, T_grid, nN, nT, nYe));
  }
}
}  // namespace

int main(int argc, char** argv) {
  namespace bpo = boost::program_options;
  try {
    bpo::options_description command_line_options(
        "This executable converts an ASCII formatted CompOSE 3d equation of "
        "state table into a SpECTRE-formatted HDF5 table. This reduces the "
        "file size by about a factor of 4. We don't use the CompOSE HDF5 "
        "tables since that requires an HDF5 that works with Fortran.\n"
        "Note: support for 1d and 2d tables can be added if the CompOSE ASCII "
        "reader is generalized to support them.\n\n"
        "Generating the ASCII table using compose:\n"
        "1.\n"
        "Download from: https://compose.obspm.fr/software (there's a "
        "GitLab link)\n\n"
        "2.\n"
        "Build by running 'make' in the directory. This process will create "
        "the 'compose' executable.\n\n"
        "3.\n"
        "Download EOS from https://compose.obspm.fr/table We will use "
        "https://compose.obspm.fr/eos/34 as an example. To download, use wget "
        "on the link from the 'eos.zip' file, or download the 'eos.zip' file "
        "directly.\n\n"
        "4.\n"
        "Unzip the eos.zip file. This will create multiple 'eos.*' files in "
        "the current directory.\n\n"
        "5.\n"
        "Run the 'compose' executable in the directory with all the eos "
        "files. There are 3 main options and you will run the executable "
        "3 times. Each main option or 'task' has a bunch of numerical value "
        "inputs."
        "\n"
        "Task 1\n"
        "How many regular thermodynamic quantities...\n"
        "8\n"
        "Please select the indices of the thermodynamic quantities...\n"
        " Index #           1 ?"
        "1\n"
        " Index #           2 ?"
        "2\n"
        "The remaining are: 3 4 5 7 11 12\n"
        "(Index 11 is dp/dε, stored as 'kappa' in SpECTRE and required by "
        "Tabulated3D.)\n"
        "The following function values and derivatives of the free energy...\n"
        "5\n"
        "Please select the indices of the thermodynamic...\n"
        "3 4 5 8 9\n"
        "(These are d2F/dT2, d2F/dTdn_b, d2F/dTdY_e, d2F/dn_bdY_e, and "
        "dF/dY_e. SpECTRE uses them to compute the bulk-viscosity-like "
        "quantity zeta = (dp/dY_e)_{rho,eps} analytically. If they are "
        "omitted, zeta is written as zeros, with a warning, like kappa.)\n"
        "How many particles do you want to select for the file eos.table?\n"
        "0\n"
        "There are average mass, charge and neutron numbers...\n"
        "0\n"
        "There are microscopic data available of the following type:...\n"
        "0\n"
        "There are error estimates available of the following type:...\n"
        "0\n"
        "If successful, you should see new file 'eos.quantities' generated. "
        "Now rerun compose for Task2.\n\n"
        "Task 2\n"
        "Temperature interpolation order:\n"
        "3\n"
        "Baryon density interpolation order:\n"
        "3\n"
        "Hadronic charge fraction interpolation order:\n"
        "3\n"
        "beta-equilibrium\n"
        "0\n"
        "entropy per baryon\n"
        "0\n"
        "Please select the tabulation scheme for the parameters from\n"
        "1\n"
        "Get the lower and upper bounds as well as the grid points from the "
        "compose website for your EOS. Spacing should be\n"
        "T: log\n"
        "n_b: log\n"
        "Y_q: linear\n"
        "You must enter the bounds as:\n"
        "lower upper\n"
        "If successful, you should see new file 'eos.parameters' generated. "
        "Now rerun compose for Task3.\n\n"
        "Task 3\n"
        "This will just run, no options needed, but it can take quite a long "
        "time.\n"
        "If successful, it will list 'file eos.table written', along with the "
        "respective labels [and units] of the columns, for example, '1 "
        "temperature T [MeV]'.\n\n"
        "Available options are");

    // clang-format off
    command_line_options.add_options()
        ("help,h", "Describe program options.\n")
        ("compose-directory", bpo::value<std::string>(),
         "The directory in which the CompOSE eos.quantities, eos.parameters, "
         "and eos.table files are.")
        ("eos-subfile", bpo::value<std::string>(),
         "Path of where to write the subfile SpECTRE EOS Table inside the "
         "HDF5 file.")
        ("output,o", bpo::value<std::string>(),
         "Path of the output HDF5 file to which the EOS subfile will be "
         "written, including the .h5 extension.")
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
