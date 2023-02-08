// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/Printf.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/TMPL.hpp"

// Charm looks for this function but since we build without a main function or
// main module supplied by Charm++, we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
void dump_barotropic_eos(
    const EquationsOfState::EquationOfState<true, 1>& eos,
    const size_t number_of_log10_number_density_points_for_dump,
    const std::string& output_file_name,
    const double lower_bound_rest_mass_density_cgs,
    const double upper_bound_rest_mass_density_cgs) {
  using std::log10;
  using std::pow;
  // Baryon mass, used to go from number density to rest mass
  // density. I.e. `rho_cgs = n_cgs * baryon_mass`, where `n_gcs` is the number
  // density in CGS units. This is the baryon mass that RotNS uses. This
  // might be different from the baryon mass that the EoS uses.
  //
  // https://github.com/sxs-collaboration/spectre/issues/4694
  const double baryon_mass_of_rotns_cgs = 1.659e-24;
  const double log10_lower_bound_number_density_cgs =
      log10(lower_bound_rest_mass_density_cgs / baryon_mass_of_rotns_cgs);
  const double log10_upper_bound_number_density_cgs =
      log10(upper_bound_rest_mass_density_cgs / baryon_mass_of_rotns_cgs);
  const double delta_log_number_density_cgs =
      (log10_upper_bound_number_density_cgs -
       log10_lower_bound_number_density_cgs) /
      static_cast<double>(number_of_log10_number_density_points_for_dump - 1);

  if (file_system::check_if_file_exists(output_file_name)) {
    ERROR_NO_TRACE("File " << output_file_name
                           << " already exists. Refusing to overwrite.");
  }
  std::ofstream outfile(output_file_name.c_str());

  for (size_t log10_number_density_index = 0;
       log10_number_density_index <
       number_of_log10_number_density_points_for_dump;
       ++log10_number_density_index) {
    using std::pow;
    const double number_density_cgs =
        pow(10.0, log10_lower_bound_number_density_cgs +
                      static_cast<double>(log10_number_density_index) *
                          delta_log_number_density_cgs);
    const double rest_mass_density_cgs =
        number_density_cgs * baryon_mass_of_rotns_cgs;

    // Note: we will want to add the baryon mass to our EOS interface.
    //
    // https://github.com/sxs-collaboration/spectre/issues/4694
    const double baryon_mass_of_eos_cgs = baryon_mass_of_rotns_cgs;
    const Scalar<double> rest_mass_density_geometric{rest_mass_density_cgs /
                                                     baryon_mass_of_eos_cgs};
    const Scalar<double> pressure_geometric =
        eos.pressure_from_density(rest_mass_density_geometric);
    const Scalar<double> specific_internal_energy_geometric =
        eos.specific_internal_energy_from_density(rest_mass_density_geometric);
    const Scalar<double> total_energy_density_geometric{
        get(rest_mass_density_geometric) *
        (1.0 + get(specific_internal_energy_geometric))};

    // Note: the energy density is divided by c^2, so the rest-mass part is rho
    // c^2
    const double total_energy_density_cgs =
        get(total_energy_density_geometric) *
        hydro::units::cgs::rest_mass_density_unit;

    // should be dyne cm^(-3)
    const double pressure_cgs =
        get(pressure_geometric) * hydro::units::cgs::pressure_unit;

    outfile << std::scientific << std::setw(24) << std::setprecision(14)
            << log10(number_density_cgs) << std::setw(24)
            << std::setprecision(14) << log10(total_energy_density_cgs)
            << std::setw(24) << std::setprecision(14) << log10(pressure_cgs)
            << std::endl;
  }
  outfile.close();
}

namespace OptionTags {
struct NumberOfPoints {
  using type = size_t;
  static constexpr Options::String help = {
      "Number of points at which to dump the EoS"};
};

struct OutputFileName {
  using type = std::string;
  static constexpr Options::String help = {
      "Name of the output file to dump the EoS to, including file extension."};
};

struct LowerBoundRestMassDensityCgs {
  using type = double;
  static constexpr Options::String help = {
      "Lower bound of rest mass density in CGS units."};
};

struct UpperBoundRestMassDensityCgs {
  using type = double;
  static constexpr Options::String help = {
      "Upper bound of rest mass density in CGS units."};
};
}  // namespace OptionTags
}  // namespace

int main(int argc, char** argv) {
  namespace bpo = boost::program_options;
  bpo::positional_options_description pos_desc;

  const std::string help_string =
      "Dump a relativistic barotropic equation of state to disk.\n"
      "All options controlling input and output are read from the input file.\n"
      "This executable can be extended to support 2d and 3d EoS, where "
      "temperature and electron fraction dependence is available.";

  bpo::options_description desc(help_string);
  desc.add_options()("help,h,", "show this help message")(
      "input-file", bpo::value<std::string>()->required(), "Input file name")(
      "check-options", "Check input file options");

  bpo::variables_map vars;

  bpo::store(bpo::command_line_parser(argc, argv)
                 .positional(pos_desc)
                 .options(desc)
                 .run(),
             vars);

  if (vars.count("help") != 0u or vars.count("input-file") == 0u) {
    Parallel::printf("%s\n", desc);
    return 1;
  }

  using option_list =
      tmpl::list<hydro::OptionTags::EquationOfState<true, 1>,
                 OptionTags::NumberOfPoints, OptionTags::OutputFileName,
                 OptionTags::LowerBoundRestMassDensityCgs,
                 OptionTags::UpperBoundRestMassDensityCgs>;

  Options::Parser<option_list> option_parser(help_string);
  option_parser.parse_file(vars["input-file"].as<std::string>());

  if (vars.count("check-options") != 0) {
    // Force all the options to be created.
    option_parser.template apply<option_list>([](auto... args) {
      (void)std::initializer_list<char>{((void)args, '0')...};
    });
    Parallel::printf("\n%s parsed successfully!\n",
                     vars["input-file"].as<std::string>());

    return 0;
  }

  const auto options =
      option_parser.template apply<option_list>([](auto... args) {
        return tuples::tagged_tuple_from_typelist<option_list>(
            std::move(args)...);
      });

  dump_barotropic_eos(
      *get<hydro::OptionTags::EquationOfState<true, 1>>(options),
      get<OptionTags::NumberOfPoints>(options),
      get<OptionTags::OutputFileName>(options),
      get<OptionTags::LowerBoundRestMassDensityCgs>(options),
      get<OptionTags::UpperBoundRestMassDensityCgs>(options));

  return 0;
}
