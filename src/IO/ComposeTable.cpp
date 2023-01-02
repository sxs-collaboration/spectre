// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/ComposeTable.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"

namespace io {
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
ComposeTable::ComposeTable(std::string directory_to_read_from)
    : directory_to_read_from_(std::move(directory_to_read_from)) {
  parse_eos_quantities();
  parse_eos_parameters();
  parse_eos_table();
}

void ComposeTable::parse_eos_quantities() {
  std::string line_buffer{};
  const std::string filename{directory_to_read_from_ + "/eos.quantities"};
  if (not file_system::check_if_file_exists(filename)) {
    ERROR("File '" << filename << "' does not exist.");
  }

  std::ifstream quantities_file(filename);

  std::getline(quantities_file, line_buffer);  // Read first comment line
  if (line_buffer !=
      " # number of regular, additional and derivative quantities (see table "
      "7.1)") {
    ERROR("Read unexpected comment line: '" << line_buffer << "'");
  }
  size_t number_of_regular_quantities = 0;
  size_t number_of_additional_quantities = 0;
  size_t number_of_derivative_quantities = 0;
  quantities_file >> number_of_regular_quantities >>
      number_of_additional_quantities >> number_of_derivative_quantities;
  std::getline(quantities_file, line_buffer);  // Read newline
  std::getline(quantities_file, line_buffer);  // Read second comment line
  if (line_buffer !=
      " # indices of regular, additional and derivative quantities") {
    ERROR("Read unexpected comment line: '" << line_buffer << "'");
  }
  const size_t number_of_regular_and_additional_quantities =
      number_of_regular_quantities + number_of_additional_quantities;
  available_quantities_.reserve(number_of_regular_and_additional_quantities +
                                number_of_derivative_quantities);
  data_.reserve(number_of_regular_and_additional_quantities +
                number_of_derivative_quantities);

  for (size_t i = 0; i < number_of_regular_and_additional_quantities; ++i) {
    size_t quantity = std::numeric_limits<size_t>::max();
    quantities_file >> quantity;
    if (quantity > compose_regular_and_additional_index_to_names_.size() or
        quantity < 1) {
      ERROR("Read in unknown quantity with number "
            << quantity << " but only know [1, "
            << compose_regular_and_additional_index_to_names_.size() << "]");
    }
    available_quantities_.emplace_back(
        compose_regular_and_additional_index_to_names_[quantity - 1]);
  }
  for (size_t i = 0; i < number_of_derivative_quantities; ++i) {
    size_t quantity = std::numeric_limits<size_t>::max();
    quantities_file >> quantity;
    if (quantity > compose_derivative_index_to_names_.size() or quantity < 1) {
      ERROR("Read in unknown quantity with number "
            << quantity << " but only know [1, "
            << compose_derivative_index_to_names_.size() << "]");
    }
    available_quantities_.emplace_back(
        compose_derivative_index_to_names_[quantity - 1]);
  }
  // Make sure we don't have any duplicate quantities
  std::vector<std::string> quantities_to_check = available_quantities();
  alg::sort(quantities_to_check);
  if (const auto adjacent_it = std::adjacent_find(quantities_to_check.begin(),
                                                  quantities_to_check.end());
      adjacent_it != quantities_to_check.end()) {
    ERROR("Found quantity '"
          << *adjacent_it
          << "' more than once. If you requested the free energy from both the "
             "'regular and additional quantities' and the 'derivative "
             "quantities' you can only use one.");
  }
}

void ComposeTable::parse_eos_parameters() {
  std::string line_buffer{};
  const std::string filename{directory_to_read_from_ + "/eos.parameters"};
  if (not file_system::check_if_file_exists(filename)) {
    ERROR("File '" << filename << "' does not exist.");
  }
  std::ifstream parameters_file(filename);

  std::getline(parameters_file, line_buffer);  // Read first comment line
  if (line_buffer !=
      " # order of interpolation in first, second and third index") {
    ERROR("Read unexpected comment line: '" << line_buffer << "'");
  }
  for (size_t i = 0; i < 3; ++i) {
    parameters_file >> gsl::at(interpolation_order_, i);
  }
  std::getline(parameters_file, line_buffer);  // Read newline
  std::getline(parameters_file, line_buffer);  // Read second comment line
  if (line_buffer !=
      " # calculation of beta-equilibrium (1: yes, else: no) and for given "
      "entropy (1: yes, else: no)") {
    ERROR("Read unexpected comment line: '" << line_buffer << "'");
  }
  parameters_file >> beta_equilibrium_;
  std::getline(parameters_file, line_buffer);  // Read rest of line
  std::getline(parameters_file, line_buffer);  // Read third comment line
  if (line_buffer !=
      " # tabulation scheme (0 = explicit listing, 1 = loops, see manual)") {
    ERROR("Read unexpected comment line: '" << line_buffer << "'");
  }
  std::getline(parameters_file, line_buffer);  // Read tabulation scheme
  std::getline(parameters_file, line_buffer);  // Read fourth comment line
  if (line_buffer !=
      " # parameter values (first, second and third index) depending on "
      "tabulation scheme") {
    ERROR("Read unexpected comment line: '" << line_buffer << "'");
  }
  parameters_file >> temperature_bounds_[0] >> number_density_bounds_[0] >>
      electron_fraction_bounds_[0];
  std::getline(parameters_file, line_buffer);  // Read rest of line (newline)
  parameters_file >> temperature_bounds_[1] >> number_density_bounds_[1] >>
      electron_fraction_bounds_[1];
  std::getline(parameters_file, line_buffer);  // Read rest of line (newline)
  parameters_file >> temperature_number_of_points_ >>
      number_density_number_of_points_ >> electron_fraction_number_of_points_;
  table_size_ = number_density_number_of_points_ *
                temperature_number_of_points_ *
                electron_fraction_number_of_points_;
  std::getline(parameters_file, line_buffer);  // Read rest of line (newline)
  parameters_file >> temperature_log_spacing_ >> number_density_log_spacing_ >>
      electron_fraction_log_spacing_;
}

void ComposeTable::parse_eos_table() {
  for (const auto& quantity_name : available_quantities()) {
    data_[quantity_name] = DataVector{table_size_};
  }

  std::string line_buffer{};
  const std::string filename{directory_to_read_from_ + "/eos.table"};
  if (not file_system::check_if_file_exists(filename)) {
    ERROR("File '" << filename << "' does not exist.");
  }
  std::ifstream table_file(filename);

  for (size_t i = 0; i < table_size_; ++i) {
    // Read in number density, temperature, and electron fraction
    double dummy = std::numeric_limits<double>::signaling_NaN();
    table_file >> dummy >> dummy >> dummy;
    for (const auto& quantity_name : available_quantities()) {
      table_file >> data_[quantity_name][i];
    }
  }
}

void ComposeTable::pup(PUP::er& p) {
  p | directory_to_read_from_;
  p | available_quantities_;
  p | interpolation_order_;
  p | number_density_bounds_;
  p | temperature_bounds_;
  p | electron_fraction_bounds_;
  p | number_density_number_of_points_;
  p | temperature_number_of_points_;
  p | electron_fraction_number_of_points_;
  p | table_size_;
  p | beta_equilibrium_;
  p | number_density_log_spacing_;
  p | temperature_log_spacing_;
  p | electron_fraction_log_spacing_;
  p | data_;
}

const std::vector<std::string>
    ComposeTable::compose_regular_and_additional_index_to_names_{
        "pressure",
        "specific entropy",
        "baryon chemical potential",
        "charge chemical potential",
        "lepton chemical potential",
        "specific free energy",
        "specific internal energy",
        "specific enthalpy",
        "specific free enthalpy",
        "dp_drho",
        "dp_depsilon",
        "sound speed squared",
        "specific heat at constant volume",
        "specific heat at constant pressure",
        "adiabatic index",
        "expansion coefficient at constant pressure",
        "tension coefficient at constant volume",
        "isothermal compressibility",
        "adiabatic compressibility",
        "free energy",
        "internal energy",
        "enthalpy",
        "free enthalpy",
        "energy density"};

const std::vector<std::string> ComposeTable::compose_derivative_index_to_names_{
    "free energy",      "d F / d T",    "d2 F / d T2",   "d2 F / d T d n_b",
    "d2 F / d T d Y_e", "d F / d n_b",  "d2 F / d n_b2", "d2 F / d n_b d Y_e",
    "d F / d Y_e",      "d2 F / d Y_e2"};
}  // namespace io
