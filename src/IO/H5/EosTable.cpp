// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/EosTable.hpp"

#include <array>
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/Version.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace h5 {
EosTable::EosTable(
    const bool subfile_exists, detail::OpenGroup&& group,
    const hid_t /*location*/, const std::string& name,
    std::vector<std::string> independent_variable_names,
    std::vector<std::array<double, 2>> independent_variable_bounds,
    std::vector<size_t> independent_variable_number_of_points,
    std::vector<bool> independent_variable_uses_log_spacing,
    const bool beta_equilibrium, const uint32_t version)
    : group_(std::move(group)),
      name_(name.size() > extension().size()
                ? (extension() == name.substr(name.size() - extension().size())
                       ? name
                       : name + extension())
                : name + extension()),
      path_(group_.group_path_with_trailing_slash() + name),
      version_(version),
      eos_table_group_(group_.id(), name_, h5::AccessType::ReadWrite),
      independent_variable_names_(std::move(independent_variable_names)),
      independent_variable_bounds_(std::move(independent_variable_bounds)),
      independent_variable_number_of_points_(
          std::move(independent_variable_number_of_points)),
      independent_variable_uses_log_spacing_(
          std::move(independent_variable_uses_log_spacing)),
      beta_equilibrium_(beta_equilibrium) {
  if (subfile_exists) {
    ERROR(
        "Opening an equation of state table with the constructor for writing a "
        "table, but the subfile "
        << name << " already exists");
  }
  // Subfiles are closed as they go out of scope, so we have the extra
  // braces here to add the necessary scope
  {
    Version open_version(false, detail::OpenGroup{}, eos_table_group_.id(),
                         "version", version_);
  }
  {
    Header header(false, detail::OpenGroup{}, eos_table_group_.id(), "header");
    header_ = header.get_header();
  }

  h5::write_to_attribute(eos_table_group_.id(), "independent variable names",
                         independent_variable_names_);
  h5::write_to_attribute(eos_table_group_.id(), "independent variable bounds",
                         independent_variable_bounds_);
  h5::write_to_attribute(eos_table_group_.id(), "number of points",
                         independent_variable_number_of_points_);
  h5::write_to_attribute(eos_table_group_.id(), "uses log spacing",
                         independent_variable_uses_log_spacing_);
  h5::write_to_attribute(eos_table_group_.id(), "beta equilibium",
                         beta_equilibrium_);
}

EosTable::EosTable(const bool /*subfile_exists*/, detail::OpenGroup&& group,
                   const hid_t /*location*/, const std::string& name)
    : group_(std::move(group)),
      name_(name.size() > extension().size()
                ? (extension() == name.substr(name.size() - extension().size())
                       ? name
                       : name + extension())
                : name + extension()),
      path_(group_.group_path_with_trailing_slash() + name),
      eos_table_group_(group_.id(), name_, h5::AccessType::ReadOnly) {
  // We treat this as an internal version for now. We'll need to deal with
  // proper versioning later.
  const Version open_version(true, detail::OpenGroup{}, eos_table_group_.id(),
                             "version");
  version_ = open_version.get_version();
  const Header header(true, detail::OpenGroup{}, eos_table_group_.id(),
                      "header");
  header_ = header.get_header();

  independent_variable_names_ = h5::read_rank1_attribute<std::string>(
      eos_table_group_.id(), "independent variable names");
  independent_variable_bounds_ = h5::read_rank1_array_attribute<double, 2>(
      eos_table_group_.id(), "independent variable bounds");
  independent_variable_number_of_points_ = h5::read_rank1_attribute<size_t>(
      eos_table_group_.id(), "number of points");
  independent_variable_uses_log_spacing_ =
      h5::read_rank1_attribute<bool>(eos_table_group_.id(), "uses log spacing");
  beta_equilibrium_ =
      h5::read_value_attribute<bool>(eos_table_group_.id(), "beta equilibium");
  // Note: if we start writing more datasets than just the dependent variables,
  // this list will need to be pruned.
  available_quantities_ = h5::get_group_names(eos_table_group_.id(), {});
}

void EosTable::write_quantity(std::string name, const DataVector& data) {
  h5::write_data(eos_table_group_.id(), data, name);
  available_quantities_.push_back(std::move(name));
}

DataVector EosTable::read_quantity(const std::string& name) const {
  return h5::read_data<1, DataVector>(eos_table_group_.id(), name);
}
}  // namespace h5
