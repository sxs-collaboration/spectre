// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Actions/NumericInitialData.hpp"

#include <boost/functional/hash.hpp>
#include <string>
#include <utility>
#include <variant>

#include "Utilities/PrettyType.hpp"

namespace grmhd::ValenciaDivClean {

NumericInitialData::NumericInitialData(
    std::string file_glob, std::string subfile_name,
    std::variant<double, importers::ObservationSelector> observation_value,
    const bool enable_interpolation, PrimitiveVars selected_variables,
    const double density_cutoff)
    : importer_options_(std::move(file_glob), std::move(subfile_name),
                        observation_value, enable_interpolation),
      selected_variables_(std::move(selected_variables)),
      density_cutoff_(density_cutoff) {}

NumericInitialData::NumericInitialData(CkMigrateMessage* msg)
    : InitialData(msg) {}

PUP::able::PUP_ID NumericInitialData::my_PUP_ID = 0;

size_t NumericInitialData::volume_data_id() const {
  size_t hash = 0;
  boost::hash_combine(hash, pretty_type::get_name<NumericInitialData>());
  boost::hash_combine(hash,
                      get<importers::OptionTags::FileGlob>(importer_options_));
  boost::hash_combine(hash,
                      get<importers::OptionTags::Subgroup>(importer_options_));
  return hash;
}

void NumericInitialData::pup(PUP::er& p) {
  p | importer_options_;
  p | selected_variables_;
  p | density_cutoff_;
}

bool operator==(const NumericInitialData& lhs, const NumericInitialData& rhs) {
  return lhs.importer_options_ == rhs.importer_options_ and
         lhs.selected_variables_ == rhs.selected_variables_ and
         lhs.density_cutoff_ == rhs.density_cutoff_;
}

}  // namespace grmhd::ValenciaDivClean
