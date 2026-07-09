// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Actions/SetInitialData.hpp"

#include <boost/functional/hash.hpp>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "Utilities/PrettyType.hpp"

namespace CurvedScalarWave {

NumericInitialData::NumericInitialData(
    importers::ImporterOptions importer_options, ScalarVars selected_variables)
    : importer_options_(std::move(importer_options)),
      selected_variables_(std::move(selected_variables)) {}

NumericInitialData::NumericInitialData(CkMigrateMessage* msg)
    : InitialData(msg) {}

const importers::ImporterOptions& NumericInitialData::importer_options() const {
  return importer_options_;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
}

bool operator==(const NumericInitialData& lhs, const NumericInitialData& rhs) {
  return lhs.importer_options_ == rhs.importer_options_ and
         lhs.selected_variables_ == rhs.selected_variables_;
}

}  // namespace CurvedScalarWave
