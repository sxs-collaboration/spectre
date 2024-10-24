// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/Actions/SetInitialData.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <string>
#include <utility>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpatialMetric.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PrettyType.hpp"

namespace ScalarTensor {

NumericInitialData::NumericInitialData(
    std::string file_glob, std::string subfile_name,
    std::variant<double, importers::ObservationSelector> observation_value,
    const bool enable_interpolation,
    typename GhNumericId::Variables::type gh_selected_variables,
    typename ScalarNumericId::Variables::type hydro_selected_variables)
    : gh_numeric_id_(file_glob, subfile_name, observation_value,
                     enable_interpolation, std::move(gh_selected_variables)),
      scalar_numeric_id_(std::move(file_glob), std::move(subfile_name),
                         observation_value, enable_interpolation,
                         std::move(hydro_selected_variables)) {}

NumericInitialData::NumericInitialData(CkMigrateMessage* msg)
    : InitialData(msg) {}

PUP::able::PUP_ID NumericInitialData::my_PUP_ID = 0;

size_t NumericInitialData::volume_data_id() const {
  size_t hash = 0;
  boost::hash_combine(hash, gh_numeric_id_.volume_data_id());
  boost::hash_combine(hash, scalar_numeric_id_.volume_data_id());
  return hash;
}

void NumericInitialData::pup(PUP::er& p) {
  p | gh_numeric_id_;
  p | scalar_numeric_id_;
}

bool operator==(const NumericInitialData& lhs, const NumericInitialData& rhs) {
  return lhs.gh_numeric_id_ == rhs.gh_numeric_id_ and
         lhs.scalar_numeric_id_ == rhs.scalar_numeric_id_;
}

}  // namespace ScalarTensor
