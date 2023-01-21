// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/RdmpTci.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace evolution::dg::subcell {
int rdmp_tci(const DataVector& max_of_current_variables,
             const DataVector& min_of_current_variables,
             const DataVector& max_of_past_variables,
             const DataVector& min_of_past_variables, const double rdmp_delta0,
             const double rdmp_epsilon) {
  const size_t number_of_vars = max_of_current_variables.size();
  ASSERT(min_of_current_variables.size() == number_of_vars and
             max_of_past_variables.size() == number_of_vars and
             min_of_past_variables.size() == number_of_vars,
         "The max and min of the current and past variables must all have the "
         "same size.");
  for (size_t i = 0; i < number_of_vars; ++i) {
    using std::max;
    const double delta = max(
        rdmp_delta0,
        rdmp_epsilon * (max_of_past_variables[i] - min_of_past_variables[i]));

    if (max_of_current_variables[i] > max_of_past_variables[i] + delta or
        min_of_current_variables[i] < min_of_past_variables[i] - delta) {
      return static_cast<int>(i + 1);
    }
  }
  return 0;
}
}  // namespace evolution::dg::subcell
