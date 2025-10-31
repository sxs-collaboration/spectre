// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Observer/ReductionActions.hpp"

#include <vector>

#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace observers::ThreadedActions::ReductionActions_detail {

void append_to_reduction_data(
    const gsl::not_null<std::vector<double>*> all_reduction_data,
    const double t) {
  all_reduction_data->push_back(t);
}

template <typename T>
void append_to_reduction_data(
    const gsl::not_null<std::vector<double>*> all_reduction_data,
    const std::vector<T>& t) {
  all_reduction_data->insert(all_reduction_data->end(), t.begin(), t.end());
}

void append_to_reduction_data(
    const gsl::not_null<std::vector<double>*> all_reduction_data,
    const std::array<double, 3>& t) {
    all_reduction_data->insert(all_reduction_data->end(), t.begin(), t.end());
}

// Generate instantiations
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                  \
  template void append_to_reduction_data<DTYPE(data)>(        \
      gsl::not_null<std::vector<double>*> all_reduction_data, \
      const std::vector<DTYPE(data)>& t);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, size_t))

#undef DTYPE
#undef INSTANTIATE

}  // namespace observers::ThreadedActions::ReductionActions_detail
