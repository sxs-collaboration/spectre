// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Events/ObserveAmrStats.hpp"

#include <cstddef>
#include <pup.h>
#include <sstream>

#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace amr::Events {

namespace detail {
std::string FormatAmrStatsOutput::operator()(
    const double time, const size_t total_num_elements,
    const size_t total_num_points,
    const std::vector<size_t>& num_points_per_dim,
    const std::vector<size_t>& min_points_per_dim,
    const std::vector<size_t>& max_points_per_dim) const {
  std::ostringstream oss;
  oss << "AMR stats at time: " << time << "\n"
      << "  Total Elements: " << total_num_elements << "\n"
      << "  Total number of grid points: " << total_num_points << "\n"
      << "  Points per Dimension:\n";
  for (size_t d = 0; d < num_points_per_dim.size(); ++d) {
    oss << "    Dimension " << d << ": " << num_points_per_dim[d] << " (min "
        << min_points_per_dim[d] << ", max " << max_points_per_dim[d]
        << " per element)\n";
  }
  return oss.str();
}
void FormatAmrStatsOutput::pup(PUP::er& /*p*/) {}
}  // namespace detail

template <size_t Dim>
ObserveAmrStats<Dim>::ObserveAmrStats() = default;

template <size_t Dim>
ObserveAmrStats<Dim>::ObserveAmrStats(bool print_to_terminal,
                                      bool observe_per_core)
    : print_to_terminal_(print_to_terminal),
      observe_per_core_(observe_per_core) {}

template <size_t Dim>
void ObserveAmrStats<Dim>::pup(PUP::er& p) {
  Event::pup(p);
  p | print_to_terminal_;
  p | observe_per_core_;
}

template <size_t Dim>
PUP::able::PUP_ID ObserveAmrStats<Dim>::my_PUP_ID = 0;  // NOLINT

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data) template class ObserveAmrStats<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))
#undef DIM
#undef INSTANTIATE
}  // namespace amr::Events
