// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/LinearSolver/Actions/BuildMatrix.hpp"

#include <cstddef>
#include <map>
#include <optional>
#include <utility>

#include "Domain/Structure/ElementId.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace LinearSolver::Actions::detail {

template <size_t Dim>
std::pair<size_t, size_t> total_num_points_and_local_first_index(
    const ElementId<Dim>& element_id,
    const std::map<ElementId<Dim>, size_t>& num_points_per_element,
    const size_t num_vars) {
  size_t total_num_points = 0;
  size_t local_first_index = 0;
  for (const auto& [element_id_i, num_points] : num_points_per_element) {
    if (element_id_i < element_id) {
      local_first_index += num_points;
    }
    total_num_points += num_points;
  }
  total_num_points *= num_vars;
  local_first_index *= num_vars;
  return {total_num_points, local_first_index};
}

std::optional<size_t> local_unit_vector_index(const size_t iteration_id,
                                              const size_t local_first_index,
                                              const size_t local_num_points) {
  if (iteration_id < local_first_index or
      iteration_id >= local_first_index + local_num_points) {
    return std::nullopt;
  } else {
    return iteration_id - local_first_index;
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                               \
  template std::pair<size_t, size_t> total_num_points_and_local_first_index( \
      const ElementId<DIM(data)>& element_id,                                \
      const std::map<ElementId<DIM(data)>, size_t>& num_points_per_element,  \
      const size_t num_vars);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

}  // namespace LinearSolver::Actions::detail
