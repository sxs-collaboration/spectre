// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Events/ObserveGridStats.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/Events/ReductionScope.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Literals.hpp"

namespace Events::ObserveGridStats_detail {
std::vector<std::string> legend(const std::string& observation_value_name,
                                const size_t dim) {
  std::vector<std::string> result{observation_value_name, "NumElements",
                                  "NumPoints", "FractionOfElementsFd",
                                  "FractionOfPointsFd"};
  result.reserve(result.size() + 6 * dim);
  for (const std::string quantity : {"PointsPerDim", "RefinementLevel"}) {
    for (const std::string stat : {"Avg", "Min", "Max"}) {
      for (size_t d = 0; d < dim; ++d) {
        result.push_back(stat + quantity + "_" + std::to_string(d));
      }
    }
  }
  return result;
}

template <size_t Dim>
ReductionData reduction_data(const double observation_value,
                             const Mesh<Dim>& observer_mesh,
                             const ElementId<Dim>& element_id) {
  const size_t num_points = observer_mesh.number_of_grid_points();
  const auto bases = observer_mesh.basis();
  const bool is_fd =
      std::any_of(bases.begin(), bases.end(), [](const Spectral::Basis basis) {
        return basis == Spectral::Basis::FiniteDifference;
      });
  const auto& extents = observer_mesh.extents();
  const std::vector<size_t> points_per_dim(extents.begin(), extents.end());
  const std::array<size_t, Dim> refinement_levels =
      element_id.refinement_levels();
  const std::vector<size_t> refinement_levels_vec(refinement_levels.begin(),
                                                  refinement_levels.end());
  return ReductionData{
      observation_value,
      1_st,
      num_points,
      is_fd ? 1.0 : 0.0,
      is_fd ? static_cast<double>(num_points) : 0.0,
      std::vector<double>(points_per_dim.begin(), points_per_dim.end()),
      points_per_dim,
      points_per_dim,
      std::vector<double>(refinement_levels_vec.begin(),
                          refinement_levels_vec.end()),
      refinement_levels_vec,
      refinement_levels_vec};
}

template <size_t Dim>
std::optional<std::string> subfile_path(
    const ReductionScope reduction_scope,
    const std::optional<std::vector<std::string>>& blocks_to_observe,
    const Domain<Dim>& domain, const ElementId<Dim>& element_id) {
  const std::optional<std::string> block_suffix =
      block_subfile_suffix(reduction_scope, blocks_to_observe,
                           domain.blocks().at(element_id.block_id()).name(),
                           domain.block_names(), domain.block_groups());
  if (not block_suffix.has_value()) {
    return std::nullopt;
  }
  return "/GridStats" + block_suffix.value();
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template ReductionData reduction_data(                                \
      double observation_value, const Mesh<DIM(data)>& observer_mesh,   \
      const ElementId<DIM(data)>& element_id);                          \
  template std::optional<std::string> subfile_path(                     \
      ReductionScope reduction_scope,                                   \
      const std::optional<std::vector<std::string>>& blocks_to_observe, \
      const Domain<DIM(data)>& domain,                                  \
      const ElementId<DIM(data)>& element_id);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace Events::ObserveGridStats_detail
