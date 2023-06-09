// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace Frame {
struct Distorted;
struct Grid;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain::creators::time_dependence {
/// \brief Make the mesh time independent so that it isn't moving.
///
/// \warning Calling the `block_maps` and `functions_of_time` functions causes
/// an error because the `None` class should be detected separately and
/// optimizations applied so that the coordinates, Jacobians, etc. are not
/// recomputed.
template <size_t MeshDim>
class None final : public TimeDependence<MeshDim> {
 public:
  using maps_list = tmpl::list<>;
  using options = tmpl::list<>;

  static constexpr Options::String help = {
      "No time dependence in the in grid."};

  None() = default;
  ~None() override = default;
  None(const None&) = default;
  None(None&&) = default;
  None& operator=(const None&) = default;
  None& operator=(None&&) = default;

  auto get_clone() const -> std::unique_ptr<TimeDependence<MeshDim>> override;

  [[noreturn]] auto block_maps_grid_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, MeshDim>>> override;

  [[noreturn]] auto block_maps_grid_to_distorted(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Distorted, MeshDim>>> override;

  [[noreturn]] auto block_maps_distorted_to_inertial(size_t number_of_blocks)
      const -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Distorted, Frame::Inertial, MeshDim>>> override;

  [[noreturn]] auto functions_of_time(
      const std::unordered_map<std::string, double>& initial_expiration_times =
          {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;
};

template <size_t Dim>
bool operator==(const None<Dim>& lhs, const None<Dim>& rhs);

template <size_t Dim>
bool operator!=(const None<Dim>& lhs, const None<Dim>& rhs);
}  // namespace domain::creators::time_dependence
