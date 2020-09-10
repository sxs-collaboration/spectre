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
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain {
namespace creators {
namespace time_dependence {
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
  None(None&&) noexcept = default;
  None& operator=(const None&) = default;
  None& operator=(None&&) noexcept = default;

  auto get_clone() const noexcept
      -> std::unique_ptr<TimeDependence<MeshDim>> override;

  [[noreturn]] auto block_maps(size_t number_of_blocks) const noexcept
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, MeshDim>>> override;

  [[noreturn]] auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;
};

template <size_t Dim>
bool operator==(const None<Dim>& lhs, const None<Dim>& rhs) noexcept;

template <size_t Dim>
bool operator!=(const None<Dim>& lhs, const None<Dim>& rhs) noexcept;
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
