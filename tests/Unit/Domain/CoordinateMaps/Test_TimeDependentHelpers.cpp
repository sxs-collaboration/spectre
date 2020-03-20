// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/TimeDependentHelpers.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
class DataVector;

namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain
/// \endcond

namespace {
template <size_t Dim>
struct TimeDepMap {
  static constexpr size_t dim = Dim;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> operator()(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;
};
template <size_t Dim>
struct TimeIndepMap {
  static constexpr size_t dim = Dim;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> operator()(
      const std::array<T, Dim>& source_coords) const noexcept;
};

template <size_t Dim>
struct TimeDepJac {
  static constexpr size_t dim = Dim;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> jacobian(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;
};
template <size_t Dim>
struct TimeIndepJac {
  static constexpr size_t dim = Dim;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> jacobian(
      const std::array<T, Dim>& source_coords) const noexcept;
};
}  // namespace

namespace domain {
static_assert(is_map_time_dependent_t<TimeDepMap<1>>::value,
              "Failed testing is_map_time_dependent_t");
static_assert(is_map_time_dependent_t<TimeDepMap<2>>::value,
              "Failed testing is_map_time_dependent_t");
static_assert(is_map_time_dependent_t<TimeDepMap<3>>::value,
              "Failed testing is_map_time_dependent_t");
static_assert(not is_map_time_dependent_t<TimeIndepMap<1>>::value,
              "Failed testing is_map_time_dependent_t");
static_assert(not is_map_time_dependent_t<TimeIndepMap<2>>::value,
              "Failed testing is_map_time_dependent_t");
static_assert(not is_map_time_dependent_t<TimeIndepMap<3>>::value,
              "Failed testing is_map_time_dependent_t");

static_assert(is_map_time_dependent_v<TimeDepMap<1>>,
              "Failed testing is_map_time_dependent_v");
static_assert(is_map_time_dependent_v<TimeDepMap<2>>,
              "Failed testing is_map_time_dependent_v");
static_assert(is_map_time_dependent_v<TimeDepMap<3>>,
              "Failed testing is_map_time_dependent_v");
static_assert(not is_map_time_dependent_v<TimeIndepMap<1>>,
              "Failed testing is_map_time_dependent_v");
static_assert(not is_map_time_dependent_v<TimeIndepMap<2>>,
              "Failed testing is_map_time_dependent_v");
static_assert(not is_map_time_dependent_v<TimeIndepMap<3>>,
              "Failed testing is_map_time_dependent_v");

static_assert(is_jacobian_time_dependent_t<TimeDepJac<1>, DataVector>::value,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(is_jacobian_time_dependent_t<TimeDepJac<1>, double>::value,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(is_jacobian_time_dependent_t<TimeDepJac<2>, DataVector>::value,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(is_jacobian_time_dependent_t<TimeDepJac<2>, double>::value,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(is_jacobian_time_dependent_t<TimeDepJac<3>, DataVector>::value,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(is_jacobian_time_dependent_t<TimeDepJac<3>, double>::value,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(
    not is_jacobian_time_dependent_t<TimeIndepJac<1>, DataVector>::value,
    "Failed testing is_jacobian_time_dependent_t");
static_assert(not is_jacobian_time_dependent_t<TimeIndepJac<1>, double>::value,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(
    not is_jacobian_time_dependent_t<TimeIndepJac<2>, DataVector>::value,
    "Failed testing is_jacobian_time_dependent_t");
static_assert(not is_jacobian_time_dependent_t<TimeIndepJac<2>, double>::value,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(
    not is_jacobian_time_dependent_t<TimeIndepJac<3>, DataVector>::value,
    "Failed testing is_jacobian_time_dependent_t");
static_assert(not is_jacobian_time_dependent_t<TimeIndepJac<3>, double>::value,
              "Failed testing is_jacobian_time_dependent_t");

static_assert(is_jacobian_time_dependent_v<TimeDepJac<1>, DataVector>,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(is_jacobian_time_dependent_v<TimeDepJac<1>, double>,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(is_jacobian_time_dependent_v<TimeDepJac<2>, DataVector>,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(is_jacobian_time_dependent_v<TimeDepJac<2>, double>,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(is_jacobian_time_dependent_v<TimeDepJac<3>, DataVector>,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(is_jacobian_time_dependent_v<TimeDepJac<3>, double>,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(not is_jacobian_time_dependent_v<TimeIndepJac<1>, DataVector>,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(not is_jacobian_time_dependent_v<TimeIndepJac<1>, double>,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(not is_jacobian_time_dependent_v<TimeIndepJac<2>, DataVector>,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(not is_jacobian_time_dependent_v<TimeIndepJac<2>, double>,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(not is_jacobian_time_dependent_v<TimeIndepJac<3>, DataVector>,
              "Failed testing is_jacobian_time_dependent_t");
static_assert(not is_jacobian_time_dependent_v<TimeIndepJac<3>, double>,
              "Failed testing is_jacobian_time_dependent_t");
}  // namespace domain
