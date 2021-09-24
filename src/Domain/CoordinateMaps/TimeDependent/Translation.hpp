// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain {
namespace CoordinateMaps {
namespace TimeDependent {
/*!
 * \ingroup CoordMapsTimeDependentGroup
 * \brief Translation map defined by \f$\vec{x} = \vec{\xi}+\vec{T}(t)\f$.
 *
 * The map adds a translation, \f$\vec{T}(t)\f$, to the coordinates
 * \f$\vec{\xi}\f$, where \f$\vec{T}(t)\f$ is a FunctionOfTime.
 */
template <size_t Dim>
class Translation {
 public:
  static constexpr size_t dim = Dim;

  Translation() = default;
  explicit Translation(std::string function_of_time_name) noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> operator()(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, Dim>> inverse(
      const std::array<double, Dim>& target_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> frame_velocity(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> inv_jacobian(
      const std::array<T, Dim>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> jacobian(
      const std::array<T, Dim>& source_coords) const noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  static bool is_identity() noexcept { return false; }

 private:
  template <size_t LocalDim>
  friend bool operator==(  // NOLINT(readability-redundant-declaration)
      const Translation<LocalDim>& lhs,
      const Translation<LocalDim>& rhs) noexcept;

  std::string f_of_t_name_{};
};

template <size_t Dim>
inline bool operator!=(const Translation<Dim>& lhs,
                       const Translation<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace TimeDependent
}  // namespace CoordinateMaps
}  // namespace domain
