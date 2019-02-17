// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/DiscreteRotation.hpp"

#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Side.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {
namespace CoordinateMaps {

template <size_t VolumeDim>
DiscreteRotation<VolumeDim>::DiscreteRotation(
    OrientationMap<VolumeDim> orientation) noexcept
    : orientation_(std::move(orientation)),
      is_identity_(orientation_ == OrientationMap<VolumeDim>{}) {}

template <size_t VolumeDim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, VolumeDim> DiscreteRotation<VolumeDim>::
operator()(const std::array<T, VolumeDim>& source_coords) const noexcept {
  return discrete_rotation(orientation_, source_coords);
}

template <size_t VolumeDim>
boost::optional<std::array<double, VolumeDim>>
DiscreteRotation<VolumeDim>::inverse(
    const std::array<double, VolumeDim>& target_coords) const noexcept {
  return discrete_rotation(orientation_.inverse_map(), target_coords);
}

template <size_t VolumeDim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, VolumeDim, Frame::NoFrame>
DiscreteRotation<VolumeDim>::jacobian(
    const std::array<T, VolumeDim>& source_coords) const noexcept {
  auto jacobian_matrix = make_with_value<
      tnsr::Ij<tt::remove_cvref_wrap_t<T>, VolumeDim, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), 0.0);
  for (size_t d = 0; d < VolumeDim; d++) {
    const auto new_direction =
        orientation_(Direction<VolumeDim>(d, Side::Upper));
    jacobian_matrix.get(d, orientation_(d)) =
        new_direction.side() == Side::Upper ? 1.0 : -1.0;
  }
  return jacobian_matrix;
}

template <size_t VolumeDim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, VolumeDim, Frame::NoFrame>
DiscreteRotation<VolumeDim>::inv_jacobian(
    const std::array<T, VolumeDim>& source_coords) const noexcept {
  auto inv_jacobian_matrix = make_with_value<
      tnsr::Ij<tt::remove_cvref_wrap_t<T>, VolumeDim, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), 0.0);
  for (size_t d = 0; d < VolumeDim; d++) {
    const auto new_direction =
        orientation_(Direction<VolumeDim>(d, Side::Upper));
    inv_jacobian_matrix.get(orientation_(d), d) =
        new_direction.side() == Side::Upper ? 1.0 : -1.0;
  }
  return inv_jacobian_matrix;
}

template <size_t VolumeDim>
void DiscreteRotation<VolumeDim>::pup(PUP::er& p) noexcept {
  p | orientation_;
  p | is_identity_;
}

template class DiscreteRotation<1>;
template class DiscreteRotation<2>;
template class DiscreteRotation<3>;

// Explicit instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)>         \
  DiscreteRotation<DIM(data)>::operator()(                                     \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),           \
                    Frame::NoFrame>                                            \
  DiscreteRotation<DIM(data)>::jacobian(                                       \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const noexcept; \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),           \
                    Frame::NoFrame>                                            \
  DiscreteRotation<DIM(data)>::inv_jacobian(                                   \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
}  // namespace domain
