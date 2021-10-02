// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/KerrHorizonConforming.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace domain::CoordinateMaps {

KerrHorizonConforming::KerrHorizonConforming(std::array<double, 3> spin)
    : spin_(spin), spin_mag_sq_(dot(spin, spin)) {}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> KerrHorizonConforming::operator()(
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  std::array<ReturnType, 3> result{};
  ReturnType& stretch_fac = get<2>(result);
  stretch_factor_square(make_not_null(&stretch_fac), source_coords);
  stretch_fac = sqrt(stretch_fac);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) = gsl::at(source_coords, i) * stretch_fac;
  }
  return result;
}

std::optional<std::array<double, 3>> KerrHorizonConforming::inverse(
    const std::array<double, 3>& target_coords) const {
  const auto coords_mag_sq = dot(target_coords, target_coords);
  const auto coords_sq_min_spin_sq = coords_mag_sq - spin_mag_sq_;
  const auto coords_dot_spin = dot(target_coords, spin_);
  const auto fac = (coords_sq_min_spin_sq +
                    sqrt(coords_sq_min_spin_sq * coords_sq_min_spin_sq +
                         4. * square(coords_dot_spin))) /
                   (2. * coords_mag_sq);
  // this is the way it was written in spec, but I dont think `fac` can
  // ever be smaller than 0
  return fac >= 0.
             ? std::optional<std::array<double, 3>>(target_coords * sqrt(fac))
             : std::nullopt;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
KerrHorizonConforming::jacobian(const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  tnsr::Ij<ReturnType, 3, Frame::NoFrame> jac(
      get_size(dereference_wrapper(source_coords[0])));

  // use allocations from `jac` for auxiliaries
  ReturnType& fac = get<0, 0>(jac);
  ReturnType& source_coords_sq = get<0, 1>(jac);
  ReturnType& coords_dot_spin = get<0, 2>(jac);
  ReturnType& subexpr_1 = get<1, 0>(jac);
  ReturnType& subexpr_2 = get<1, 1>(jac);

  std::array<ReturnType, 3> dfac_dx{};
  if constexpr (std::is_same_v<ReturnType, DataVector>) {
    dfac_dx[0].set_data_ref(&get<2, 0>(jac));
    dfac_dx[1].set_data_ref(&get<2, 1>(jac));
    dfac_dx[2].set_data_ref(&get<2, 2>(jac));
  }

  stretch_factor_square(make_not_null(&fac), source_coords);
  source_coords_sq = dot(source_coords, source_coords);
  coords_dot_spin = dot(source_coords, spin_);
  subexpr_1 = 4. * source_coords_sq * (1. - fac);
  subexpr_2 = 2. * fac * coords_dot_spin;

  for (size_t i = 0; i < 3; ++i) {
    gsl::at(dfac_dx, i) = subexpr_1 * gsl::at(source_coords, i) +
                          2. * spin_mag_sq_ * gsl::at(source_coords, i) -
                          subexpr_2 * gsl::at(spin_, i);
  }
  dfac_dx = dfac_dx / (square(source_coords_sq) + square(coords_dot_spin));

  const ReturnType sqrt_fac = sqrt(fac);

  // not mathematically a part of `dfac_dx` but can be absorbed to avoid
  // allocation for temporary
  dfac_dx = dfac_dx / (2. * sqrt_fac);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jac.get(i, j) = gsl::at(dfac_dx, j) * gsl::at(source_coords, i);
    }
    jac.get(i, i) += sqrt_fac;
  }
  return jac;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
KerrHorizonConforming::inv_jacobian(
    const std::array<T, 3>& source_coords) const {
  using ReturnType = tt::remove_cvref_wrap_t<T>;

  tnsr::Ij<ReturnType, 3, Frame::NoFrame> inv_jac(
      get_size(dereference_wrapper(source_coords[0])));

  ReturnType& mapped_mag_sq = get<0, 0>(inv_jac);
  ReturnType& mapped_mag = get<0, 1>(inv_jac);
  ReturnType& mapped_dot_spin = get<0, 2>(inv_jac);
  ReturnType& mapped_sq_min_spin_sq = get<1, 0>(inv_jac);
  ReturnType& r = get<1, 1>(inv_jac);
  ReturnType& fac = get<1, 2>(inv_jac);
  std::array<ReturnType, 3> dr_dx{};
  if constexpr (std::is_same_v<ReturnType, DataVector>) {
    dr_dx[0].set_data_ref(&get<2, 0>(inv_jac));
    dr_dx[1].set_data_ref(&get<2, 1>(inv_jac));
    dr_dx[2].set_data_ref(&get<2, 2>(inv_jac));
  }

  auto mapped = operator()(source_coords);

  mapped_mag_sq = dot(mapped, mapped);
  mapped_mag = sqrt(mapped_mag_sq);
  mapped_dot_spin = dot(mapped, spin_);
  mapped_sq_min_spin_sq = mapped_mag_sq - spin_mag_sq_;
  r = sqrt(0.5 * (mapped_sq_min_spin_sq + sqrt(square(mapped_sq_min_spin_sq) +
                                               4 * square(mapped_dot_spin))));
  fac = 1. / (2. * cube(r) - mapped_sq_min_spin_sq * r);

  for (size_t i = 0; i < 3; ++i) {
    gsl::at(dr_dx, i) =
        (square(r) * mapped.at(i) + mapped_dot_spin * gsl::at(spin_, i)) * fac;
  }

  // normalized from this point
  mapped = mapped / mapped_mag;
  const ReturnType r_by_mapped = r / mapped_mag;

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      inv_jac.get(i, j) = gsl::at(dr_dx, j) * gsl::at(mapped, i) -
                          r_by_mapped * gsl::at(mapped, i) * gsl::at(mapped, j);
    }
    inv_jac.get(i, i) += r_by_mapped;
  }
  return inv_jac;
}

template <typename T>
void KerrHorizonConforming::stretch_factor_square(
    const gsl::not_null<tt::remove_cvref_wrap_t<T>*> result,
    const std::array<T, 3>& source_coords) const {
  auto& source_coords_sq = *result;
  source_coords_sq = dot(source_coords, source_coords);
  *result =
      source_coords_sq * (source_coords_sq + spin_mag_sq_) /
      (source_coords_sq * source_coords_sq + square(dot(source_coords, spin_)));
}

void KerrHorizonConforming::pup(PUP::er& p) {
  p | spin_;
  p | spin_mag_sq_;
}

bool operator==(const KerrHorizonConforming& lhs,
                const KerrHorizonConforming& rhs) {
  return lhs.spin_ == rhs.spin_;
}

bool operator!=(const KerrHorizonConforming& lhs,
                const KerrHorizonConforming& rhs) {
  return not(lhs == rhs);
}
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  KerrHorizonConforming::operator()(                                         \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  KerrHorizonConforming::jacobian(                                           \
      const std::array<DTYPE(data), 3>& source_coords) const;                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  KerrHorizonConforming::inv_jacobian(                                       \
      const std::array<DTYPE(data), 3>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE
}  // namespace domain::CoordinateMaps
