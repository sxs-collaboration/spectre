// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/TensorYlm/CartToSphere.hpp"
#include "NumericalAlgorithms/TensorYlm/TensorYlm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/RuntimeCache.hpp"

namespace gr::surfaces {
namespace {

// Check modal storage before the formulas index a complete Goldberg mode set.
void assert_size(const ComplexModalVector& data, const size_t expected_size,
                 const char* const name) {
  ASSERT(data.size() == expected_size,
         "Expected " << name << " to have size " << expected_size << " but got "
                     << data.size());
}

// Check every tensor component because their views can have independent sizes.
template <typename TensorType>
void assert_tensor_size(const TensorType& tensor, const size_t expected_size,
                        const char* const name) {
  for (size_t storage_index = 0; storage_index < tensor.size();
       ++storage_index) {
    ASSERT(tensor[storage_index].size() == expected_size,
           "Expected component "
               << storage_index << " of " << name << " to have size "
               << expected_size << " but got " << tensor[storage_index].size());
  }
}

// Read a mode while keeping the Goldberg-ordering detail out of the formulas.
std::complex<double> mode(const ComplexModalVector& data, const size_t l_max,
                          const size_t l, const int m) {
  return data[Spectral::Swsh::goldberg_mode_index(l_max, l, m)];
}

// Write a mode while keeping the Goldberg-ordering detail in one place.
void set_mode(const gsl::not_null<ComplexModalVector*> data, const size_t l_max,
              const size_t l, const int m, const std::complex<double>& value) {
  (*data)[Spectral::Swsh::goldberg_mode_index(l_max, l, m)] = value;
}

// Complete modes computed only for m >= 0 using scalar-field reality.
void fill_scalar_negative_m_modes(const gsl::not_null<ComplexModalVector*> data,
                                  const size_t l_max) {
  for (size_t l = 0; l <= l_max; ++l) {
    for (int m = 1; m <= static_cast<int>(l); ++m) {
      const double sign = m % 2 == 0 ? 1.0 : -1.0;
      set_mode(data, l_max, l, -m, sign * conj(mode(*data, l_max, l, m)));
    }
  }
}

// Convert Spherepack's real a/b coefficients to orthonormal complex Ylm modes;
// the phase and normalization account for the two libraries' conventions.
std::complex<double> standard_mode_from_spherepack(const DataVector& data,
                                                   const size_t l_max,
                                                   const size_t l,
                                                   const size_t m) {
  ylm::SpherepackIterator iterator(l_max, l_max, 1, false);
  const size_t a_index =
      iterator.set(l, m, ylm::SpherepackIterator::CoefficientArray::a)();
  const size_t b_index =
      iterator.set(l, m, ylm::SpherepackIterator::CoefficientArray::b)();
  const std::complex<double> spherepack_mode{data[a_index], data[b_index]};
  const double sign = m % 2 == 0 ? 1.0 : -1.0;
  return sign * sqrt(M_PI / 2.0) * spherepack_mode;
}

// Bundle the two tensor-basis transforms needed by RWZ extraction.
struct CartesianToSphericalMatrices {
  SimpleSparseMatrix rank_one{};
  SimpleSparseMatrix rank_two_symmetric{};
};

// Build the standard TensorYlm basis transforms; only their rank-specific
// bundling and caching are local here.
CartesianToSphericalMatrices make_cartesian_to_spherical_matrices(
    const size_t l_max) {
  CartesianToSphericalMatrices result{};
  ylm::TensorYlm::fill_cart_to_sphere<
      typename tnsr::i<DataVector, 3, Frame::Inertial>::structure>(
      make_not_null(&result.rank_one), l_max,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);
  ylm::TensorYlm::fill_cart_to_sphere<
      typename tnsr::ii<DataVector, 3, Frame::Inertial>::structure>(
      make_not_null(&result.rank_two_symmetric), l_max,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);
  return result;
}

// This matches the upper bound of the standard Spherepack cache. Larger
// transforms remain supported, but are uncommon and are built on demand
// without retaining potentially large sparse matrices.
constexpr size_t maximum_cached_transform_l_max = 150;

// Lazily reuse the comparatively expensive sparse transforms at common l_max.
const CartesianToSphericalMatrices& cached_cartesian_to_spherical_matrices(
    const size_t l_max) {
  static const auto cache = make_runtime_cache<
      CacheRange<size_t{2}, maximum_cached_transform_l_max + 1>>(
      [](const size_t cached_l_max) {
        return make_cartesian_to_spherical_matrices(cached_l_max);
      });
  return cache(l_max);
}

// Transform nodal Cartesian components first to Spherepack modes and then to
// spherical tensor components. The GH Variables-wide transform has the same
// stages but operates on a fixed tag list, so this tensor-level form
// avoids manufacturing unrelated GH fields. Spherepack padding is zeroed
// because it does not represent modes but is included in the sparse storage.
template <typename TensorType>
TensorType cartesian_to_spherical_tensor_modes(
    const TensorType& nodal_tensor, const ylm::Spherepack& spherepack,
    const SimpleSparseMatrix& cart_to_sphere_matrix) {
  const size_t spectral_size = spherepack.spectral_size();
  TensorType cartesian_modes{spectral_size, 0.0};
  TensorType spherical_modes{spectral_size, 0.0};

  for (size_t storage_index = 0; storage_index < nodal_tensor.size();
       ++storage_index) {
    cartesian_modes[storage_index] =
        spherepack.phys_to_spec(nodal_tensor[storage_index]);
  }

  const ylm::SpherepackIterator iterator(spherepack.l_max(), spherepack.m_max(),
                                         1, false);
  for (size_t storage_index = 0; storage_index < cartesian_modes.size();
       ++storage_index) {
    for (size_t offset = 0; offset < spectral_size; ++offset) {
      if (not iterator.compact_index(offset).has_value()) {
        cartesian_modes[storage_index][offset] = 0.0;
      }
    }
  }

  std::vector<double> flattened_cartesian_modes(cartesian_modes.size() *
                                                spectral_size);
  std::vector<double> flattened_spherical_modes(
      spherical_modes.size() * spectral_size, 0.0);
  for (size_t storage_index = 0; storage_index < cartesian_modes.size();
       ++storage_index) {
    for (size_t mode_index = 0; mode_index < spectral_size; ++mode_index) {
      flattened_cartesian_modes[storage_index * spectral_size + mode_index] =
          cartesian_modes[storage_index][mode_index];
    }
  }

  const gsl::span<double> cartesian_modes_span{flattened_cartesian_modes};
  gsl::span<double> spherical_modes_span{flattened_spherical_modes};
  cart_to_sphere_matrix.increment_multiply_on_right(
      make_not_null(&spherical_modes_span), 0, 1, cartesian_modes_span, 0, 1);

  for (size_t storage_index = 0; storage_index < spherical_modes.size();
       ++storage_index) {
    for (size_t mode_index = 0; mode_index < spectral_size; ++mode_index) {
      spherical_modes[storage_index][mode_index] =
          flattened_spherical_modes[storage_index * spectral_size + mode_index];
    }
  }
  return spherical_modes;
}

// Enforce the coordinate-sphere assumption used for radial projections and
// for interpreting the supplied Spherepack grid as extraction angles.
void assert_points_are_on_coordinate_sphere(
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const std::array<double, 3>& center, const double extraction_radius) {
  auto centered_coords = inertial_coords;
  for (size_t i = 0; i < 3; ++i) {
    centered_coords.get(i) -= gsl::at(center, i);
  }
  const auto coordinate_radii = magnitude(centered_coords);
  for (size_t s = 0; s < get(coordinate_radii).size(); ++s) {
    ASSERT(equal_within_roundoff(get(coordinate_radii)[s], extraction_radius),
           "Expected point "
               << s << " to lie at coordinate radius " << extraction_radius
               << " about the extraction center, but its radius is "
               << get(coordinate_radii)[s]);
  }
}

}  // namespace

ReggeWheelerZerilli::ReggeWheelerZerilli(const size_t l_max)
    : phi_plus(square(l_max + 1), 0.0),
      phi_minus(square(l_max + 1), 0.0),
      r_times_strain(square(l_max + 1), 0.0) {}

void regge_wheeler_zerilli_moncrief(
    const gsl::not_null<ReggeWheelerZerilli*> rwz_quantities,
    const ComplexModalVector& h_t, const ComplexModalVector& dr_h_t,
    const ComplexModalVector& dt_h_r, const ComplexModalVector& h_rr,
    const ComplexModalVector& q_r, const ComplexModalVector& k,
    const ComplexModalVector& dr_k, const ComplexModalVector& g,
    const ComplexModalVector& dr_g, const size_t l_max,
    const double extraction_radius) {
  ASSERT(
      extraction_radius > 0.0,
      "The extraction radius must be positive, but is " << extraction_radius);
  const size_t number_of_modes = square(l_max + 1);
  assert_size(h_t, number_of_modes, "h_t");
  assert_size(dr_h_t, number_of_modes, "dr_h_t");
  assert_size(dt_h_r, number_of_modes, "dt_h_r");
  assert_size(h_rr, number_of_modes, "h_rr");
  assert_size(q_r, number_of_modes, "q_r");
  assert_size(k, number_of_modes, "k");
  assert_size(dr_k, number_of_modes, "dr_k");
  assert_size(g, number_of_modes, "g");
  assert_size(dr_g, number_of_modes, "dr_g");

  rwz_quantities->phi_plus = ComplexModalVector{number_of_modes, 0.0};
  rwz_quantities->phi_minus = ComplexModalVector{number_of_modes, 0.0};
  rwz_quantities->r_times_strain =
      SpinWeighted<ComplexModalVector, -2>{number_of_modes, 0.0};

  const std::complex<double> imaginary_unit{0.0, 1.0};
  for (size_t l = 2; l <= l_max; ++l) {
    const auto lambda = static_cast<double>((l - 1) * (l + 2));
    const auto l_l_plus_1 = static_cast<double>(l * (l + 1));
    const double strain_prefactor =
        sqrt(static_cast<double>((l - 1) * l * (l + 1) * (l + 2)));
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); ++m) {
      const auto p_r = mode(q_r, l_max, l, m) - 0.5 *
                                                    square(extraction_radius) *
                                                    mode(dr_g, l_max, l, m);
      const auto z_r =
          mode(h_rr, l_max, l, m) -
          extraction_radius * mode(dr_k, l_max, l, m) -
          0.5 * extraction_radius * l_l_plus_1 * mode(dr_g, l_max, l, m) -
          2.0 * p_r / extraction_radius;
      const auto k_invariant = mode(k, l_max, l, m) +
                               0.5 * l_l_plus_1 * mode(g, l_max, l, m) -
                               2.0 * p_r / extraction_radius;
      const auto phi_minus = (extraction_radius * (mode(dt_h_r, l_max, l, m) -
                                                   mode(dr_h_t, l_max, l, m)) +
                              2.0 * mode(h_t, l_max, l, m)) /
                             lambda;
      const auto phi_plus = extraction_radius *
                            (2.0 * z_r + lambda * k_invariant) /
                            (lambda * l_l_plus_1);
      const size_t index = Spectral::Swsh::goldberg_mode_index(l_max, l, m);
      rwz_quantities->phi_plus[index] = phi_plus;
      rwz_quantities->phi_minus[index] = phi_minus;
      rwz_quantities->r_times_strain.data()[index] =
          strain_prefactor * (phi_plus + imaginary_unit * phi_minus);
    }
  }
}

ReggeWheelerZerilli regge_wheeler_zerilli_moncrief(
    const ComplexModalVector& h_t, const ComplexModalVector& dr_h_t,
    const ComplexModalVector& dt_h_r, const ComplexModalVector& h_rr,
    const ComplexModalVector& q_r, const ComplexModalVector& k,
    const ComplexModalVector& dr_k, const ComplexModalVector& g,
    const ComplexModalVector& dr_g, const size_t l_max,
    const double extraction_radius) {
  ReggeWheelerZerilli rwz_quantities{l_max};
  regge_wheeler_zerilli_moncrief(make_not_null(&rwz_quantities), h_t, dr_h_t,
                                 dt_h_r, h_rr, q_r, k, dr_k, g, dr_g, l_max,
                                 extraction_radius);
  return rwz_quantities;
}

void regge_wheeler_zerilli_moncrief_from_gh_vars(
    const gsl::not_null<ReggeWheelerZerilli*> rwz_quantities,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ylm::Spherepack& ylm_spherepack, const size_t extraction_l_max,
    const std::array<double, 3>& center, const double extraction_radius) {
  ASSERT(
      extraction_radius > 0.0,
      "The extraction radius must be positive, but is " << extraction_radius);
  const size_t expected_number_of_points = ylm_spherepack.physical_size();
  assert_tensor_size(inertial_coords, expected_number_of_points,
                     "inertial_coords");
  assert_tensor_size(spacetime_metric, expected_number_of_points,
                     "spacetime_metric");
  assert_tensor_size(pi, expected_number_of_points, "pi");
  assert_tensor_size(phi, expected_number_of_points, "phi");
  ASSERT(ylm_spherepack.m_max() == ylm_spherepack.l_max(),
         "Regge-Wheeler-Zerilli extraction requires m_max == l_max, but got "
             << "l_max = " << ylm_spherepack.l_max()
             << " and m_max = " << ylm_spherepack.m_max());
  ASSERT(extraction_l_max <= ylm_spherepack.l_max() and
             ylm_spherepack.l_max() - extraction_l_max >= 2,
         "Regge-Wheeler-Zerilli extraction of modes through l = "
             << extraction_l_max
             << " requires a TensorYlm grid with l_max >= " << extraction_l_max
             << " + 2, but got l_max = " << ylm_spherepack.l_max());
  assert_points_are_on_coordinate_sphere(inertial_coords, center,
                                         extraction_radius);

  const size_t transform_l_max = ylm_spherepack.l_max();
  const size_t number_of_points = get<0>(inertial_coords).size();
  const size_t number_of_modes = square(extraction_l_max + 1);

  tnsr::i<DataVector, 3, Frame::Inertial> radial_unit_vector{number_of_points,
                                                             0.0};
  tnsr::i<DataVector, 3, Frame::Inertial> metric_time_space_perturbation{
      number_of_points, 0.0};
  tnsr::i<DataVector, 3, Frame::Inertial> dr_metric_time_space_perturbation{
      number_of_points, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric_perturbation{
      number_of_points, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> dt_spatial_metric{number_of_points,
                                                             0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> dr_spatial_metric{number_of_points,
                                                             0.0};

  for (size_t i = 0; i < 3; ++i) {
    radial_unit_vector.get(i) =
        (inertial_coords.get(i) - gsl::at(center, i)) / extraction_radius;
  }

  for (size_t i = 0; i < 3; ++i) {
    metric_time_space_perturbation.get(i) = spacetime_metric.get(i + 1, 0);
    dr_metric_time_space_perturbation.get(i) = 0.0;
    for (size_t k = 0; k < 3; ++k) {
      dr_metric_time_space_perturbation.get(i) +=
          radial_unit_vector.get(k) * phi.get(k, i + 1, 0);
    }
    for (size_t j = i; j < 3; ++j) {
      spatial_metric_perturbation.get(i, j) =
          spacetime_metric.get(i + 1, j + 1);
      if (i == j) {
        spatial_metric_perturbation.get(i, j) -= 1.0;
      }
      dt_spatial_metric.get(i, j) = -pi.get(i + 1, j + 1);
      dr_spatial_metric.get(i, j) = 0.0;
      for (size_t k = 0; k < 3; ++k) {
        dr_spatial_metric.get(i, j) +=
            radial_unit_vector.get(k) * phi.get(k, i + 1, j + 1);
      }
    }
  }

  std::optional<CartesianToSphericalMatrices> uncached_matrices{};
  if (transform_l_max > maximum_cached_transform_l_max) {
    uncached_matrices.emplace(
        make_cartesian_to_spherical_matrices(transform_l_max));
  }
  const auto& cart_to_sphere_matrices =
      uncached_matrices.has_value()
          ? *uncached_matrices
          : cached_cartesian_to_spherical_matrices(transform_l_max);

  const auto metric_time_space_modes = cartesian_to_spherical_tensor_modes(
      metric_time_space_perturbation, ylm_spherepack,
      cart_to_sphere_matrices.rank_one);
  const auto dr_metric_time_space_modes = cartesian_to_spherical_tensor_modes(
      dr_metric_time_space_perturbation, ylm_spherepack,
      cart_to_sphere_matrices.rank_one);
  const auto metric_modes = cartesian_to_spherical_tensor_modes(
      spatial_metric_perturbation, ylm_spherepack,
      cart_to_sphere_matrices.rank_two_symmetric);
  const auto dt_metric_modes = cartesian_to_spherical_tensor_modes(
      dt_spatial_metric, ylm_spherepack,
      cart_to_sphere_matrices.rank_two_symmetric);
  const auto dr_metric_modes = cartesian_to_spherical_tensor_modes(
      dr_spatial_metric, ylm_spherepack,
      cart_to_sphere_matrices.rank_two_symmetric);

  ComplexModalVector h_t{number_of_modes, 0.0};
  ComplexModalVector dr_h_t{number_of_modes, 0.0};
  ComplexModalVector dt_h_r{number_of_modes, 0.0};
  ComplexModalVector h_rr{number_of_modes, 0.0};
  ComplexModalVector q_r{number_of_modes, 0.0};
  ComplexModalVector k{number_of_modes, 0.0};
  ComplexModalVector dr_k{number_of_modes, 0.0};
  ComplexModalVector g{number_of_modes, 0.0};
  ComplexModalVector dr_g{number_of_modes, 0.0};

  const std::complex<double> imaginary_unit{0.0, 1.0};
  for (size_t l = 2; l <= extraction_l_max; ++l) {
    const double vector_prefactor =
        1.0 / sqrt(2.0 * static_cast<double>(l * (l + 1)));
    const double tensor_prefactor =
        1.0 / sqrt(static_cast<double>((l - 1) * l * (l + 1) * (l + 2)));
    for (size_t m = 0; m <= l; ++m) {
      const auto v_m = standard_mode_from_spherepack(
          get<1>(metric_time_space_modes), transform_l_max, l, m);
      const auto v_mbar = standard_mode_from_spherepack(
          get<2>(metric_time_space_modes), transform_l_max, l, m);
      const auto dr_v_m = standard_mode_from_spherepack(
          get<1>(dr_metric_time_space_modes), transform_l_max, l, m);
      const auto dr_v_mbar = standard_mode_from_spherepack(
          get<2>(dr_metric_time_space_modes), transform_l_max, l, m);
      const auto t_ll = standard_mode_from_spherepack(get<0, 0>(metric_modes),
                                                      transform_l_max, l, m);
      const auto t_lm = standard_mode_from_spherepack(get<0, 1>(metric_modes),
                                                      transform_l_max, l, m);
      const auto t_lmbar = standard_mode_from_spherepack(
          get<0, 2>(metric_modes), transform_l_max, l, m);
      const auto t_mm = standard_mode_from_spherepack(get<1, 1>(metric_modes),
                                                      transform_l_max, l, m);
      const auto t_mmbar = standard_mode_from_spherepack(
          get<1, 2>(metric_modes), transform_l_max, l, m);
      const auto t_mbarmbar = standard_mode_from_spherepack(
          get<2, 2>(metric_modes), transform_l_max, l, m);
      const auto dt_t_lm = standard_mode_from_spherepack(
          get<0, 1>(dt_metric_modes), transform_l_max, l, m);
      const auto dt_t_lmbar = standard_mode_from_spherepack(
          get<0, 2>(dt_metric_modes), transform_l_max, l, m);
      const auto dr_t_mm = standard_mode_from_spherepack(
          get<1, 1>(dr_metric_modes), transform_l_max, l, m);
      const auto dr_t_mmbar = standard_mode_from_spherepack(
          get<1, 2>(dr_metric_modes), transform_l_max, l, m);
      const auto dr_t_mbarmbar = standard_mode_from_spherepack(
          get<2, 2>(dr_metric_modes), transform_l_max, l, m);
      const size_t index = Spectral::Swsh::goldberg_mode_index(
          extraction_l_max, l, static_cast<int>(m));
      h_t[index] = imaginary_unit * extraction_radius * vector_prefactor *
                   (v_mbar + v_m);
      dr_h_t[index] = imaginary_unit * vector_prefactor *
                      (extraction_radius * (dr_v_mbar + dr_v_m) + v_mbar + v_m);
      dt_h_r[index] = imaginary_unit * extraction_radius * vector_prefactor *
                      (dt_t_lmbar + dt_t_lm);
      h_rr[index] = t_ll;
      q_r[index] = -extraction_radius * vector_prefactor * (t_lmbar - t_lm);
      k[index] = t_mmbar;
      dr_k[index] = dr_t_mmbar;
      g[index] = tensor_prefactor * (t_mbarmbar + t_mm);
      dr_g[index] = tensor_prefactor * (dr_t_mbarmbar + dr_t_mm);
    }
  }

  fill_scalar_negative_m_modes(make_not_null(&h_t), extraction_l_max);
  fill_scalar_negative_m_modes(make_not_null(&dr_h_t), extraction_l_max);
  fill_scalar_negative_m_modes(make_not_null(&dt_h_r), extraction_l_max);
  fill_scalar_negative_m_modes(make_not_null(&h_rr), extraction_l_max);
  fill_scalar_negative_m_modes(make_not_null(&q_r), extraction_l_max);
  fill_scalar_negative_m_modes(make_not_null(&k), extraction_l_max);
  fill_scalar_negative_m_modes(make_not_null(&dr_k), extraction_l_max);
  fill_scalar_negative_m_modes(make_not_null(&g), extraction_l_max);
  fill_scalar_negative_m_modes(make_not_null(&dr_g), extraction_l_max);

  regge_wheeler_zerilli_moncrief(rwz_quantities, h_t, dr_h_t, dt_h_r, h_rr, q_r,
                                 k, dr_k, g, dr_g, extraction_l_max,
                                 extraction_radius);
}

ReggeWheelerZerilli regge_wheeler_zerilli_moncrief_from_gh_vars(
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ylm::Spherepack& ylm_spherepack, const size_t extraction_l_max,
    const std::array<double, 3>& center, const double extraction_radius) {
  ReggeWheelerZerilli rwz_quantities{extraction_l_max};
  regge_wheeler_zerilli_moncrief_from_gh_vars(
      make_not_null(&rwz_quantities), spacetime_metric, pi, phi,
      inertial_coords, ylm_spherepack, extraction_l_max, center,
      extraction_radius);
  return rwz_quantities;
}

}  // namespace gr::surfaces
