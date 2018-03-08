// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/YlmSpherepack.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <tuple>

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Spherepack.hpp"

//============================================================================
// Note that SPHEREPACK (which is wrapped by YlmSpherepack) takes
// n_theta and n_phi as input, and is ok with arbitrary values of
// n_theta and n_phi.  SPHEREPACK computes the maximum Ylm l and m
// using the formulas l_max=n_theta-1 and m_max=min(l_max,n_phi/2).
//
// However, some combinations of n_theta and n_phi have strange properties:
// - The maximum m that is AT LEAST PARTIALLY represented by (n_theta,n_phi)
//   is std::min(n_theta-1,n_phi/2). This is called m_max here. But an arbitrary
//   (n_theta,n_phi) does not necessarily fully represent all m's up to m_max,
//   because sin(m_max phi) might be zero at all collocation points, and
//   therefore sin(m_max phi) might not be representable on the grid.
// - The largest m that is fully represented by (n_theta,n_phi) is
//   m_max_represented = std::min(n_theta-1,(n_phi-1)/2).
// - Therefore, if n_phi is odd,  m_max = m_max_represented,
//              if n_phi is even, m_max = m_max_represented+1.
// - To remedy this situation, we choose YlmSpherepack to take as arguments
//   l_max and m_max, instead of n_theta and n_phi.
//   We then choose
//      n_theta = l_max+1
//      n_phi   = 2*m_max+1
//   This ensures that m_max = m_max_represented
//   (as opposed to m_max = m_max_represented+1)
//============================================================================

YlmSpherepack::YlmSpherepack(const size_t l_max, const size_t m_max) noexcept
    : l_max_{l_max},
      m_max_{m_max},
      n_theta_{l_max_ + 1},
      n_phi_{2 * m_max_ + 1},
      spectral_size_{2 * (l_max_ + 1) * (m_max_ + 1)} {
  if (l_max_ < 2) {
    ERROR("Must use l_max>=2, not l_max=" << l_max_);
  }
  if (m_max_ < 2) {
    ERROR("Must use m_max>=2, not m_max=" << m_max_);
  }
  fill_scalar_work_arrays();
}

void YlmSpherepack::phys_to_spec_impl(
    const gsl::not_null<double*> spectral_coefs,
    const gsl::not_null<const double*> collocation_values,
    const size_t physical_stride, const size_t physical_offset,
    const size_t spectral_stride, const size_t spectral_offset,
    const bool loop_over_offset) const noexcept {
  size_t work_size = 2 * n_theta_ * n_phi_;
  if (loop_over_offset) {
    ASSERT(physical_stride == spectral_stride, "invalid call");
    work_size *= spectral_stride;
  }
  auto& work = memory_pool_.get(work_size);
  double* const a = spectral_coefs;
  // clang-tidy: 'do not use pointer arithmetic'.
  double* const b =
      a + (m_max_ + 1) * (l_max_ + 1) * spectral_stride;  // NOLINT
  int err = 0;
  const int effective_physical_offset =
      loop_over_offset ? -1 : int(physical_offset);
  const int effective_spectral_offset =
      loop_over_offset ? -1 : int(spectral_offset);
  auto& work_phys_to_spec = storage_.work_phys_to_spec;
  shags_(static_cast<int>(physical_stride), static_cast<int>(spectral_stride),
         effective_physical_offset, effective_spectral_offset,
         static_cast<int>(n_theta_), static_cast<int>(n_phi_), 0, 1,
         collocation_values, static_cast<int>(n_theta_),
         static_cast<int>(n_phi_), a, b, static_cast<int>(m_max_ + 1),
         static_cast<int>(l_max_ + 1), static_cast<int>(m_max_ + 1),
         static_cast<int>(l_max_ + 1), work_phys_to_spec.data(),
         static_cast<int>(work_phys_to_spec.size()), work.data(),
         static_cast<int>(work_size), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("shags error " << err << " in YlmSpherepack");
  }
  memory_pool_.free(work);
}

void YlmSpherepack::spec_to_phys_impl(
    const gsl::not_null<double*> collocation_values,
    const gsl::not_null<const double*> spectral_coefs,
    const size_t spectral_stride, const size_t spectral_offset,
    const size_t physical_stride, const size_t physical_offset,
    const bool loop_over_offset) const noexcept {
  size_t work_size = 2 * n_theta_ * n_phi_;
  if (loop_over_offset) {
    ASSERT(physical_stride == spectral_stride, "invalid call");
    work_size *= spectral_stride;
  }
  auto& work = memory_pool_.get(work_size);
  // 'a' and 'b' are Spherepack's coefficient arrays.
  const double* const a = spectral_coefs;
  // clang-tidy: 'do not use pointer arithmetic'.
  const double* const b =
      a + (m_max_ + 1) * (l_max_ + 1) * spectral_stride;  // NOLINT
  int err = 0;
  const int effective_physical_offset =
      loop_over_offset ? -1 : int(physical_offset);
  const int effective_spectral_offset =
      loop_over_offset ? -1 : int(spectral_offset);

  auto& work_scalar_spec_to_phys = storage_.work_scalar_spec_to_phys;
  shsgs_(static_cast<int>(physical_stride), static_cast<int>(spectral_stride),
         effective_physical_offset, effective_spectral_offset,
         static_cast<int>(n_theta_), static_cast<int>(n_phi_), 0, 1,
         collocation_values, static_cast<int>(n_theta_),
         static_cast<int>(n_phi_), a, b, static_cast<int>(m_max_ + 1),
         static_cast<int>(l_max_ + 1), static_cast<int>(m_max_ + 1),
         static_cast<int>(l_max_ + 1), work_scalar_spec_to_phys.data(),
         static_cast<int>(work_scalar_spec_to_phys.size()), work.data(),
         static_cast<int>(work_size), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("shsgs error " << err << " in YlmSpherepack");
  }
  memory_pool_.free(work);
}

DataVector YlmSpherepack::phys_to_spec(const DataVector& collocation_values,
                                       const size_t physical_stride,
                                       const size_t physical_offset) const
    noexcept {
  ASSERT(collocation_values.size() == physical_size() * physical_stride,
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size() * physical_stride);
  DataVector result(spectral_size());
  phys_to_spec_impl(result.data(), collocation_values.data(), physical_stride,
                    physical_offset, 1, 0, false);
  return result;
}

DataVector YlmSpherepack::spec_to_phys(const DataVector& spectral_coefs,
                                       const size_t spectral_stride,
                                       const size_t spectral_offset) const
    noexcept {
  ASSERT(spectral_coefs.size() == spectral_size() * spectral_stride,
         "Sizes don't match: " << spectral_coefs.size() << " vs "
                               << spectral_size() * spectral_stride);
  DataVector result(physical_size());
  spec_to_phys_impl(result.data(), spectral_coefs.data(), spectral_stride,
                    spectral_offset, 1, 0, false);
  return result;
}

DataVector YlmSpherepack::phys_to_spec_all_offsets(
    const DataVector& collocation_values, const size_t stride) const noexcept {
  ASSERT(collocation_values.size() == physical_size() * stride,
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size() * stride);
  DataVector result(spectral_size() * stride);
  phys_to_spec_impl(result.data(), collocation_values.data(), stride, 0, stride,
                    0, true);
  return result;
}

DataVector YlmSpherepack::spec_to_phys_all_offsets(
    const DataVector& spectral_coefs, const size_t stride) const noexcept {
  ASSERT(spectral_coefs.size() == spectral_size() * stride,
         "Sizes don't match: " << spectral_coefs.size() << " vs "
                               << spectral_size() * stride);
  DataVector result(physical_size() * stride);
  spec_to_phys_impl(result.data(), spectral_coefs.data(), stride, 0, stride, 0,
                    true);
  return result;
}

/// \cond DOXYGEN_FAILS_TO_PARSE_THIS
void YlmSpherepack::gradient(
    const std::array<double*, 2>& df,
    const gsl::not_null<const double*> collocation_values,
    const size_t physical_stride, const size_t physical_offset) const noexcept {
  auto& f_k = memory_pool_.get(spectral_size());
  phys_to_spec_impl(f_k.data(), collocation_values, physical_stride,
                    physical_offset, 1, 0, false);
  gradient_from_coefs_impl(df, f_k.data(), 1, 0, physical_stride,
                           physical_offset, false);
  memory_pool_.free(f_k);
}
/// \endcond

/// \cond DOXYGEN_FAILS_TO_PARSE_THIS
void YlmSpherepack::gradient_all_offsets(
    const std::array<double*, 2>& df,
    const gsl::not_null<const double*> collocation_values,
    const size_t stride) const noexcept {
  const size_t spectral_stride = stride;
  auto& f_k = memory_pool_.get(spectral_stride * spectral_size());
  phys_to_spec_impl(f_k.data(), collocation_values, stride, 0, spectral_stride,
                    0, true);
  gradient_from_coefs_impl(df, f_k.data(), spectral_stride, 0, stride, 0, true);
  memory_pool_.free(f_k);
}
/// \endcond

void YlmSpherepack::gradient_from_coefs_impl(
    const std::array<double*, 2>& df,
    const gsl::not_null<const double*> spectral_coefs,
    const size_t spectral_stride, const size_t spectral_offset,
    const size_t physical_stride, const size_t physical_offset,
    bool loop_over_offset) const noexcept {
  ASSERT((not loop_over_offset) or spectral_stride == physical_stride,
         "physical and spectral strides must be equal "
         "for loop_over_offset=true");

  if (storage_.work_vector_spec_to_phys.empty()) {
    fill_vector_work_arrays();
  }
  const size_t l1 = m_max_ + 1;
  const double* const f_k = spectral_coefs;
  const double* const a = f_k;
  // clang-tidy: 'do not use pointer arithmetic'.
  const double* const b = f_k + l1 * n_theta_ * spectral_stride;  // NOLINT

  size_t work_size = n_theta_ * (3 * n_phi_ + 2 * l1 + 1);
  if (loop_over_offset) {
    work_size *= spectral_stride;
  }
  auto& work = memory_pool_.get(work_size);
  int err = 0;
  const int effective_physical_offset =
      loop_over_offset ? -1 : int(physical_offset);
  const int effective_spectral_offset =
      loop_over_offset ? -1 : int(spectral_offset);
  auto& work_vector_spec_to_phys = storage_.work_vector_spec_to_phys;
  gradgs_(static_cast<int>(physical_stride), static_cast<int>(spectral_stride),
          effective_physical_offset, effective_spectral_offset,
          static_cast<int>(n_theta_), static_cast<int>(n_phi_), 0, 1, df[0],
          df[1], static_cast<int>(n_theta_), static_cast<int>(n_phi_), a, b,
          static_cast<int>(l1), static_cast<int>(n_theta_),
          work_vector_spec_to_phys.data(),
          static_cast<int>(work_vector_spec_to_phys.size()), work.data(),
          static_cast<int>(work_size), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("gradgs error " << err << " in YlmSpherepack");
  }
  memory_pool_.free(work);
}

YlmSpherepack::FirstDeriv YlmSpherepack::gradient(
    const DataVector& collocation_values, const size_t physical_stride,
    const size_t physical_offset) const noexcept {
  ASSERT(collocation_values.size() == physical_size() * physical_stride,
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size() * physical_stride);
  auto& f_k = memory_pool_.get(spectral_size());
  phys_to_spec_impl(f_k.data(), collocation_values.data(), physical_stride,
                    physical_offset, 1, 0, false);
  FirstDeriv result(physical_size());
  std::array<double*, 2> temp = {{result.get(0).data(), result.get(1).data()}};
  gradient_from_coefs_impl(temp, f_k.data(), 1, 0, 1, 0, false);
  memory_pool_.free(f_k);
  return result;
}

YlmSpherepack::FirstDeriv YlmSpherepack::gradient_all_offsets(
    const DataVector& collocation_values, const size_t stride) const noexcept {
  ASSERT(collocation_values.size() == physical_size() * stride,
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size() * stride);
  FirstDeriv result(physical_size() * stride);
  std::array<double*, 2> temp = {{result.get(0).data(), result.get(1).data()}};
  gradient_all_offsets(temp, collocation_values.data(), stride);
  return result;
}

YlmSpherepack::FirstDeriv YlmSpherepack::gradient_from_coefs(
    const DataVector& spectral_coefs, const size_t spectral_stride,
    const size_t spectral_offset) const noexcept {
  ASSERT(spectral_coefs.size() == spectral_size() * spectral_stride,
         "Sizes don't match: " << spectral_coefs.size() << " vs "
                               << spectral_size() * spectral_stride);
  FirstDeriv result(physical_size());
  std::array<double*, 2> temp = {{result.get(0).data(), result.get(1).data()}};
  gradient_from_coefs_impl(temp, spectral_coefs.data(), spectral_stride,
                           spectral_offset, 1, 0, false);
  return result;
}

YlmSpherepack::FirstDeriv YlmSpherepack::gradient_from_coefs_all_offsets(
    const DataVector& spectral_coefs, const size_t stride) const noexcept {
  ASSERT(spectral_coefs.size() == spectral_size() * stride,
         "Sizes don't match: " << spectral_coefs.size() << " vs "
                               << spectral_size() * stride);
  FirstDeriv result(physical_size() * stride);
  std::array<double*, 2> temp = {{result.get(0).data(), result.get(1).data()}};
  gradient_from_coefs_impl(temp, spectral_coefs.data(), stride, 0, stride, 0,
                           true);
  return result;
}

/// \cond DOXYGEN_FAILS_TO_PARSE_THIS
void YlmSpherepack::scalar_laplacian(
    const gsl::not_null<double*> scalar_laplacian,
    const gsl::not_null<const double*> collocation_values,
    const size_t physical_stride, const size_t physical_offset) const noexcept {
  auto& f_k = memory_pool_.get(spectral_size());
  phys_to_spec(f_k.data(), collocation_values, physical_stride, physical_offset,
               1, 0);
  scalar_laplacian_from_coefs(scalar_laplacian, f_k.data(), 1, 0,
                              physical_stride, physical_offset);
  memory_pool_.free(f_k);
}
/// \endcond

/// \cond DOXYGEN_FAILS_TO_PARSE_THIS
void YlmSpherepack::scalar_laplacian_from_coefs(
    const gsl::not_null<double*> scalar_laplacian,
    const gsl::not_null<const double*> spectral_coefs,
    const size_t spectral_stride, const size_t spectral_offset,
    const size_t physical_stride, const size_t physical_offset) const noexcept {
  const size_t l1 = m_max_ + 1;
  const double* const a = spectral_coefs;
  // clang-tidy: 'do not use pointer arithmetic'.
  const double* const b = a + l1 * n_theta_ * spectral_stride;  // NOLINT
  const size_t work_size = n_theta_ * (3 * n_phi_ + 2 * l1 + 1);
  auto& work = memory_pool_.get(work_size);
  int err = 0;
  auto& work_scalar_spec_to_phys = storage_.work_scalar_spec_to_phys;
  slapgs_(static_cast<int>(physical_stride), static_cast<int>(spectral_stride),
          static_cast<int>(physical_offset), static_cast<int>(spectral_offset),
          static_cast<int>(n_theta_), static_cast<int>(n_phi_), 0, 1,
          scalar_laplacian, static_cast<int>(n_theta_),
          static_cast<int>(n_phi_), a, b, static_cast<int>(l1),
          static_cast<int>(n_theta_), work_scalar_spec_to_phys.data(),
          static_cast<int>(work_scalar_spec_to_phys.size()), work.data(),
          static_cast<int>(work_size), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("slapgs error " << err << " in YlmSpherepack");
  }
  memory_pool_.free(work);
}
/// \endcond

DataVector YlmSpherepack::scalar_laplacian(const DataVector& collocation_values,
                                           const size_t physical_stride,
                                           const size_t physical_offset) const
    noexcept {
  ASSERT(collocation_values.size() == physical_size() * physical_stride,
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size() * physical_stride);
  DataVector result(physical_size());
  scalar_laplacian(result.data(), collocation_values.data(), physical_stride,
                   physical_offset);
  return result;
}

DataVector YlmSpherepack::scalar_laplacian_from_coefs(
    const DataVector& spectral_coefs, const size_t spectral_stride,
    const size_t spectral_offset) const noexcept {
  ASSERT(spectral_coefs.size() == spectral_size() * spectral_stride,
         "Sizes don't match: " << spectral_coefs.size() << " vs "
                               << spectral_size() * spectral_stride);
  DataVector result(physical_size());
  scalar_laplacian_from_coefs(result.data(), spectral_coefs.data(),
                              spectral_stride, spectral_offset);
  return result;
}

std::array<DataVector, 2> YlmSpherepack::theta_phi_points() const noexcept {
  std::array<DataVector, 2> result = make_array<2>(DataVector(physical_size()));
  const auto& theta = theta_points();
  const auto& phi = phi_points();
  // Storage in SPHEREPACK: theta varies fastest (i.e. fortran ordering).
  for (size_t i_phi = 0, s = 0; i_phi < n_phi_; ++i_phi) {
    for (size_t i_theta = 0; i_theta < n_theta_; ++i_theta, ++s) {
      result[0][s] = theta[i_theta];
      result[1][s] = phi[i_phi];
    }
  }
  return result;
}

const std::vector<double>& YlmSpherepack::theta_points() const noexcept {
  auto& theta = storage_.theta;
  if (theta.empty()) {
    theta.resize(n_theta_);
    auto& work = memory_pool_.get(n_theta_ + 1);
    auto& unused_weights = memory_pool_.get(n_theta_);
    int err = 0;
    gaqd_(static_cast<int>(n_theta_), theta.data(), unused_weights.data(),
          work.data(), static_cast<int>(n_theta_ + 1), &err);
    memory_pool_.free(unused_weights);
    memory_pool_.free(work);
    if (UNLIKELY(err != 0)) {
      ERROR("gaqd error " << err << " in YlmSpherepack");
    }
  }
  return theta;
}

const std::vector<double>& YlmSpherepack::phi_points() const noexcept {
  // The following is not static or constexpr because n_phi_ depends
  // on *this.
  const double two_pi_over_n_phi = 2.0 * M_PI / n_phi_;
  auto& phi = storage_.phi;
  if (phi.empty()) {
    phi.resize(n_phi_);
    for (size_t i = 0; i < n_phi_; ++i) {
      phi[i] = two_pi_over_n_phi * i;
    }
  }
  return phi;
}

void YlmSpherepack::second_derivative(
    const std::array<double*, 2>& df, gsl::not_null<SecondDeriv*> ddf,
    const gsl::not_null<const double*> collocation_values,
    const size_t physical_stride, const size_t physical_offset) const noexcept {
  // Initialize trig functions at collocation points
  auto& cos_theta = storage_.cos_theta;
  auto& sin_theta = storage_.sin_theta;
  auto& sin_phi = storage_.sin_phi;
  auto& cos_phi = storage_.cos_phi;
  auto& cot_theta = storage_.cot_theta;
  auto& cosec_theta = storage_.cosec_theta;
  if (cos_theta.empty()) {
    cos_theta.resize(n_theta_);
    sin_theta.resize(n_theta_);
    cosec_theta.resize(n_theta_);
    cot_theta.resize(n_theta_);
    cos_phi.resize(n_phi_);
    sin_phi.resize(n_phi_);

    const std::vector<double>& theta = theta_points();
    for (size_t i = 0; i < n_theta_; ++i) {
      cos_theta[i] = cos(theta[i]);
      sin_theta[i] = sin(theta[i]);
      cosec_theta[i] = 1.0 / sin_theta[i];
      cot_theta[i] = cos_theta[i] * cosec_theta[i];
    }
    const std::vector<double>& phi = phi_points();
    for (size_t i = 0; i < n_phi_; ++i) {
      cos_phi[i] = cos(phi[i]);
      sin_phi[i] = sin(phi[i]);
    }
  }
  // Get first derivatives
  gradient(df, collocation_values, physical_stride, physical_offset);

  // Now get Cartesian derivatives.

  // First derivative
  std::vector<double*> dfc(3, nullptr);
  for (size_t i = 0; i < 3; ++i) {
    dfc[i] = memory_pool_.get(physical_size()).data();
  }
  for (size_t j = 0, s = 0; j < n_phi_; ++j) {
    for (size_t i = 0; i < n_theta_; ++i, ++s) {
      dfc[0][s] = cos_theta[i] * cos_phi[j] *
                      df[0][s * physical_stride + physical_offset] -
                  sin_phi[j] * df[1][s * physical_stride + physical_offset];
      dfc[1][s] = cos_theta[i] * sin_phi[j] *
                      df[0][s * physical_stride + physical_offset] +
                  cos_phi[j] * df[1][s * physical_stride + physical_offset];
      dfc[2][s] = -sin_theta[i] * df[0][s * physical_stride + physical_offset];
    }
  }

  // Take derivatives of Cartesian derivatives to get second derivatives.
  std::vector<std::array<double*, 2>> ddfc(3, {{nullptr, nullptr}});
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      gsl::at(ddfc[i], j) = memory_pool_.get(physical_size()).data();
    }
    gradient(ddfc[i], dfc[i], 1, 0);
    // Here we free the storage for dfc[i], so we can reuse it in ddfc.
    memory_pool_.free(dfc[i]);
  }

  // Combine into Pfaffian second derivatives
  for (size_t j = 0, s = 0; j < n_phi_; ++j) {
    for (size_t i = 0; i < n_theta_; ++i, ++s) {
      ddf->get(1, 0)[s * physical_stride + physical_offset] =
          -ddfc[2][1][s] * cosec_theta[i];
      ddf->get(0, 1)[s * physical_stride + physical_offset] =
          ddf->get(1, 0)[s * physical_stride + physical_offset] -
          cot_theta[i] * df[1][s * physical_stride + physical_offset];
      ddf->get(1, 1)[s * physical_stride + physical_offset] =
          cos_phi[j] * ddfc[1][1][s] - sin_phi[j] * ddfc[0][1][s] -
          cot_theta[i] * df[0][s * physical_stride + physical_offset];
      ddf->get(0, 0)[s * physical_stride + physical_offset] =
          cos_theta[i] *
              (cos_phi[j] * ddfc[0][0][s] + sin_phi[j] * ddfc[1][0][s]) -
          sin_theta[i] * ddfc[2][0][s];
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      memory_pool_.free(gsl::at(ddfc[i], j));
    }
  }
}

std::pair<YlmSpherepack::FirstDeriv, YlmSpherepack::SecondDeriv>
YlmSpherepack::first_and_second_derivative(
    const DataVector& collocation_values) const noexcept {
  ASSERT(collocation_values.size() == physical_size(),
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size());
  std::pair<FirstDeriv, SecondDeriv> result(
      std::piecewise_construct, std::forward_as_tuple(physical_size()),
      std::forward_as_tuple(physical_size()));
  std::array<double*, 2> temp = {
      {std::get<0>(result).get(0).data(), std::get<0>(result).get(1).data()}};
  second_derivative(temp, &std::get<1>(result), collocation_values.data());
  return result;
}

YlmSpherepack::InterpolationInfoPerPoint::InterpolationInfoPerPoint(
    const size_t m_max, const std::vector<double>& pmm,
    const std::array<double, 2>& target)
    : cos_theta(cos(target[0])),
      cos_m_phi(m_max + 1),
      sin_m_phi(m_max + 1),
      pbar_factor(m_max + 1) {
  const double sin_theta = sin(target[0]);
  const double phi = target[1];

  // Evaluate cos(m*phi) and sin(m*phi) by numerical recipes eq. 5.5.6
  {
    cos_m_phi[0] = 1.0;
    sin_m_phi[0] = 0.0;
    const double beta = sin(phi);
    const double alpha = 2.0 * square(sin(0.5 * phi));
    double sinmphi = 0.0;
    double cosmphi = 1.0;
    for (size_t m = 1; m < m_max + 1; ++m) {
      const double deltacosmphi = alpha * cosmphi + beta * sinmphi;
      const double deltasinmphi = alpha * sinmphi - beta * cosmphi;
      cosmphi -= deltacosmphi;
      sinmphi -= deltasinmphi;
      cos_m_phi[m] = cosmphi;
      sin_m_phi[m] = sinmphi;
    }
  }

  // Fill pbar_factor[m] = Pbar(m,m)*sin(theta)^m  (m<l1)
  double sinmtheta = 1.0;
  for (size_t m = 0; m < m_max + 1; ++m) {
    pbar_factor[m] = pmm[m] * sinmtheta;
    sinmtheta *= sin_theta;
  }
}

YlmSpherepack::InterpolationInfo YlmSpherepack::set_up_interpolation_info(
    const std::vector<std::array<double, 2>>& target_points) const noexcept {
  // SPHEREPACK expands f(theta,phi) as
  //
  // f(theta,phi) =
  // 1/2 Sum_{l=0}^{l_max} Pbar(l,0) a(0,l)
  //   + Sum_{m=1}^{m_max} Sum_{l=m}^{l_max} Pbar(l,m)(  a(m,l) cos(m phi)
  //                                                   - b(m,l) sin(m phi))
  //
  // where Pbar(l,m) are unit-orthonormal Associated Legendre
  // polynomials (which are functions of x = cos(theta)), and a(m,l)
  // and b(m,l) are the SPHEREPACK spectral coefficients.
  //
  // Note that Pbar(l,m) = sqrt((2l+1)(l-m)!/(2(l+m)!)) P_l^m.
  // and that Integral_{-1}^{+1} Pbar(k,m)(x) Pbar(l,m)(x) = delta_kl
  //
  // We will interpolate via Clenshaw's recurrence formula in l, for fixed m.
  //
  // The recursion relation between Associated Legendre polynomials is
  // Pbar(l+1)(m) = alpha(l,x) Pbar(l)(m) + beta(l,x) Pbar(l-1)(m)
  // where alpha(l,x) = x sqrt((2l+3)(2l+1)/((l+1-m)(l+1+m)))
  // where beta(l,x)  = - sqrt((2l+3)(l-m)(l+m)/((l+1-m)(l+1+m)(2l-1)))
  //
  // The Clenshaw recurrence formula for
  // f(x) = Sum_{l=m}^{l_max} c_(l)(m) Pbar(l)(m) is
  // y_{l_max+1} = y_{l_max+2} = 0
  // y_k  = alpha(k,x) y_{k+1} + beta(k+1,x) y_{k+2} + c_(k)(m)  (m<k<=l_max)
  // f(x) = beta(m+1,x) Pbar(m,m) y_{m+2} + Pbar(m+1,m) y_{m+1}
  //      + Pbar(m,m) c_(m)(m).
  //
  // So we will compute and store alpha(l,x)/x in 'alpha' and
  // beta(l+1,x) [NOT beta(l,x)] in 'beta'.  We will also compute and
  // store the x-independent piece of Pbar(m)(m) in 'pmm' and we
  // will compute and store the x-independent piece of Pbar(m+1)(m)/Pbar(m)(m)
  // in a component of 'alpha'. See below for storage.
  // Note Pbar(m)(m)   is (2m-1)!! (1-x^2)^(n/2) sqrt((2m+1)/(2 (2m)!))
  // and  Pbar(m+1)(m) is (2m+1)!! x(1-x^2)^(n/2)sqrt((2m+3)/(2(2m+1)!))
  //  Ratio Pbar(m+1)(m)/Pbar(m)(m)   = x sqrt(2m+3)
  //  Ratio Pbar(m+1)(m+1)/Pbar(m)(m) = sqrt(1-x^2) sqrt((2m+3)/(2m+2))

  const size_t l1 = m_max_ + 1;

  auto& alpha = storage_.work_interp_alpha;
  auto& beta = storage_.work_interp_beta;
  auto& pmm = storage_.work_interp_pmm;
  auto& index = storage_.work_interp_index;
  if (alpha.empty()) {
    // We need to compute alpha, beta, index, and pmm only once.
    const size_t array_size = n_theta_ * l1 - l1 * (l1 - 1) / 2;
    index.resize(array_size);
    alpha.resize(array_size);
    beta.resize(array_size);
    pmm.resize(l1);

    // Fill alpha,beta,index arrays in the same order as the Clenshaw
    // recurrence, so that we can index them easier during the recurrence.
    // First do m=0.
    size_t idx = 0;
    for (size_t n = n_theta_ - 1; n > 0; --n, ++idx) {
      const double tnp1 = 2.0 * n + 1;
      const double np1sq = n * n + 2.0 * n + 1.0;
      alpha[idx] = sqrt(tnp1 * (tnp1 + 2.0) / np1sq);
      beta[idx] = -sqrt((tnp1 + 4.0) / tnp1 * np1sq / (np1sq + 2 * n + 3));
      index[idx] = n * l1;
    }
    // The next value of beta stores beta(n=1,m=0).
    // The next value of alpha stores Pbar(n=1,m=0)/(x*Pbar(n=0,m=0)).
    // These two values are needed for the final Clenshaw recurrence formula.
    beta[idx] = -0.5 * sqrt(5.0);
    alpha[idx] = sqrt(3.0);
    index[idx] = 0;  // Index of coef in the final recurrence formula
    ++idx;

    // Now do other m.
    for (size_t m = 1; m < l1; ++m) {
      for (size_t n = n_theta_ - 1; n > m; --n, ++idx) {
        const double tnp1 = 2.0 * n + 1;
        const double np1sqmmsq = (n + 1.0 + m) * (n + 1.0 - m);
        alpha[idx] = sqrt(tnp1 * (tnp1 + 2.0) / np1sqmmsq);
        beta[idx] =
            -sqrt((tnp1 + 4.0) / tnp1 * np1sqmmsq / (np1sqmmsq + 2. * n + 3.));
        index[idx] = m + n * l1;
      }
      // The next value of beta stores beta(n=m+1,m).
      // The next value of alpha stores Pbar(n=m+1,m)/(x*Pbar(n=m,m)).
      // These two values are needed for the final Clenshaw recurrence formula.
      beta[idx] = -0.5 * sqrt((2.0 * m + 5) / (m + 1.0));
      alpha[idx] = sqrt(2.0 * m + 3);
      index[idx] =
          m + m * l1;  // Index of coef in the final recurrence formula.
      ++idx;
    }
    ASSERT(idx == index.size(), "Wrong size " << idx << ", expected "
                                              << index.size());

    // Now do pmm, which stores Pbar(m,m).
    pmm[0] = M_SQRT1_2;  // 1/sqrt(2) = Pbar(0)(0)
    for (size_t m = 1; m < l1; ++m) {
      pmm[m] = pmm[m - 1] * sqrt((2.0 * m + 1.0) / (2.0 * m));
    }
  }

  // Now fill interpolation_info.
  InterpolationInfo interpolation_info;
  interpolation_info.reserve(target_points.size());
  for (const auto& pt : target_points) {
    interpolation_info.emplace_back(m_max_, pmm, pt);
  }
  return interpolation_info;
}

/// \cond DOXYGEN_FAILS_TO_PARSE_THIS
void YlmSpherepack::interpolate(
    const gsl::not_null<std::vector<double>*> result,
    const gsl::not_null<const double*> collocation_values,
    const InterpolationInfo& interpolation_info, const size_t physical_stride,
    const size_t physical_offset) const noexcept {
  ASSERT(result->size() == interpolation_info.size(),
         "Size mismatch: " << result->size() << ","
                           << interpolation_info.size());
  auto& f_k = memory_pool_.get(spectral_size());
  phys_to_spec(f_k.data(), collocation_values, physical_stride, physical_offset,
               1, 0);
  interpolate_from_coefs(result, f_k, interpolation_info);
  memory_pool_.free(f_k);
}
/// \endcond

template <typename T>
void YlmSpherepack::interpolate_from_coefs(
    const gsl::not_null<std::vector<double>*> result, const T& spectral_coefs,
    const InterpolationInfo& interpolation_info, const size_t spectral_stride,
    const size_t spectral_offset) const noexcept {
  const auto& alpha = storage_.work_interp_alpha;
  const auto& beta = storage_.work_interp_beta;
  const auto& index = storage_.work_interp_index;
  // alpha holds alpha(n,m,x)/x, beta holds beta(n+1,m).
  // index holds the index into the coefficient array.
  // All are indexed together.

  ASSERT(result->size() == interpolation_info.size(),
         "Size mismatch: " << result->size() << ","
                           << interpolation_info.size());

  const size_t l1 = m_max_ + 1;

  // Offsets of 'a' and 'b' in spectral_coefs.
  const size_t a_offset = spectral_offset;
  const size_t b_offset = spectral_offset + (l1 * n_theta_) * spectral_stride;

  std::fill(result->begin(), result->end(), 0.0);

  for (size_t i = 0; i < result->size(); ++i) {
    const auto& intrp_info = interpolation_info[i];
    const auto& cos_theta = intrp_info.cos_theta;

    // Clenshaw recurrence for m=0.  Separate because there is no phi
    // dependence, and there is a factor of 1/2.
    size_t idx = 0;
    {
      double ycn = 0.0;
      double ycnp1 = 0.0;
      for (size_t n = n_theta_ - 1; n > 0;
           --n, ++idx) {  // Loops from n_theta_-1 to 1.
        double ycnp2 = ycnp1;
        ycnp1 = ycn;
        ycn = cos_theta * alpha[idx] * ycnp1 + beta[idx] * ycnp2 +
              spectral_coefs[a_offset + spectral_stride * index[idx]];
      }
      (*result)[i] += 0.5 * intrp_info.pbar_factor[0] *
                      (beta[idx] * ycnp1 + cos_theta * alpha[idx] * ycn +
                       spectral_coefs[a_offset + spectral_stride * index[idx]]);
      ++idx;
    }
    // Now do recurrence for other m.
    for (size_t m = 1; m < l1; ++m) {
      double ycn = 0.0;
      double ycnp1 = 0.0;
      double ysn = 0.0;
      double ysnp1 = 0.0;
      for (size_t n = n_theta_ - 1; n > m; --n, ++idx) {
        double ycnp2 = ycnp1;
        double ysnp2 = ysnp1;
        ycnp1 = ycn;
        ysnp1 = ysn;
        ycn = cos_theta * alpha[idx] * ycnp1 + beta[idx] * ycnp2 +
              spectral_coefs[a_offset + spectral_stride * index[idx]];
        ysn = cos_theta * alpha[idx] * ysnp1 + beta[idx] * ysnp2 +
              spectral_coefs[b_offset + spectral_stride * index[idx]];
      }
      const double fc =
          intrp_info.pbar_factor[m] *
          (beta[idx] * ycnp1 + cos_theta * alpha[idx] * ycn +
           spectral_coefs[a_offset + spectral_stride * index[idx]]);
      const double fs =
          intrp_info.pbar_factor[m] *
          (beta[idx] * ysnp1 + cos_theta * alpha[idx] * ysn +
           spectral_coefs[b_offset + spectral_stride * index[idx]]);
      (*result)[i] +=
          fc * intrp_info.cos_m_phi[m] - fs * intrp_info.sin_m_phi[m];
      ++idx;
    }
    ASSERT(idx == index.size(), "Wrong size " << idx << ", expected "
                                              << index.size());
  }
}

std::vector<double> YlmSpherepack::interpolate(
    const DataVector& collocation_values,
    const std::vector<std::array<double, 2>>& target_points) const noexcept {
  ASSERT(collocation_values.size() == physical_size(),
         "Sizes don't match: " << collocation_values.size() << " vs "
                               << physical_size());
  std::vector<double> result(target_points.size());
  interpolate(&result, collocation_values.data(),
              set_up_interpolation_info(target_points));
  return result;
}

std::vector<double> YlmSpherepack::interpolate_from_coefs(
    const DataVector& spectral_coefs,
    const std::vector<std::array<double, 2>>& target_points) const noexcept {
  ASSERT(spectral_coefs.size() == spectral_size(),
         "Sizes don't match: " << spectral_coefs.size() << " vs "
                               << spectral_size());
  std::vector<double> result(target_points.size());
  interpolate_from_coefs(&result, spectral_coefs,
                         set_up_interpolation_info(target_points));
  return result;
}

void YlmSpherepack::fill_vector_work_arrays() const noexcept {
  const size_t l2 = (n_theta_ + 1) / 2;
  auto& work_vector_spec_to_phys = storage_.work_vector_spec_to_phys;

  // Allocate memory for vector work arrays
  // Throughout this file, the l1, mdb, mdc values
  // are set to min(n_theta_, (n_phi_+2)/2)), which is consistent
  // with l1 and mdab for scalars.  This is generally
  // larger than what is required by Spherepack
  // for vectors: min(n_theta_, (n_phi_+1)/2)).
  const size_t l_vhsgs =
      n_theta_ * l2 * (n_theta_ + 1) + n_phi_ + 15 + 2 * n_theta_;
  work_vector_spec_to_phys.assign(l_vhsgs, 0.0);

  // Initialize vector work arrays
  const size_t ldwg = (3 * n_theta_ * (n_theta_ + 3) + 2) / 2;
  auto& dworkg = memory_pool_.get(ldwg);
  int err = 0;
  vhsgsi_(static_cast<int>(n_theta_), static_cast<int>(n_phi_),
          work_vector_spec_to_phys.data(), static_cast<int>(l_vhsgs),
          dworkg.data(), static_cast<int>(ldwg), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("vhsgsi error " << err << " in YlmSpherepack");
  }
  memory_pool_.free(dworkg);
}

void YlmSpherepack::fill_scalar_work_arrays() const noexcept {
  auto& work_phys_to_spec = storage_.work_phys_to_spec;
  auto& work_scalar_spec_to_phys = storage_.work_scalar_spec_to_phys;
  auto& quadrature_weights = storage_.quadrature_weights;

  {
    auto& weights = memory_pool_.get(n_theta_);
    auto& work0 = memory_pool_.get(n_theta_);
    auto& work1 = memory_pool_.get(n_theta_ + 1);
    int err = 0;
    gaqd_(static_cast<int>(n_theta_), work0.data(), weights.data(),
          work1.data(), static_cast<int>(n_theta_ + 1), &err);
    memory_pool_.free(work0);
    memory_pool_.free(work1);
    if (UNLIKELY(err != 0)) {
      ERROR("gaqd error " << err << " in YlmSpherepack");
    }

    quadrature_weights.assign(n_theta_ * n_phi_, 2 * M_PI / n_phi_);
    for (size_t i = 0; i < n_theta_; ++i) {
      for (size_t j = 0; j < n_phi_; ++j) {
        quadrature_weights[i + j * n_theta_] *= weights[i];
      }
    }
    memory_pool_.free(weights);
  }

  // Allocate memory for scalar work arrays.

  // Below note that l1, l2, and ylm_work_size are ints, not size_ts, and
  // note the use of n_phi and n_theta.  The reason for this: The last
  // term in the expression for ylm_work_size can sometimes be negative.  If
  // evaluated using unsigned ints instead of ints, then this will
  // underflow and give a huge number.
  const auto l1 = static_cast<int>(m_max_ + 1);
  const auto l2 = static_cast<int>(n_theta_ + 1) / 2;
  const auto n_phi = static_cast<int>(n_phi_);
  const auto n_theta = static_cast<int>(n_theta_);
  const int ylm_work_size = n_phi + 15 + n_theta * (3 * (l1 + l2) - 2) +
                            (l1 - 1) * (l2 * (2 * n_theta - l1) - 3 * l1) / 2;
  ASSERT(ylm_work_size >= 0, "Bad size " << ylm_work_size);
  const auto l_ylm = static_cast<size_t>(ylm_work_size);

  work_phys_to_spec.assign(l_ylm, 0.0);
  work_scalar_spec_to_phys.assign(l_ylm, 0.0);

  // Initialize scalar work arrays
  const size_t work_size = 4 * n_theta_ * (n_theta_ + 2) + 2;
  auto& work = memory_pool_.get(work_size);
  const size_t deriv_work_size = n_theta_ * (n_theta_ + 4);
  auto& deriv_work = memory_pool_.get(deriv_work_size);
  int err = 0;
  shagsi_(n_theta, n_phi, work_phys_to_spec.data(), ylm_work_size, work.data(),
          static_cast<int>(work_size), deriv_work.data(),
          static_cast<int>(deriv_work_size), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("shagsi error " << err << " in YlmSpherepack");
  }
  shsgsi_(n_theta, n_phi, work_scalar_spec_to_phys.data(), ylm_work_size,
          work.data(), static_cast<int>(work_size), deriv_work.data(),
          static_cast<int>(deriv_work_size), &err);
  if (UNLIKELY(err != 0)) {
    ERROR("shsgsi error " << err << " in YlmSpherepack");
  }
  memory_pool_.free(deriv_work);
  memory_pool_.free(work);
}

DataVector YlmSpherepack::prolong_or_restrict(const DataVector& spectral_coefs,
                                              const YlmSpherepack& target) const
    noexcept {
  ASSERT(spectral_coefs.size() == spectral_size(),
         "Expecting " << spectral_size() << ", got " << spectral_coefs.size());
  DataVector result(target.spectral_size(), 0.0);
  SpherepackIterator src_it(l_max_, m_max_),
      dest_it(target.l_max_, target.m_max_);
  for (; dest_it; ++dest_it) {
    if (dest_it.l() <= src_it.l_max() and dest_it.m() <= src_it.m_max()) {
      src_it.set(dest_it.l(), dest_it.m(), dest_it.coefficient_array());
      result[dest_it()] = spectral_coefs[src_it()];
    }
  }
  return result;
}

bool operator==(const YlmSpherepack& lhs, const YlmSpherepack& rhs) noexcept {
  return lhs.l_max() == rhs.l_max() and lhs.m_max() == rhs.m_max();
}

bool operator!=(const YlmSpherepack& lhs, const YlmSpherepack& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
template void YlmSpherepack::interpolate_from_coefs<std::vector<double>>(
    const gsl::not_null<std::vector<double>*>, const std::vector<double>&,
    const InterpolationInfo&, size_t, size_t) const noexcept;
