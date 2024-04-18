// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"

#include <deque>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/FiniteDifference/NonUniform1D.hpp"
#include "NumericalAlgorithms/Interpolation/LinearLeastSquares.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace ylm {
template <typename Fr>
Scalar<DataVector> radius(const Strahlkorper<Fr>& strahlkorper) {
  Scalar<DataVector> result{
      DataVector{strahlkorper.ylm_spherepack().physical_size()}};
  radius(make_not_null(&result), strahlkorper);
  return result;
}

template <typename Fr>
void radius(const gsl::not_null<Scalar<DataVector>*> result,
            const Strahlkorper<Fr>& strahlkorper) {
  get(*result) =
      strahlkorper.ylm_spherepack().spec_to_phys(strahlkorper.coefficients());
}

template <typename Fr>
tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>> theta_phi(
    const Strahlkorper<Fr>& strahlkorper) {
  tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>> result{
      DataVector{strahlkorper.ylm_spherepack().physical_size()}};
  theta_phi(make_not_null(&result), strahlkorper);
  return result;
}

template <typename Fr>
void theta_phi(
    const gsl::not_null<tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>*>
        theta_phi,
    const Strahlkorper<Fr>& strahlkorper) {
  // If ylm_spherepack ever gets a not-null version of theta_phi_points,
  // then that can be used here to avoid an allocation.
  auto temp = strahlkorper.ylm_spherepack().theta_phi_points();
  get<0>(*theta_phi) = temp[0];
  get<1>(*theta_phi) = temp[1];
}

template <typename Fr>
tnsr::i<DataVector, 3, Fr> rhat(
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi) {
  tnsr::i<DataVector, 3, Fr> result{DataVector{theta_phi[0].size()}};
  rhat(make_not_null(&result), theta_phi);
  return result;
}

template <typename Fr>
void rhat(const gsl::not_null<tnsr::i<DataVector, 3, Fr>*> r_hat,
          const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi) {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);

  // Use one component of rhat for temporary storage, to avoid allocations.
  get<1>(*r_hat) = sin(theta);

  get<0>(*r_hat) = get<1>(*r_hat) * cos(phi);
  get<1>(*r_hat) *= sin(phi);
  get<2>(*r_hat) = cos(theta);
}

template <typename Fr>
ylm::Tags::aliases::Jacobian<Fr> jacobian(
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi) {
  ylm::Tags::aliases::Jacobian<Fr> result{DataVector{theta_phi[0].size()}};
  jacobian(make_not_null(&result), theta_phi);
  return result;
}

template <typename Fr>
void jacobian(const gsl::not_null<ylm::Tags::aliases::Jacobian<Fr>*> jac,
              const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi) {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  set_number_of_grid_points(jac, theta);

  // Use 1,0 component of jac for temporary storage, to avoid allocations.
  get<1, 0>(*jac) = cos(theta);

  // Fill in components in carefully chosen order so as to avoid allocations.
  get<0, 1>(*jac) = -sin(phi);                          // 1/(R sin(th)) dx/dph
  get<1, 1>(*jac) = cos(phi);                           // 1/(R sin(th)) dy/dph
  get<2, 1>(*jac) = 0.0;                                // 1/(R sin(th)) dz/dph
  get<0, 0>(*jac) = get<1, 0>(*jac) * get<1, 1>(*jac);  // 1/R dx/dth
  get<1, 0>(*jac) *= -get<0, 1>(*jac);                  // 1/R dy/dth
  get<2, 0>(*jac) = -sin(theta);                        // 1/R dz/dth
}

template <typename Fr>
ylm::Tags::aliases::InvJacobian<Fr> inv_jacobian(
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi) {
  ylm::Tags::aliases::InvJacobian<Fr> result{DataVector{theta_phi[0].size()}};
  inv_jacobian(make_not_null(&result), theta_phi);
  return result;
}

template <typename Fr>
void inv_jacobian(
    const gsl::not_null<ylm::Tags::aliases::InvJacobian<Fr>*> inv_jac,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi) {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  set_number_of_grid_points(inv_jac, theta);

  // Use 0,1 component of inv_jac for temporary storage, to avoid allocations.
  get<0, 1>(*inv_jac) = cos(theta);

  // Fill in components in carefully chosen order so as to avoid allocations.
  get<1, 0>(*inv_jac) = -sin(phi);  // R sin(th) dph/dx
  get<1, 1>(*inv_jac) = cos(phi);   // R sin(th) dph/dy
  get<1, 2>(*inv_jac) = 0.0;        // R sin(th) dph/dz
  get<0, 0>(*inv_jac) = get<0, 1>(*inv_jac) * get<1, 1>(*inv_jac);  // R dth/dx
  get<0, 1>(*inv_jac) *= -get<1, 0>(*inv_jac);                      // R dth/dy
  get<0, 2>(*inv_jac) = -sin(theta);                                // R dth/dz
}

template <typename Fr>
ylm::Tags::aliases::InvHessian<Fr> inv_hessian(
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi) {
  ylm::Tags::aliases::InvHessian<Fr> result{DataVector{theta_phi[0].size()}};
  inv_hessian(make_not_null(&result), theta_phi);
  return result;
}

template <typename Fr>
void inv_hessian(
    const gsl::not_null<ylm::Tags::aliases::InvHessian<Fr>*> inv_hess,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi) {
  const auto& theta = get<0>(theta_phi);
  const auto& phi = get<1>(theta_phi);
  set_number_of_grid_points(inv_hess, theta);

  // Use some components of inv_hess for temporary storage, to avoid
  // allocations.
  get<1, 2, 2>(*inv_hess) = cos(theta);
  get<1, 1, 2>(*inv_hess) = sin(theta);
  get<1, 0, 2>(*inv_hess) = cos(phi);
  get<0, 2, 1>(*inv_hess) = sin(phi);
  get<1, 1, 1>(*inv_hess) = square(get<1, 1, 2>(*inv_hess));  // sin^2 theta
  get<0, 2, 0>(*inv_hess) = 1.0 / get<1, 1, 2>(*inv_hess);    // csc theta
  get<1, 2, 1>(*inv_hess) = square(get<0, 2, 1>(*inv_hess));  // sin^2 phi
  get<0, 2, 2>(*inv_hess) = square(get<1, 0, 2>(*inv_hess));  // cos^2 phi

  // Fill in components of inv_hess in carefully chosen order,
  // to avoid allocations.

  // R^2 d^2 th/(dydz)
  get<0, 1, 2>(*inv_hess) =
      (2.0 * get<1, 1, 1>(*inv_hess) - 1.0) * get<0, 2, 1>(*inv_hess);
  // R^2 d/dz (sin(th) dph/dx)
  get<1, 2, 0>(*inv_hess) = get<1, 2, 2>(*inv_hess) * get<0, 2, 1>(*inv_hess);

  // cos(phi) sin(phi) [overwrites sin_phi]
  get<0, 2, 1>(*inv_hess) = get<1, 0, 2>(*inv_hess) * get<0, 2, 1>(*inv_hess);

  // Fill in components of inv_hess in carefully chosen order,
  // to avoid allocations.

  // R^2 d/dx (sin(th) dph/dx)
  get<1, 0, 0>(*inv_hess) = get<0, 2, 1>(*inv_hess) *
                            (1.0 + get<1, 1, 1>(*inv_hess)) *
                            get<0, 2, 0>(*inv_hess);
  // R^2 d/dx (sin(th) dph/dy)
  get<1, 0, 1>(*inv_hess) =
      (get<1, 2, 1>(*inv_hess) -
       get<1, 1, 1>(*inv_hess) * get<0, 2, 2>(*inv_hess)) *
      get<0, 2, 0>(*inv_hess);
  // R^2 d/dy (sin(th) dph/dx)
  get<1, 1, 0>(*inv_hess) = (get<1, 1, 1>(*inv_hess) * get<1, 2, 1>(*inv_hess) -
                             get<0, 2, 2>(*inv_hess)) *
                            get<0, 2, 0>(*inv_hess);

  // More temps: now don't need csc theta anymore
  get<0, 2, 0>(*inv_hess) *= get<1, 2, 2>(*inv_hess);  // cot(theta)
  // 1 + 2 sin^2(theta)
  get<0, 0, 1>(*inv_hess) = 1.0 + 2.0 * get<1, 1, 1>(*inv_hess);

  // Fill in more inv_hess components in careful order, since some
  // of those components still contain temporaries.

  // R^2 d^2 th/(dy^2)
  get<0, 1, 1>(*inv_hess) =
      get<0, 2, 0>(*inv_hess) *
      (1.0 - get<1, 2, 1>(*inv_hess) * get<0, 0, 1>(*inv_hess));
  // R^2 d^2 th/(dx^2)
  get<0, 0, 0>(*inv_hess) =
      get<0, 2, 0>(*inv_hess) *
      (1.0 - get<0, 2, 2>(*inv_hess) * get<0, 0, 1>(*inv_hess));
  // R^2 d^2 th/(dxdy)
  get<0, 0, 1>(*inv_hess) = -get<0, 2, 0>(*inv_hess) * get<0, 2, 1>(*inv_hess) *
                            get<0, 0, 1>(*inv_hess);
  // R^2 d^2 th/(dxdz)
  get<0, 0, 2>(*inv_hess) =
      (2.0 * get<1, 1, 1>(*inv_hess) - 1.0) * get<1, 0, 2>(*inv_hess);
  // R^2 d/dz (sin(th) dph/dy)
  get<1, 2, 1>(*inv_hess) = -get<1, 2, 2>(*inv_hess) * get<1, 0, 2>(*inv_hess);
  // R^2 d^2 th/(dz^2)
  get<0, 2, 2>(*inv_hess) =
      2.0 * get<1, 2, 2>(*inv_hess) * get<1, 1, 2>(*inv_hess);

  // R^2 d^2 th/(dydx)
  get<0, 1, 0>(*inv_hess) = get<0, 0, 1>(*inv_hess);
  // R^2 d^2 th/(dzdx)
  get<0, 2, 0>(*inv_hess) = get<0, 0, 2>(*inv_hess);
  // R^2 d^2 th/(dzdy)
  get<0, 2, 1>(*inv_hess) = get<0, 1, 2>(*inv_hess);
  // R^2 d/dx (sin(th) dph/dz)
  get<1, 0, 2>(*inv_hess) = 0.0;
  // R^2 d/dy (sin(th) dph/dy)
  get<1, 1, 1>(*inv_hess) = -get<1, 0, 0>(*inv_hess);
  // R^2 d/dy (sin(th) dph/dz)
  get<1, 1, 2>(*inv_hess) = 0.0;
  // R^2 d/dz (sin(th) dph/dz)
  get<1, 2, 2>(*inv_hess) = 0.0;
}

template <typename Fr>
tnsr::I<DataVector, 3, Fr> cartesian_coords(
    const Strahlkorper<Fr>& strahlkorper, const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Fr>& r_hat) {
  tnsr::I<DataVector, 3, Fr> result{DataVector{get(radius).size()}};
  cartesian_coords(make_not_null(&result), strahlkorper, radius, r_hat);
  return result;
}

template <typename Fr>
void cartesian_coords(const gsl::not_null<tnsr::I<DataVector, 3, Fr>*> coords,
                      const Strahlkorper<Fr>& strahlkorper,
                      const Scalar<DataVector>& radius,
                      const tnsr::i<DataVector, 3, Fr>& r_hat) {
  for (size_t d = 0; d < 3; ++d) {
    coords->get(d) = gsl::at(strahlkorper.expansion_center(), d) +
                     r_hat.get(d) * get(radius);
  }
}

template <typename Fr>
tnsr::I<DataVector, 3, Fr> cartesian_coords(
    const Strahlkorper<Fr>& strahlkorper) {
  const auto radius = ylm::radius(strahlkorper);
  const auto theta_phi = ylm::theta_phi(strahlkorper);
  const auto r_hat = ylm::rhat(theta_phi);
  return cartesian_coords(strahlkorper, radius, r_hat);
}

template <typename Fr>
tnsr::i<DataVector, 3, Fr> cartesian_derivs_of_scalar(
    const Scalar<DataVector>& scalar, const Strahlkorper<Fr>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const ylm::Tags::aliases::InvJacobian<Fr>& inv_jac) {
  tnsr::i<DataVector, 3, Fr> result{DataVector{get(scalar).size()}};
  cartesian_derivs_of_scalar(make_not_null(&result), scalar, strahlkorper,
                             radius_of_strahlkorper, inv_jac);
  return result;
}

template <typename Fr>
void cartesian_derivs_of_scalar(
    const gsl::not_null<tnsr::i<DataVector, 3, Fr>*> dx_scalar,
    const Scalar<DataVector>& scalar, const Strahlkorper<Fr>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const ylm::Tags::aliases::InvJacobian<Fr>& inv_jac) {
  // If ylm_spherepack().gradient() ever gets a not_null function,
  // that function can be used here.
  const auto gradient = strahlkorper.ylm_spherepack().gradient(get(scalar));

  // Use dx_scalar component as temp to store 1/r to avoid allocation.
  get<2>(*dx_scalar) = 1.0 / get(radius_of_strahlkorper);

  // Now fill in components.
  get<0>(*dx_scalar) = (get<0, 0>(inv_jac) * get<0>(gradient) +
                        get<1, 0>(inv_jac) * get<1>(gradient)) *
                       get<2>(*dx_scalar);
  get<1>(*dx_scalar) = (get<0, 1>(inv_jac) * get<0>(gradient) +
                        get<1, 1>(inv_jac) * get<1>(gradient)) *
                       get<2>(*dx_scalar);
  get<2>(*dx_scalar) *= get<0, 2>(inv_jac) * get<0>(gradient);
}

template <typename Fr>
tnsr::ii<DataVector, 3, Fr> cartesian_second_derivs_of_scalar(
    const Scalar<DataVector>& scalar, const Strahlkorper<Fr>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const ylm::Tags::aliases::InvJacobian<Fr>& inv_jac,
    const ylm::Tags::aliases::InvHessian<Fr>& inv_hess) {
  tnsr::ii<DataVector, 3, Fr> result{DataVector{get(scalar).size()}};
  cartesian_second_derivs_of_scalar(make_not_null(&result), scalar,
                                    strahlkorper, radius_of_strahlkorper,
                                    inv_jac, inv_hess);
  return result;
}

template <typename Fr>
void cartesian_second_derivs_of_scalar(
    const gsl::not_null<tnsr::ii<DataVector, 3, Fr>*> d2x_scalar,
    const Scalar<DataVector>& scalar, const Strahlkorper<Fr>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const ylm::Tags::aliases::InvJacobian<Fr>& inv_jac,
    const ylm::Tags::aliases::InvHessian<Fr>& inv_hess) {
  set_number_of_grid_points(d2x_scalar, scalar);
  for (auto& component : *d2x_scalar) {
    component = 0.0;
  }

  // If ylm_spherepack().first_and_second_derivative() ever gets a not_null
  // function, that function can be used here.
  const auto derivs =
      strahlkorper.ylm_spherepack().first_and_second_derivative(get(scalar));

  for (size_t i = 0; i < 3; ++i) {
    // Diagonal terms.  Divide by square(r) later.
    for (size_t k = 0; k < 2; ++k) {  // Angular derivs are 2-dimensional
      d2x_scalar->get(i, i) += derivs.first.get(k) * inv_hess.get(k, i, i);
      for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
        d2x_scalar->get(i, i) +=
            derivs.second.get(l, k) * inv_jac.get(k, i) * inv_jac.get(l, i);
      }
    }
    d2x_scalar->get(i, i) /= square(get(radius_of_strahlkorper));

    // off_diagonal terms.  Symmetrize over i and j.
    // Divide by 2*square(r) later.
    for (size_t j = i + 1; j < 3; ++j) {
      for (size_t k = 0; k < 2; ++k) {  // Angular derivs are 2-dimensional
        d2x_scalar->get(i, j) += derivs.first.get(k) * (inv_hess.get(k, i, j) +
                                                        inv_hess.get(k, j, i));
        for (size_t l = 0; l < 2; ++l) {  // Angular derivs are 2-dimensional
          d2x_scalar->get(i, j) +=
              derivs.second.get(l, k) * (inv_jac.get(k, i) * inv_jac.get(l, j) +
                                         inv_jac.get(k, j) * inv_jac.get(l, i));
        }
      }
      d2x_scalar->get(i, j) /= 2.0 * square(get(radius_of_strahlkorper));
    }
  }
}

template <typename Fr>
Scalar<DataVector> laplacian_of_scalar(
    const Scalar<DataVector>& scalar, const Strahlkorper<Fr>& strahlkorper,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi) {
  Scalar<DataVector> result{DataVector{get(scalar).size()}};
  laplacian_of_scalar(make_not_null(&result), scalar, strahlkorper, theta_phi);
  return result;
}

template <typename Fr>
void laplacian_of_scalar(
    const gsl::not_null<Scalar<DataVector>*> laplacian,
    const Scalar<DataVector>& scalar, const Strahlkorper<Fr>& strahlkorper,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi) {
  get(*laplacian).destructive_resize(get(scalar).size());
  // If ylm_spherepack().first_and_second_derivative() ever gets a not_null
  // function, that function can be used here.
  const auto derivs =
      strahlkorper.ylm_spherepack().first_and_second_derivative(get(scalar));
  get(*laplacian) = get<0, 0>(derivs.second) + get<1, 1>(derivs.second) +
                    get<0>(derivs.first) / tan(get<0>(theta_phi));
}

template <typename Fr>
ylm::Tags::aliases::Jacobian<Fr> tangents(
    const Strahlkorper<Fr>& strahlkorper, const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Fr>& r_hat,
    const ylm::Tags::aliases::Jacobian<Fr>& jac) {
  ylm::Tags::aliases::Jacobian<Fr> result{DataVector{get(radius).size()}};
  tangents(make_not_null(&result), strahlkorper, radius, r_hat, jac);
  return result;
}

template <typename Fr>
void tangents(const gsl::not_null<ylm::Tags::aliases::Jacobian<Fr>*> result,
              const Strahlkorper<Fr>& strahlkorper,
              const Scalar<DataVector>& radius,
              const tnsr::i<DataVector, 3, Fr>& r_hat,
              const ylm::Tags::aliases::Jacobian<Fr>& jac) {
  const auto dr = strahlkorper.ylm_spherepack().gradient(get(radius));
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result->get(j, i) =
          dr.get(i) * r_hat.get(j) + get(radius) * jac.get(j, i);
    }
  }
}

template <typename Fr>
tnsr::i<DataVector, 3, Fr> normal_one_form(
    const tnsr::i<DataVector, 3, Fr>& dx_radius,
    const tnsr::i<DataVector, 3, Fr>& r_hat) {
  tnsr::i<DataVector, 3, Fr> result{DataVector{r_hat.begin()->size()}};
  normal_one_form(make_not_null(&result), dx_radius, r_hat);
  return result;
}

template <typename Fr>
void normal_one_form(const gsl::not_null<tnsr::i<DataVector, 3, Fr>*> one_form,
                     const tnsr::i<DataVector, 3, Fr>& dx_radius,
                     const tnsr::i<DataVector, 3, Fr>& r_hat) {
  for (size_t d = 0; d < 3; ++d) {
    one_form->get(d) = r_hat.get(d) - dx_radius.get(d);
  }
}

template <typename Fr>
std::vector<std::array<double, 4>> fit_ylm_coeffs(
    const DataVector& times,
    const std::vector<Strahlkorper<Fr>>& strahlkorpers) {
  size_t num_observations = strahlkorpers.size();
  size_t num_coefficients = strahlkorpers[0].coefficients().size();
  intrp::LinearLeastSquares<3> predictor(num_observations);
  std::vector<DataVector> coefficients;
  for (size_t i = 0; i < num_coefficients; ++i) {
    DataVector same_coeff_of_each_strahlkorper(num_observations);
    for (size_t j = 0; j < num_observations; ++j) {
      // Contains the same coefficient of each strahlkorper in the time series.
      same_coeff_of_each_strahlkorper[j] = strahlkorpers[j].coefficients()[i];
      if (i == 0 and j > 0) {
        ASSERT(strahlkorpers[j - 1].l_max() == strahlkorpers[j].l_max() and
                   strahlkorpers[j - 1].m_max() == strahlkorpers[j].m_max(),
               "Each Strahlkorper must have the same l_max and m_max.");
      }
    }
    coefficients.push_back(std::move(same_coeff_of_each_strahlkorper));
  }
  return predictor.fit_coefficients(times, coefficients);
}

namespace {
template <size_t NumTimes>
static DataVector compute_coefs(
    const std::deque<double>& times,
    const std::deque<const DataVector*>& coefficients) {
  const auto weights = fd::non_uniform_1d_weights<NumTimes>(times);

  DataVector new_coefficients{coefficients.front()->size(), 0.0};

  for (size_t i = 0; i < NumTimes; i++) {
    new_coefficients += *coefficients[i] * gsl::at(gsl::at(weights, 1), i);
  }

  return new_coefficients;
}
}  // namespace

template <typename Frame>
void time_deriv_of_strahlkorper(
    gsl::not_null<Strahlkorper<Frame>*> time_deriv,
    const std::deque<std::pair<double, Strahlkorper<Frame>>>&
        previous_strahlkorpers) {
  std::deque<double> times{};
  std::deque<const DataVector*> coefficients{};

  for (const auto& [time, strahlkorper] : previous_strahlkorpers) {
    // This only happens toward the beginning because the first time is NaN and
    // if that happens we can't actually take a derivative
    if (UNLIKELY(std::isnan(time))) {
      time_deriv->coefficients() =
          DataVector{strahlkorper.coefficients().size(), 0.0};
      return;
    }

    times.emplace_back(time);
    coefficients.emplace_back(&strahlkorper.coefficients());
  }

  DataVector new_coefficients{};

  // Switch needed to convert times size into template parameter
  switch (previous_strahlkorpers.size()) {
    case 2:
      new_coefficients = compute_coefs<2>(times, coefficients);
      break;
    case 3:
      new_coefficients = compute_coefs<3>(times, coefficients);
      break;
    case 4:
      new_coefficients = compute_coefs<4>(times, coefficients);
      break;
    default:
      ERROR("Unsupported size for number of previous Strahlkorpers "
            << previous_strahlkorpers.size());
  }

  time_deriv->coefficients() = std::move(new_coefficients);
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template Scalar<DataVector> ylm::radius(                                    \
      const Strahlkorper<FRAME(data)>& strahlkorper);                         \
  template void ylm::radius(const gsl::not_null<Scalar<DataVector>*> result,  \
                            const Strahlkorper<FRAME(data)>& strahlkorper);   \
  template tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>            \
  ylm::theta_phi(const Strahlkorper<FRAME(data)>& strahlkorper);              \
  template void ylm::theta_phi(                                               \
      const gsl::not_null<                                                    \
          tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>*>           \
          theta_phi,                                                          \
      const Strahlkorper<FRAME(data)>& strahlkorper);                         \
  template tnsr::i<DataVector, 3, FRAME(data)> ylm::rhat(                     \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&          \
          theta_phi);                                                         \
  template void ylm::rhat(                                                    \
      const gsl::not_null<tnsr::i<DataVector, 3, FRAME(data)>*> r_hat,        \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&          \
          theta_phi);                                                         \
  template ylm::Tags::aliases::Jacobian<FRAME(data)> ylm::jacobian(           \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&          \
          theta_phi);                                                         \
  template void ylm::jacobian(                                                \
      const gsl::not_null<ylm::Tags::aliases::Jacobian<FRAME(data)>*> jac,    \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&          \
          theta_phi);                                                         \
  template ylm::Tags::aliases::InvJacobian<FRAME(data)> ylm::inv_jacobian(    \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&          \
          theta_phi);                                                         \
  template void ylm::inv_jacobian(                                            \
      const gsl::not_null<ylm::Tags::aliases::InvJacobian<FRAME(data)>*>      \
          inv_jac,                                                            \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&          \
          theta_phi);                                                         \
  template ylm::Tags::aliases::InvHessian<FRAME(data)> ylm::inv_hessian(      \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&          \
          theta_phi);                                                         \
  template void ylm::inv_hessian(                                             \
      const gsl::not_null<ylm::Tags::aliases::InvHessian<FRAME(data)>*>       \
          inv_hess,                                                           \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&          \
          theta_phi);                                                         \
  template tnsr::I<DataVector, 3, FRAME(data)> ylm::cartesian_coords(         \
      const Strahlkorper<FRAME(data)>& strahlkorper,                          \
      const Scalar<DataVector>& radius,                                       \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat);                      \
  template void ylm::cartesian_coords(                                        \
      const gsl::not_null<tnsr::I<DataVector, 3, FRAME(data)>*> coords,       \
      const Strahlkorper<FRAME(data)>& strahlkorper,                          \
      const Scalar<DataVector>& radius,                                       \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat);                      \
  template tnsr::I<DataVector, 3, FRAME(data)> ylm::cartesian_coords(         \
      const Strahlkorper<FRAME(data)>& strahlkorper);                         \
  template tnsr::i<DataVector, 3, FRAME(data)>                                \
  ylm::cartesian_derivs_of_scalar(                                            \
      const Scalar<DataVector>& scalar,                                       \
      const Strahlkorper<FRAME(data)>& strahlkorper,                          \
      const Scalar<DataVector>& radius_of_strahlkorper,                       \
      const ylm::Tags::aliases::InvJacobian<FRAME(data)>& inv_jac);           \
  template void ylm::cartesian_derivs_of_scalar(                              \
      const gsl::not_null<tnsr::i<DataVector, 3, FRAME(data)>*> dx_scalar,    \
      const Scalar<DataVector>& scalar,                                       \
      const Strahlkorper<FRAME(data)>& strahlkorper,                          \
      const Scalar<DataVector>& radius_of_strahlkorper,                       \
      const ylm::Tags::aliases::InvJacobian<FRAME(data)>& inv_jac);           \
  template tnsr::ii<DataVector, 3, FRAME(data)>                               \
  ylm::cartesian_second_derivs_of_scalar(                                     \
      const Scalar<DataVector>& scalar,                                       \
      const Strahlkorper<FRAME(data)>& strahlkorper,                          \
      const Scalar<DataVector>& radius_of_strahlkorper,                       \
      const ylm::Tags::aliases::InvJacobian<FRAME(data)>& inv_jac,            \
      const ylm::Tags::aliases::InvHessian<FRAME(data)>& inv_hess);           \
  template void ylm::cartesian_second_derivs_of_scalar(                       \
      const gsl::not_null<tnsr::ii<DataVector, 3, FRAME(data)>*> result,      \
      const Scalar<DataVector>& scalar,                                       \
      const Strahlkorper<FRAME(data)>& strahlkorper,                          \
      const Scalar<DataVector>& radius_of_strahlkorper,                       \
      const ylm::Tags::aliases::InvJacobian<FRAME(data)>& inv_jac,            \
      const ylm::Tags::aliases::InvHessian<FRAME(data)>& inv_hess);           \
  template Scalar<DataVector> ylm::laplacian_of_scalar(                       \
      const Scalar<DataVector>& scalar,                                       \
      const Strahlkorper<FRAME(data)>& strahlkorper,                          \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&          \
          theta_phi);                                                         \
  template void ylm::laplacian_of_scalar(                                     \
      const gsl::not_null<Scalar<DataVector>*> result,                        \
      const Scalar<DataVector>& scalar,                                       \
      const Strahlkorper<FRAME(data)>& strahlkorper,                          \
      const tnsr::i<DataVector, 2, ::Frame::Spherical<FRAME(data)>>&          \
          theta_phi);                                                         \
  template ylm::Tags::aliases::Jacobian<FRAME(data)> ylm::tangents(           \
      const Strahlkorper<FRAME(data)>& strahlkorper,                          \
      const Scalar<DataVector>& radius,                                       \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat,                       \
      const ylm::Tags::aliases::Jacobian<FRAME(data)>& jac);                  \
  template void ylm::tangents(                                                \
      const gsl::not_null<ylm::Tags::aliases::Jacobian<FRAME(data)>*> result, \
      const Strahlkorper<FRAME(data)>& strahlkorper,                          \
      const Scalar<DataVector>& radius,                                       \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat,                       \
      const ylm::Tags::aliases::Jacobian<FRAME(data)>& jac);                  \
  template tnsr::i<DataVector, 3, FRAME(data)> ylm::normal_one_form(          \
      const tnsr::i<DataVector, 3, FRAME(data)>& dx_radius,                   \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat);                      \
  template void ylm::normal_one_form(                                         \
      const gsl::not_null<tnsr::i<DataVector, 3, FRAME(data)>*> one_form,     \
      const tnsr::i<DataVector, 3, FRAME(data)>& dx_radius,                   \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat);                      \
  template std::vector<std::array<double, 4>> ylm::fit_ylm_coeffs(            \
      const DataVector& times,                                                \
      const std::vector<Strahlkorper<FRAME(data)>>& strahlkorpers);           \
  template void ylm::time_deriv_of_strahlkorper(                              \
      const gsl::not_null<Strahlkorper<FRAME(data)>*>,                        \
      const std::deque<std::pair<double, Strahlkorper<FRAME(data)>>>&);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Distorted, Frame::Grid, Frame::Inertial))

#undef INSTANTIATE
#undef FRAME
}  // namespace ylm
