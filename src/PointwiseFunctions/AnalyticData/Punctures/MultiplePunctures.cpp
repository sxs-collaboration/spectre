// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Punctures/MultiplePunctures.hpp"

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Punctures/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Punctures::AnalyticData {

bool operator==(const Puncture& lhs, const Puncture& rhs) {
  return lhs.position == rhs.position and lhs.mass == rhs.mass and
         lhs.dimensionless_momentum == rhs.dimensionless_momentum and
         lhs.dimensionless_spin == rhs.dimensionless_spin;
}

bool operator!=(const Puncture& lhs, const Puncture& rhs) {
  return not(lhs == rhs);
}

namespace detail {

void MultiplePuncturesVariables::operator()(
    const gsl::not_null<Scalar<DataVector>*> one_over_alpha,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::OneOverAlpha /*meta*/) const {
  const size_t num_points = x.begin()->size();
  tnsr::I<DataVector, 3> x_centered{num_points};
  DataVector r{num_points};
  get(*one_over_alpha) = 0.;
  for (const auto& puncture : punctures) {
    x_centered = x;
    get<0>(x_centered) -= puncture.position[0];
    get<1>(x_centered) -= puncture.position[1];
    get<2>(x_centered) -= puncture.position[2];
    r = get(magnitude(x_centered));
    // Emit an error when a grid point coincides with a puncture. This is an
    // error and not an assert because it's a user error and should probably be
    // fixed in the input file.
    for (size_t i = 0; i < r.size(); ++i) {
      if (equal_within_roundoff(r[i], 0.)) {
        ERROR(
            "A grid point coincides with a puncture. Please reposition the "
            "grid or the puncture. Position: "
            << puncture.position);
      }
    }
    get(*one_over_alpha) += 0.5 * puncture.mass / r;
  }
}

void MultiplePuncturesVariables::operator()(
    const gsl::not_null<Scalar<DataVector>*> alpha,
    const gsl::not_null<Cache*> cache, Punctures::Tags::Alpha /*meta*/) const {
  const auto& one_over_alpha =
      cache->get_var(*this, detail::Tags::OneOverAlpha{});
  get(*alpha) = 1. / get(one_over_alpha);
}

void MultiplePuncturesVariables::operator()(
    const gsl::not_null<tnsr::II<DataVector, 3>*>
        traceless_conformal_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    Punctures::Tags::TracelessConformalExtrinsicCurvature /*meta*/) const {
  const size_t num_points = x.begin()->size();
  tnsr::I<DataVector, 3> x_centered{num_points};
  DataVector r{num_points};
  tnsr::I<DataVector, 3> n{num_points};
  DataVector n_dot_P{num_points};
  tnsr::I<DataVector, 3> S_cross_n{num_points};
  std::fill(traceless_conformal_extrinsic_curvature->begin(),
            traceless_conformal_extrinsic_curvature->end(), 0.);
  for (const auto& puncture : punctures) {
    x_centered = x;
    get<0>(x_centered) -= puncture.position[0];
    get<1>(x_centered) -= puncture.position[1];
    get<2>(x_centered) -= puncture.position[2];
    r = get(magnitude(x_centered));
    n = x_centered;
    get<0>(n) /= r;
    get<1>(n) /= r;
    get<2>(n) /= r;
    n_dot_P = 0.;
    for (size_t d = 0; d < 3; ++d) {
      n_dot_P += n.get(d) * gsl::at(puncture.dimensionless_momentum, d);
    }
    std::fill(S_cross_n.begin(), S_cross_n.end(), 0.);
    for (LeviCivitaIterator<3> levi_civita_it; levi_civita_it;
         ++levi_civita_it) {
      const auto [i, j, k] = levi_civita_it();
      S_cross_n.get(i) += levi_civita_it.sign() *
                          gsl::at(puncture.dimensionless_spin, j) * n.get(k);
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        // Momentum
        traceless_conformal_extrinsic_curvature->get(i, j) +=
            1.5 / square(r) * puncture.mass *
            (gsl::at(puncture.dimensionless_momentum, i) * n.get(j) +
             gsl::at(puncture.dimensionless_momentum, j) * n.get(i) +
             n.get(i) * n.get(j) * n_dot_P);
        // Spin
        traceless_conformal_extrinsic_curvature->get(i, j) +=
            3. / cube(r) * square(puncture.mass) *
            (n.get(i) * S_cross_n.get(j) + n.get(j) * S_cross_n.get(i));
      }
      // Diagonal momentum
      traceless_conformal_extrinsic_curvature->get(i, i) -=
          1.5 / square(r) * puncture.mass * n_dot_P;
    }
  }
}

void MultiplePuncturesVariables::operator()(
    const gsl::not_null<Scalar<DataVector>*> beta,
    const gsl::not_null<Cache*> cache, Punctures::Tags::Beta /*meta*/) const {
  get(*beta) = 0.;
  const auto& A = cache->get_var(
      *this, Punctures::Tags::TracelessConformalExtrinsicCurvature{});
  for (size_t i = 0; i < A.size(); ++i) {
    get(*beta) += A.multiplicity(i) * square(A[i]);
  }
  const auto& alpha = cache->get_var(*this, Punctures::Tags::Alpha{});
  get(*beta) *= 0.125 * pow<7>(get(alpha));
}

void MultiplePuncturesVariables::operator()(
    const gsl::not_null<Scalar<DataVector>*> fixed_source_for_field,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Punctures::Tags::Field> /*meta*/) const {
  // Initial guess
  get(*fixed_source_for_field) = 0.;
}

}  // namespace detail

PUP::able::PUP_ID MultiplePunctures::my_PUP_ID = 0;  // NOLINT

}  // namespace Punctures::AnalyticData
