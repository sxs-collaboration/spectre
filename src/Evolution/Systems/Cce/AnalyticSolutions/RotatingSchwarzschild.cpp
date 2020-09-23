// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"

#include <complex>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::Solutions {

/// \cond
RotatingSchwarzschild::RotatingSchwarzschild(const double extraction_radius,
                                             const double mass,
                                             const double frequency) noexcept
    : SphericalMetricData{extraction_radius},
      frequency_{frequency},
      mass_{mass} {}

std::unique_ptr<WorldtubeData> RotatingSchwarzschild::get_clone() const
    noexcept {
  return std::make_unique<RotatingSchwarzschild>(*this);
}

void RotatingSchwarzschild::spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        spherical_metric,
    const size_t l_max, double /*time*/) const noexcept {
  get<0, 1>(*spherical_metric) = 0.0;
  get<0, 2>(*spherical_metric) = 0.0;

  get<1, 1>(*spherical_metric) = 1.0 / (1.0 - 2.0 * mass_ / extraction_radius_);
  get<1, 2>(*spherical_metric) = 0.0;
  get<1, 3>(*spherical_metric) = 0.0;

  get<2, 2>(*spherical_metric) = square(extraction_radius_);
  get<2, 3>(*spherical_metric) = 0.0;

  // note: omit the sin factors for the phi components due to pfaffian
  // Jacobian factors
  get<3, 3>(*spherical_metric) = square(extraction_radius_);

  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get<0, 0>(*spherical_metric)[collocation_point.offset] =
        -(1.0 - 2.0 * mass_ / extraction_radius_ -
          square(frequency_) * square(extraction_radius_) *
              square(sin(collocation_point.theta)));
    get<0, 3>(*spherical_metric)[collocation_point.offset] =
        square(extraction_radius_) * frequency_ * sin(collocation_point.theta);
  }
}

void RotatingSchwarzschild::dr_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dr_spherical_metric,
    const size_t l_max, double /*time*/) const noexcept {
  get<0, 1>(*dr_spherical_metric) = 0.0;
  get<0, 2>(*dr_spherical_metric) = 0.0;

  get<1, 1>(*dr_spherical_metric) =
      -2.0 * mass_ / (square(extraction_radius_ - 2.0 * mass_));
  get<1, 2>(*dr_spherical_metric) = 0.0;
  get<1, 3>(*dr_spherical_metric) = 0.0;

  get<2, 2>(*dr_spherical_metric) = 2.0 * extraction_radius_;
  get<2, 3>(*dr_spherical_metric) = 0.0;

  // note: omit the sin factors for the phi components due to pfaffian
  // Jacobian factors
  get<3, 3>(*dr_spherical_metric) = 2.0 * extraction_radius_;

  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get<0, 0>(*dr_spherical_metric)[collocation_point.offset] =
        -(2.0 * mass_ / square(extraction_radius_) -
          2.0 * square(frequency_) * extraction_radius_ *
              square(sin(collocation_point.theta)));
    get<0, 3>(*dr_spherical_metric)[collocation_point.offset] =
        2.0 * extraction_radius_ * frequency_ * sin(collocation_point.theta);
  }
}

void RotatingSchwarzschild::dt_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dt_spherical_metric,
    size_t /*l_max*/, double /*time*/) const noexcept {
  for(auto& component : *dt_spherical_metric) {
    component = 0.0;
  }
}

void RotatingSchwarzschild::variables_impl(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
    size_t /*output_l_max*/, double /*time*/,
    tmpl::type_<Tags::News> /*meta*/) const noexcept {
  get(*news).data() = 0.0;
}

PUP::able::PUP_ID RotatingSchwarzschild::my_PUP_ID = 0;
/// \endcond
}  // namespace Cce::Solutions
