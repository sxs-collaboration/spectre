// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/TeukolskyWave.hpp"
#include "Evolution/Systems/Cce/BoundaryDataTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/AnalyticSolutions/AnalyticDataHelpers.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::Solutions {

namespace {

struct TeukolskyWaveWrapper : public TeukolskyWave {
  using taglist =
      tmpl::list<gr::Tags::SpacetimeMetric<
                     3, ::Frame::Spherical<::Frame::Inertial>, DataVector>,
                 Tags::Dr<gr::Tags::SpacetimeMetric<
                     3, ::Frame::Spherical<::Frame::Inertial>, DataVector>>,
                 ::Tags::dt<gr::Tags::SpacetimeMetric<
                     3, ::Frame::Spherical<::Frame::Inertial>, DataVector>>,
                 Tags::News>;

  using TeukolskyWave::TeukolskyWave;

  void test_spherical_metric(const size_t l_max, const double time) noexcept {
    const size_t size =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    Scalar<DataVector> sin_theta{size};
    Scalar<DataVector> cos_theta{size};
    const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
        Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
    for (const auto& collocation_point : collocation) {
      get(sin_theta)[collocation_point.offset] = sin(collocation_point.theta);
      get(cos_theta)[collocation_point.offset] = cos(collocation_point.theta);
    }
    tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>
        local_spherical_metric{size};
    tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>
        local_dr_spherical_metric{size};
    tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>
        local_dt_spherical_metric{size};
    Scalar<SpinWeighted<ComplexDataVector, -2>> local_news{size};

    this->spherical_metric(make_not_null(&local_spherical_metric), l_max, time);
    this->dr_spherical_metric(make_not_null(&local_dr_spherical_metric), l_max,
                              time);
    this->dt_spherical_metric(make_not_null(&local_dt_spherical_metric), l_max,
                              time);
    this->variables_impl(make_not_null(&local_news), l_max, time,
                         tmpl::type_<Tags::News>{});

    // Pypp call expects all of the objects to be the same category -- here we
    // need to use tensors, so we must pack up the `double` arguments into
    // tensors.
    Scalar<DataVector> time_vector;
    get(time_vector) = DataVector{size, time};
    Scalar<DataVector> radius_vector;
    get(radius_vector) = DataVector{size, extraction_radius_};
    Scalar<DataVector> amplitude_vector;
    get(amplitude_vector) = DataVector{size, amplitude_};
    Scalar<DataVector> duration_vector;
    get(duration_vector) = DataVector{size, duration_};

    const auto py_spherical_metric = pypp::call<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>>(
        "TeukolskyWave", "spherical_metric", sin_theta, cos_theta, time_vector,
        radius_vector, amplitude_vector, duration_vector);
    const auto py_dt_spherical_metric = pypp::call<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>>(
        "TeukolskyWave", "dt_spherical_metric", sin_theta, cos_theta,
        time_vector, radius_vector, amplitude_vector, duration_vector);
    const auto py_dr_spherical_metric = pypp::call<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>>(
        "TeukolskyWave", "dr_spherical_metric", sin_theta, cos_theta,
        time_vector, radius_vector, amplitude_vector, duration_vector);
    const auto py_news = pypp::call<Scalar<SpinWeighted<ComplexDataVector, 2>>>(
        "TeukolskyWave", "news", sin_theta, time_vector, radius_vector,
        amplitude_vector, duration_vector);

    for (size_t a = 0; a < 4; ++a) {
      for (size_t b = a; b < 4; ++b) {
        CAPTURE(a);
        CAPTURE(b);
        const auto& lhs = local_spherical_metric.get(a, b);
        const auto& rhs = py_spherical_metric.get(a, b);
        CHECK_ITERABLE_APPROX(lhs, rhs);
        const auto& dr_lhs = local_dr_spherical_metric.get(a, b);
        const auto& dr_rhs = py_dr_spherical_metric.get(a, b);
        CHECK_ITERABLE_APPROX(dr_lhs, dr_rhs);
        const auto& dt_lhs = local_dt_spherical_metric.get(a, b);
        const auto& dt_rhs = py_dt_spherical_metric.get(a, b);
        CHECK_ITERABLE_APPROX(dt_lhs, dt_rhs);
      }
    }
  }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.TeukolskyWave", "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> radius_dist{5.0, 10.0};
  UniformCustomDistribution<double> parameter_dist{0.5, 2.0};
  const double extraction_radius = radius_dist(gen);
  const size_t l_max = 16;
  // use a low frequency so that the dimensionless parameter in the metric is ~1
  const double amplitude = parameter_dist(gen);
  const double duration = 1.0 + parameter_dist(gen);
  TeukolskyWaveWrapper boundary_solution{extraction_radius, amplitude,
                                         duration};
  const double time = duration + extraction_radius + parameter_dist(gen);
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/Cce/AnalyticSolutions/"};
  boundary_solution.test_spherical_metric(l_max, time);
}
}  // namespace Cce::Solutions
