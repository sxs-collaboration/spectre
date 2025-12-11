// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceConstrainedEvolution.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::fd {
namespace {

void test(bool constrained_evolution) {
  constexpr size_t dim = System::volume_dim;
  const size_t num_pts = 5;
  auto conformal_spatial_metric =
      make_with_value<tnsr::ii<DataVector, dim>>(num_pts, 0.);
  auto a_tilde = make_with_value<tnsr::ii<DataVector, dim>>(num_pts, 0.);

  for (size_t i = 0; i < dim; ++i) {
    conformal_spatial_metric.get(i, i) = DataVector{num_pts, 2.};
    a_tilde.get(i, i) = DataVector{num_pts, 1.};
  }

  auto box = db::create<
      db::AddSimpleTags<::Ccz4::fd::Tags::ConstrainedEvolution,
                        ::Ccz4::Tags::ConformalMetric<DataVector, dim>,
                        ::Ccz4::Tags::ATilde<DataVector, dim>>>(
      constrained_evolution, conformal_spatial_metric, a_tilde);

  for (size_t i = 0; i < dim; ++i) {
    for (size_t j = i; j < dim; ++j) {
      if (i == j) {
        CHECK_ITERABLE_APPROX(
            (get<::Ccz4::Tags::ConformalMetric<DataVector, dim>>(box))
                .get(i, j),
            (DataVector{num_pts, 2.}));
        CHECK_ITERABLE_APPROX(
            (get<::Ccz4::Tags::ATilde<DataVector, dim>>(box)).get(i, j),
            (DataVector{num_pts, 1.}));
      } else {
        CHECK_ITERABLE_APPROX(
            (get<::Ccz4::Tags::ConformalMetric<DataVector, dim>>(box))
                .get(i, j),
            (DataVector{num_pts, 0.}));
        CHECK_ITERABLE_APPROX(
            (get<::Ccz4::Tags::ATilde<DataVector, dim>>(box)).get(i, j),
            (DataVector{num_pts, 0.}));
      }
    }
  }

  db::mutate_apply<EnforceConstrainedEvolution>(make_not_null(&box));

  for (size_t i = 0; i < dim; ++i) {
    if (get<::Ccz4::fd::Tags::ConstrainedEvolution>(box)) {
      conformal_spatial_metric.get(i, i) = DataVector{num_pts, 1.};
      a_tilde.get(i, i) = DataVector{num_pts, 0.};
    } else {
      conformal_spatial_metric.get(i, i) = DataVector{num_pts, 2.};
      a_tilde.get(i, i) = DataVector{num_pts, 1.};
    }
  }
  for (size_t i = 0; i < dim; ++i) {
    for (size_t j = i; j < dim; ++j) {
      CHECK_ITERABLE_APPROX(
          (get<::Ccz4::Tags::ConformalMetric<DataVector, dim>>(box)).get(i, j),
          conformal_spatial_metric.get(i, j));
      CHECK_ITERABLE_APPROX(
          (get<::Ccz4::Tags::ATilde<DataVector, dim>>(box)).get(i, j),
          a_tilde.get(i, j));
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.ECE", "[Unit][Evolution]") {
  test(true);
  test(false);
}
}  // namespace
}  // namespace Ccz4::fd
