// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/Side.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Weighting.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace LinearSolver::Schwarz {

namespace {

// This function explicitly implements Eqn. (41) in
// https://arxiv.org/abs/1907.01572
template <size_t Dim>
void test_element_weight() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist_coords(-2., 2.);
  std::uniform_real_distribution<> dist_widths(0.1, 2.);
  const auto logical_coords =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::Logical>>(
          make_not_null(&generator), make_not_null(&dist_coords), size_t{5});
  const auto widths = make_with_random_values<std::array<double, Dim>>(
      make_not_null(&generator), make_not_null(&dist_widths));
  CAPTURE(Dim);
  CAPTURE(logical_coords);
  CAPTURE(widths);
  const auto phi = [](const DataVector& xi) {
    DataVector result{xi.size()};
    for (size_t i = 0; i < xi.size(); ++i) {
      if (xi[i] < -1.) {
        result[i] = -1.;
      } else if (xi[i] > 1.) {
        result[i] = 1.;
      } else {
        result[i] =
            0.125 * (15. * xi[i] - 10. * cube(xi[i]) + 3. * pow<5>(xi[i]));
      }
    }
    return result;
  };
  Scalar<DataVector> reference_element_weight{logical_coords.begin()->size(),
                                              1.};
  for (size_t d = 0; d < Dim; ++d) {
    get(reference_element_weight) *=
        0.5 * (phi((logical_coords.get(d) + 1.) / gsl::at(widths, d)) -
               phi((logical_coords.get(d) - 1.) / gsl::at(widths, d)));
  }
  CHECK_ITERABLE_APPROX(
      get(LinearSolver::Schwarz::element_weight(logical_coords, widths, {})),
      get(reference_element_weight));
}

template <size_t Dim>
void test_weights_conservation(const Element<Dim>& element,
                               const Mesh<Dim>& mesh,
                               const size_t max_overlap) noexcept {
  INFO("Weight conservation");
  CAPTURE(Dim);
  CAPTURE(element);
  CAPTURE(mesh);
  CAPTURE(max_overlap);
  const auto logical_coords = logical_coordinates(mesh);
  CAPTURE(logical_coords);
  const size_t num_points = mesh.number_of_grid_points();
  std::array<double, Dim> overlap_widths{};
  for (size_t d = 0; d < Dim; ++d) {
    const auto overlap_extent =
        LinearSolver::Schwarz::overlap_extent(mesh.extents(d), max_overlap);
    const auto collocation_points =
        Spectral::collocation_points(mesh.slice_through(d));
    gsl::at(overlap_widths, d) = LinearSolver::Schwarz::overlap_width(
        overlap_extent, collocation_points);
  }
  CAPTURE(overlap_widths);
  // The weights for the element data and those for data on intruding overlaps
  // should sum to one.
  const auto element_weights = LinearSolver::Schwarz::element_weight(
      logical_coords, overlap_widths, element.external_boundaries());
  CAPTURE(element_weights);
  auto summed_weights = element_weights;
  for (const auto& direction_and_neighbors : element.neighbors()) {
    const auto& direction = direction_and_neighbors.first;
    const size_t num_neighbors = direction_and_neighbors.second.size();
    get(summed_weights) +=
        num_neighbors * get(LinearSolver::Schwarz::intruding_weight(
                            logical_coords, direction, overlap_widths,
                            num_neighbors, element.external_boundaries()));
  }
  CHECK_ITERABLE_APPROX(get(summed_weights), DataVector(num_points, 1.));
}

template <size_t Dim>
void test_weights(const Mesh<Dim>& mesh, const size_t max_overlap) noexcept {
  typename Element<Dim>::Neighbors_t neighbors{};
  for (size_t d = 0; d < Dim; ++d) {
    for (const auto& side : std::array<Side, 2>{{Side::Lower, Side::Upper}}) {
      auto& direction_neighbors =
          neighbors.emplace(Direction<Dim>{d, side}, Neighbors<Dim>{})
              .first->second;
      for (size_t i = 0; i < two_to_the(Dim - 1); ++i) {
        direction_neighbors.add_ids({ElementId<Dim>{i + 1}});
        const Element<Dim> element{ElementId<Dim>{0}, neighbors};
        for (size_t max_overlap_i = 1; max_overlap_i <= max_overlap;
             ++max_overlap_i) {
          test_weights_conservation(element, mesh, max_overlap_i);
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelSchwarz.Weighting",
                  "[Unit][ParallelAlgorithms][LinearSolver]") {
  {
    INFO("Weighting function");
    // Extruding weight on upper side:
    // ___
    //    \       (disabling backslash line continuation)
    //     --
    // ---+--> xi
    //    1.
    CHECK_ITERABLE_APPROX(
        extruding_weight(DataVector({-1., 0., 1., 2., 3.}), 1., Side::Upper),
        DataVector({1., 1., 0.5, 0., 0.}));
    // Intruding weight on upper side:
    //      __
    //    /
    // ---
    // ---+---> xi
    //    1.
    CHECK_ITERABLE_APPROX(
        intruding_weight(DataVector({-1., 0., 1., 2., 3.}), 1., Side::Upper),
        DataVector({0., 0., 0.5, 1., 1.}));
    // Extruding weight on lower side:
    //     ___
    //   /
    // --
    // --+----> xi
    //  -1.
    CHECK_ITERABLE_APPROX(
        extruding_weight(DataVector({-3., -2., -1., 0., 1.}), 1., Side::Lower),
        DataVector({0., 0., 0.5, 1., 1.}));
    // Intruding weight on lower side:
    // __
    //   \        (disabling backslash line continuation)
    //    ---
    // --+---> xi
    //  -1.
    CHECK_ITERABLE_APPROX(
        intruding_weight(DataVector({-3., -2., -1., 0., 1.}), 1., Side::Lower),
        DataVector({1., 1., 0.5, 0., 0.}));
  }
  {
    const auto element_weight = LinearSolver::Schwarz::element_weight(
        tnsr::I<DataVector, 1, Frame::Logical>{
            {{{-2., -1.5, -1., -0.5, -0.25, 0., 0.25, 0.5, 1., 1.5, 2.}}}},
        {{1.}}, {});
    CHECK_ITERABLE_APPROX(
        get(element_weight),
        DataVector({0., 0.103515625, 0.5, 0.896484375, 0.98394775390625, 1.,
                    0.98394775390625, 0.896484375, 0.5, 0.103515625, 0.}));
  }
  // Test against an explicit reference implementation
  {
    test_element_weight<1>();
    test_element_weight<2>();
    test_element_weight<3>();
  }
  // Test weights conservation on all possible element-neighbor configurations
  {
    test_weights(Mesh<1>{3, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto},
                 4);
    test_weights(Mesh<2>{{3, 4},
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto},
                 4);
    test_weights(Mesh<3>{{2, 3, 4},
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto},
                 4);
  }
}

}  // namespace LinearSolver::Schwarz
