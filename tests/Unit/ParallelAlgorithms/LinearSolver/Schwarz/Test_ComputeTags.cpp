// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ComputeTags.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"

namespace {
struct DummyOptionsGroup {};
}  // namespace

namespace LinearSolver::Schwarz {

SPECTRE_TEST_CASE("Unit.ParallelSchwarz.ComputeTags",
                  "[Unit][ParallelAlgorithms][LinearSolver]") {
  {
    TestHelpers::db::test_compute_tag<
        Tags::IntrudingExtentsCompute<2, DummyOptionsGroup>>(
        "IntrudingExtents(DummyOptionsGroup)");
    const auto box =
        db::create<db::AddSimpleTags<domain::Tags::Mesh<2>,
                                     Tags::MaxOverlap<DummyOptionsGroup>>,
                   db::AddComputeTags<
                       Tags::IntrudingExtentsCompute<2, DummyOptionsGroup>>>(
            Mesh<2>{{{3, 2}},
                    Spectral::Basis::Legendre,
                    Spectral::Quadrature::GaussLobatto},
            size_t{2});
    CHECK(get<Tags::IntrudingExtents<2, DummyOptionsGroup>>(box) ==
          std::array<size_t, 2>{{2, 1}});
  }
  {
    TestHelpers::db::test_compute_tag<
        Tags::IntrudingOverlapWidthsCompute<2, DummyOptionsGroup>>(
        "IntrudingOverlapWidths(DummyOptionsGroup)");
    const auto box = db::create<
        db::AddSimpleTags<domain::Tags::Mesh<2>,
                          Tags::IntrudingExtents<2, DummyOptionsGroup>>,
        db::AddComputeTags<
            Tags::IntrudingOverlapWidthsCompute<2, DummyOptionsGroup>>>(
        Mesh<2>{{{3, 2}},
                Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto},
        std::array<size_t, 2>{{2, 1}});
    CHECK(get<Tags::IntrudingOverlapWidths<2, DummyOptionsGroup>>(box) ==
          std::array<double, 2>{{2., 2.}});
  }
  {
    TestHelpers::db::test_compute_tag<
        Tags::ElementWeightCompute<1, DummyOptionsGroup>>(
        "Weight(DummyOptionsGroup)");
    const auto box = db::create<
        db::AddSimpleTags<domain::Tags::Element<1>,
                          domain::Tags::Coordinates<1, Frame::Logical>,
                          Tags::IntrudingOverlapWidths<1, DummyOptionsGroup>,
                          Tags::MaxOverlap<DummyOptionsGroup>>,
        db::AddComputeTags<Tags::ElementWeightCompute<1, DummyOptionsGroup>>>(
        Element<1>{ElementId<1>{0},
                   {{Direction<1>::lower_xi(), {{ElementId<1>{1}}, {}}},
                    {Direction<1>::upper_xi(), {{ElementId<1>{2}}, {}}}}},
        tnsr::I<DataVector, 1, Frame::Logical>{{{{-1., 0., 1.}}}},
        std::array<double, 1>{{1.}}, size_t{1});
    const DataVector expected_weights{0.5, 1., 0.5};
    CHECK_ITERABLE_APPROX(get(get<Tags::Weight<DummyOptionsGroup>>(box)),
                          expected_weights);
  }
  {
    TestHelpers::db::test_compute_tag<
        Tags::IntrudingOverlapWeightCompute<1, DummyOptionsGroup>>(
        "Weight(DummyOptionsGroup)");
    const auto box = db::create<
        db::AddSimpleTags<domain::Tags::Mesh<1>, domain::Tags::Element<1>,
                          Tags::IntrudingExtents<1, DummyOptionsGroup>,
                          domain::Tags::Coordinates<1, Frame::Logical>,
                          Tags::IntrudingOverlapWidths<1, DummyOptionsGroup>>,
        db::AddComputeTags<
            domain::Tags::InternalDirectionsCompute<1>,
            domain::Tags::InterfaceCompute<domain::Tags::InternalDirections<1>,
                                           domain::Tags::Direction<1>>,
            domain::Tags::InterfaceCompute<
                domain::Tags::InternalDirections<1>,
                Tags::IntrudingOverlapWeightCompute<1, DummyOptionsGroup>>>>(
        Mesh<1>{3, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto},
        Element<1>{ElementId<1>{0},
                   {{Direction<1>::lower_xi(), {{ElementId<1>{1}}, {}}},
                    {Direction<1>::upper_xi(), {{ElementId<1>{2}}, {}}}}},
        std::array<size_t, 1>{{1}},
        tnsr::I<DataVector, 1, Frame::Logical>{{{{-1., 0., 1.}}}},
        std::array<double, 1>{{1.}});
    const DataVector expected_weights{0.5};
    CHECK_ITERABLE_APPROX(
        get(get<domain::Tags::Interface<domain::Tags::InternalDirections<1>,
                                        Tags::Weight<DummyOptionsGroup>>>(box)
                .at(Direction<1>::lower_xi())),
        expected_weights);
  }
  {
    TestHelpers::db::test_compute_tag<
        Tags::SummedIntrudingOverlapWeightsCompute<1, DummyOptionsGroup>>(
        "SummedIntrudingOverlapWeights(DummyOptionsGroup)");
    const auto box = db::create<
        db::AddSimpleTags<
            domain::Tags::Interface<domain::Tags::InternalDirections<1>,
                                    Tags::Weight<DummyOptionsGroup>>,
            domain::Tags::Mesh<1>,
            Tags::IntrudingExtents<1, DummyOptionsGroup>>,
        db::AddComputeTags<
            Tags::SummedIntrudingOverlapWeightsCompute<1, DummyOptionsGroup>>>(
        std::unordered_map<Direction<1>, Scalar<DataVector>>{
            {Direction<1>::lower_xi(),
             Scalar<DataVector>{DataVector{0.5, 0.1}}},
            {Direction<1>::upper_xi(),
             Scalar<DataVector>{DataVector{0.1, 0.5}}}},
        Mesh<1>{3, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto},
        std::array<size_t, 1>{{2}});
    const DataVector expected_weights{0.5, 0.2, 0.5};
    CHECK_ITERABLE_APPROX(
        get(get<Tags::SummedIntrudingOverlapWeights<DummyOptionsGroup>>(box)),
        expected_weights);
  }
}

}  // namespace LinearSolver::Schwarz
