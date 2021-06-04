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
