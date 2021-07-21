// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Freezing.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
using frame = Frame::Inertial;

template <size_t Dim>
void test() {
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);

  for (const std::string& bc_string : {"Freezing"}) {
    CAPTURE(bc_string);

    helpers::test_boundary_condition_with_python<
        GeneralizedHarmonic::BoundaryConditions::FreezingBjorhus<Dim>,
        GeneralizedHarmonic::BoundaryConditions::BoundaryCondition<Dim>,
        GeneralizedHarmonic::System<Dim>,
        tmpl::list<
            GeneralizedHarmonic::BoundaryCorrections::UpwindPenalty<Dim>>>(
        make_not_null(&gen),
        "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Freezing",
        tuples::TaggedTuple<
            helpers::Tags::PythonFunctionForErrorMessage<>,
            helpers::Tags::PythonFunctionName<
                ::Tags::dt<gr::Tags::SpacetimeMetric<Dim, frame, DataVector>>>,
            helpers::Tags::PythonFunctionName<
                ::Tags::dt<GeneralizedHarmonic::Tags::Pi<Dim, frame>>>,
            helpers::Tags::PythonFunctionName<
                ::Tags::dt<GeneralizedHarmonic::Tags::Phi<Dim, frame>>>>{
            "error", "dt_spacetime_metric_" + bc_string, "dt_pi_" + bc_string,
            "dt_phi_" + bc_string},
        "FreezingBjorhus:\n", Index<Dim - 1>{Dim == 1 ? 1 : 5},
        db::DataBox<tmpl::list<>>{},
        tuples::TaggedTuple<
            helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
            helpers::Tags::Range<gr::Tags::Shift<Dim, frame, DataVector>>>{
            std::array<double, 2>{{0.8, 1.}}, std::array<double, 2>{{0.1, 0.2}}}
        //,
        // 1.e-6
    );
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.BCFreezing",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test<1>();
  test<2>();
  test<3>();
}
