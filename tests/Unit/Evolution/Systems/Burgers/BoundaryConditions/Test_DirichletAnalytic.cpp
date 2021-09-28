// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Burgers/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/Burgers/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/AnalyticData/Burgers/Sinusoid.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Step.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
struct ConvertSinusoid {
  using unpacked_container = double;
  using packed_container = Burgers::AnalyticData::Sinusoid;
  using packed_type = double;

  static inline unpacked_container unpack(const packed_container /*packed*/,
                                          const size_t /*grid_point_index*/) {
    return 0.0;  // no parameter but we need some placeholder type
  }

  static inline void pack(const gsl::not_null<packed_container*> /*packed*/,
                          const unpacked_container /*unpacked*/,
                          const size_t /*grid_point_index*/) {
    // no parameters but we need a placeholder function
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

struct ConvertStep {
  using unpacked_container = std::array<double, 3>;
  using packed_container = Burgers::Solutions::Step;
  using packed_type = double;

  static packed_container create_container() { return {2.0, 1.0, 0.5}; }

  static inline unpacked_container unpack(const packed_container /*packed*/,
                                          const size_t /*grid_point_index*/) {
    // No way of getting the args from the boundary condition.
    return {{2.0, 1.0, 0.5}};
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container unpacked,
                          const size_t /*grid_point_index*/) {
    *packed = packed_container{unpacked[0], unpacked[1], unpacked[2]};
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Burgers.BoundaryConditions.DirichletAnalytic",
                  "[Unit][Burgers]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/Burgers/BoundaryConditions/"};
  MAKE_GENERATOR(gen);
  const auto box_analytic_data = db::create<db::AddSimpleTags<
      Tags::Time, Tags::AnalyticData<Burgers::AnalyticData::Sinusoid>>>(
      0.4, Burgers::AnalyticData::Sinusoid{});

  helpers::test_boundary_condition_with_python<
      Burgers::BoundaryConditions::DirichletAnalytic,
      Burgers::BoundaryConditions::BoundaryCondition, Burgers::System,
      tmpl::list<Burgers::BoundaryCorrections::Rusanov>,
      tmpl::list<ConvertSinusoid>>(
      make_not_null(&gen), "DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<Burgers::Tags::U>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>>{
          "error_sinusoid", "u_sinusoid", "flux_sinusoid"},
      "DirichletAnalytic:\n", Index<0>{1}, box_analytic_data,
      tuples::TaggedTuple<>{});

  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time, Tags::AnalyticSolution<Burgers::Solutions::Step>>>(
      0.5, ConvertStep::create_container());

  helpers::test_boundary_condition_with_python<
      Burgers::BoundaryConditions::DirichletAnalytic,
      Burgers::BoundaryConditions::BoundaryCondition, Burgers::System,
      tmpl::list<Burgers::BoundaryCorrections::Rusanov>,
      tmpl::list<ConvertStep>>(
      make_not_null(&gen), "DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<Burgers::Tags::U>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>>{
          "error_step", "u_step", "flux_step"},
      "DirichletAnalytic:\n", Index<0>{1}, box_analytic_soln,
      tuples::TaggedTuple<>{});
}
