// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Hydro/SmoothFlow.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

template <size_t Dim, bool IsRelativistic>
struct SmoothFlowProxy : hydro::Solutions::SmoothFlow<Dim, IsRelativistic> {
 public:
  SmoothFlowProxy() = default;
  SmoothFlowProxy(const SmoothFlowProxy& /*rhs*/) = delete;
  SmoothFlowProxy& operator=(const SmoothFlowProxy& /*rhs*/) = delete;
  SmoothFlowProxy(SmoothFlowProxy&& /*rhs*/) noexcept = default;
  SmoothFlowProxy& operator=(SmoothFlowProxy&& /*rhs*/) noexcept = default;
  ~SmoothFlowProxy() = default;

  SmoothFlowProxy(const std::array<double, Dim>& mean_velocity,
                  const std::array<double, Dim>& wavevector,
                  const double pressure, const double adiabatic_index,
                  const double perturbation_size) noexcept
      : hydro::Solutions::SmoothFlow<Dim, IsRelativistic>(
            mean_velocity, wavevector, pressure, adiabatic_index,
            perturbation_size) {}

  template <typename DataType>
  using core_variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, Dim>,
                 hydro::Tags::SpecificInternalEnergy<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  using variables_tags =
      tmpl::conditional_t<IsRelativistic,
                          tmpl::push_back<core_variables_tags<DataType>,
                                          hydro::Tags::LorentzFactor<DataType>>,
                          core_variables_tags<DataType>>;

  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, Dim>& x, const double t,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {
        get<Tags>(hydro::Solutions::SmoothFlow<Dim, IsRelativistic>::variables(
            x, t, tmpl::list<Tags>{}))...};
  }

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>>
  primitive_variables(const tnsr::I<DataType, Dim>& x,
                      const double t) const noexcept {
    return this->variables(x, t, variables_tags<DataType>{});
  }
};

template <size_t Dim, bool IsRelativistic, typename DataType>
void test_solution(const DataType& used_for_size,
                   const std::array<double, Dim>& mean_velocity,
                   const std::array<double, Dim>& wave_vector) {
  CAPTURE(Dim);
  CAPTURE(IsRelativistic);
  const double pressure = 1.23;
  const double adiabatic_index = 1.3334;
  const double perturbation_size = 0.78;

  SmoothFlowProxy<Dim, IsRelativistic> solution(
      mean_velocity, wave_vector, pressure, adiabatic_index, perturbation_size);
  if constexpr (IsRelativistic) {
    pypp::check_with_random_values<1>(
        &SmoothFlowProxy<Dim, IsRelativistic>::template primitive_variables<
            DataType>,
        solution, "SmoothFlow",
        {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
         "pressure", "specific_enthalpy_relativistic", "lorentz_factor"},
        {{{-15., 15.}}},
        std::make_tuple(mean_velocity, wave_vector, pressure, adiabatic_index,
                        perturbation_size),
        used_for_size);
  } else {
    pypp::check_with_random_values<1>(
        &SmoothFlowProxy<Dim, IsRelativistic>::template primitive_variables<
            DataType>,
        solution, "SmoothFlow",
        {"rest_mass_density", "spatial_velocity", "specific_internal_energy",
         "pressure", "specific_enthalpy"},
        {{{-15., 15.}}},
        std::make_tuple(mean_velocity, wave_vector, pressure, adiabatic_index,
                        perturbation_size),
        used_for_size);
  }

  test_serialization(solution);
}

template <bool IsRelativistic>
void test() {
  test_solution<1, IsRelativistic>(std::numeric_limits<double>::signaling_NaN(),
                                   {{-0.3}}, {{0.4}});
  test_solution<1, IsRelativistic>(DataVector(5), {{-0.3}}, {{0.4}});
  test_solution<2, IsRelativistic>(std::numeric_limits<double>::signaling_NaN(),
                                   {{-0.3, 0.1}}, {{0.4, -0.24}});
  test_solution<2, IsRelativistic>(DataVector(5), {{-0.3, 0.1}},
                                   {{0.4, -0.24}});
  test_solution<3, IsRelativistic>(std::numeric_limits<double>::signaling_NaN(),
                                   {{-0.3, 0.1, -0.002}},
                                   {{0.4, -0.24, 0.054}});
  test_solution<3, IsRelativistic>(DataVector(5), {{-0.3, 0.1, -0.002}},
                                   {{0.4, -0.24, 0.054}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Hydro.SmoothFlow",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Hydro"};
  test<true>();
  test<false>();
}
