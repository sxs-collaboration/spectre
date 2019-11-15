// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/ScalarWaveKerrSchild.hpp"

#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Element.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/CurvedScalarWave/Equations.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace CurvedScalarWave {
namespace AnalyticData {

// Enable or disable this constructor based on ScalarFieldSolution
template <>
template <typename Dummy,
          Requires<std::is_same<ScalarWave::Solutions::RegularSphericalWave,
                                Dummy>::value>>
ScalarWaveKerrSchild<ScalarWave::Solutions::RegularSphericalWave>::
    ScalarWaveKerrSchild(
        const double mass,
        const std::array<double, volume_dim> dimensionless_spin,
        const std::array<double, volume_dim> center,
        std::unique_ptr<MathFunction<1>> profile,
        const OptionContext& context) noexcept
    : flat_space_wave_soln_(std::move(profile)),
      kerr_schild_soln_(mass, dimensionless_spin, center, context) {}

// Enable or disable this constructor based on ScalarFieldSolution
template <>
template <
    typename Dummy,
    Requires<std::is_same<ScalarWave::Solutions::PlaneWave<3>, Dummy>::value>>
ScalarWaveKerrSchild<ScalarWave::Solutions::PlaneWave<3>>::ScalarWaveKerrSchild(
    const double mass, const std::array<double, volume_dim> dimensionless_spin,
    const std::array<double, volume_dim> center,
    const std::array<double, volume_dim> wave_vector,
    const std::array<double, volume_dim> wave_center,
    std::unique_ptr<MathFunction<1>> profile,
    const OptionContext& context) noexcept
    : flat_space_wave_soln_(wave_vector, wave_center, std::move(profile)),
      kerr_schild_soln_(mass, dimensionless_spin, center, context) {}

template <typename ScalarFieldSolution>
void ScalarWaveKerrSchild<ScalarFieldSolution>::pup(PUP::er& p) noexcept {
  p | flat_space_wave_soln_;
  p | kerr_schild_soln_;
}

template <typename ScalarFieldSolution>
tuples::TaggedTuple<Pi> ScalarWaveKerrSchild<ScalarFieldSolution>::variables(
    const tnsr::I<DataVector, volume_dim>& x, tmpl::list<Pi> /*meta*/) const
    noexcept {
  constexpr double default_initial_time = 0.;
  const auto flat_space_scalar_wave_vars = flat_space_wave_soln_.variables(
      x, default_initial_time,
      tmpl::list<ScalarWave::Pi, ScalarWave::Phi<volume_dim>,
                 ScalarWave::Psi>{});
  const auto flat_space_dt_scalar_wave_vars = flat_space_wave_soln_.variables(
      x, default_initial_time,
      tmpl::list<::Tags::dt<ScalarWave::Pi>,
                 ::Tags::dt<ScalarWave::Phi<volume_dim>>,
                 ::Tags::dt<ScalarWave::Psi>>{});
  const auto kerr_schild_spacetime_variables = kerr_schild_soln_.variables(
      x, default_initial_time, spacetime_tags<DataVector>{});
  auto result = make_with_value<tuples::TaggedTuple<Pi>>(x, 0.);

  {
    const auto shift_dot_dpsi = dot_product(
        get<gr::Tags::Shift<volume_dim, Frame::Inertial, DataVector>>(
            kerr_schild_spacetime_variables),
        get<ScalarWave::Phi<volume_dim>>(flat_space_scalar_wave_vars));

    get(get<Pi>(result)) =
        (get(shift_dot_dpsi) - get(get<::Tags::dt<ScalarWave::Psi>>(
                                   flat_space_dt_scalar_wave_vars))) /
        get(get<gr::Tags::Lapse<DataVector>>(kerr_schild_spacetime_variables));
  }

  return result;
}

template <typename LocalScalarFieldSolution>
bool operator==(
    const ScalarWaveKerrSchild<LocalScalarFieldSolution>& lhs,
    const ScalarWaveKerrSchild<LocalScalarFieldSolution>& rhs) noexcept {
  return lhs.mass() == rhs.mass() and
         lhs.dimensionless_spin() == rhs.dimensionless_spin() and
         lhs.center() == rhs.center();
}

template <typename LocalScalarFieldSolution>
bool operator!=(
    const ScalarWaveKerrSchild<LocalScalarFieldSolution>& lhs,
    const ScalarWaveKerrSchild<LocalScalarFieldSolution>& rhs) noexcept {
  return not(lhs == rhs);
}

// Generate instantiations
#define STYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                  \
  template class ScalarWaveKerrSchild<STYPE(data)>;           \
  template bool operator==(                                   \
      const ScalarWaveKerrSchild<STYPE(data)>& lhs,           \
      const ScalarWaveKerrSchild<STYPE(data)>& rhs) noexcept; \
  template bool operator!=(                                   \
      const ScalarWaveKerrSchild<STYPE(data)>& lhs,           \
      const ScalarWaveKerrSchild<STYPE(data)>& rhs) noexcept;

#define INSTANTIATE_CONSTRUCTOR1(_, data)                           \
  template ScalarWaveKerrSchild<STYPE(data)>::ScalarWaveKerrSchild( \
      const double mass,                                            \
      const std::array<double, volume_dim> dimensionless_spin,      \
      const std::array<double, volume_dim> center,                  \
      std::unique_ptr<MathFunction<1>> profile,                     \
      const OptionContext& context) noexcept;

#define INSTANTIATE_CONSTRUCTOR2(_, data)                           \
  template ScalarWaveKerrSchild<STYPE(data)>::ScalarWaveKerrSchild( \
      const double mass,                                            \
      const std::array<double, volume_dim> dimensionless_spin,      \
      const std::array<double, volume_dim> center,                  \
      std::array<double, volume_dim> wave_vector,                   \
      std::array<double, volume_dim> wave_center,                   \
      std::unique_ptr<MathFunction<1>> profile,                     \
      const OptionContext& context) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (ScalarWave::Solutions::RegularSphericalWave,
                         ScalarWave::Solutions::PlaneWave<3>))
GENERATE_INSTANTIATIONS(INSTANTIATE_CONSTRUCTOR1,
                        (ScalarWave::Solutions::RegularSphericalWave))
GENERATE_INSTANTIATIONS(INSTANTIATE_CONSTRUCTOR2,
                        (ScalarWave::Solutions::PlaneWave<3>))

#undef INSTANTIATE
#undef INSTANTIATE_CONSTRUCTOR1
#undef INSTANTIATE_CONSTRUCTOR2
#undef STYPE
}  // namespace AnalyticData
}  // namespace CurvedScalarWave
/// \endcond
