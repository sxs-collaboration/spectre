// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/Minmod.hpp"

#include <array>
#include <cstdlib>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/CharacteristicHelpers.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/Flattener.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <size_t ThermodynamicDim>
bool characteristic_minmod_impl(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const Limiters::MinmodType minmod_type, const double tvb_constant,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3>& spatial_metric, const Mesh<3>& mesh,
    const Element<3>& element,
    const tnsr::I<DataVector, 3, Frame::Logical>& logical_coords,
    const std::array<double, 3>& element_size,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<3>, ElementId<3>>,
        typename grmhd::ValenciaDivClean::Limiters::Minmod::PackagedData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>&
        neighbor_data) noexcept {
  // Storage for transforming neighbor_data into char variables
  using CharacteristicVarsMinmod = Limiters::Minmod<
      3, tmpl::list<grmhd::ValenciaDivClean::Tags::VDivCleanMinus,
                    grmhd::ValenciaDivClean::Tags::VMinus,
                    grmhd::ValenciaDivClean::Tags::VMomentum,
                    grmhd::ValenciaDivClean::Tags::VPlus,
                    grmhd::ValenciaDivClean::Tags::VDivCleanPlus>>;
  std::unordered_map<std::pair<Direction<3>, ElementId<3>>,
                     typename CharacteristicVarsMinmod::PackagedData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_char_data{};
  for (const auto& [key, data] : neighbor_data) {
    neighbor_char_data[key].element_size = data.element_size;
  }

  // Buffers for minmod limiter and TCI
  DataVector u_lin_buffer(mesh.number_of_grid_points());
  Limiters::Minmod_detail::BufferWrapper<3> tci_buffer(mesh);

  // Outer lambda: wraps applying minmod to the ValenciaDivClean characteristics
  // for one particular choice of characteristic decomposition
  const auto minmod_convert_neighbor_data_then_limit =
      [&u_lin_buffer, &tci_buffer, &minmod_type, &tvb_constant, &mesh, &element,
       &logical_coords, &element_size, &neighbor_data, &neighbor_char_data](
          const gsl::not_null<Scalar<DataVector>*> char_v_div_clean_minus,
          const gsl::not_null<Scalar<DataVector>*> char_v_minus,
          const gsl::not_null<tnsr::I<DataVector, 5>*> char_v_momentum,
          const gsl::not_null<Scalar<DataVector>*> char_v_plus,
          const gsl::not_null<Scalar<DataVector>*> char_v_div_clean_plus,
          const Matrix& left_eigenvectors) noexcept -> bool {
    // Convert neighbor data to characteristics
    for (const auto& [key, data] : neighbor_data) {
      grmhd::ValenciaDivClean::Limiters::characteristic_fields(
          make_not_null(&(neighbor_char_data[key].means)), data.means,
          left_eigenvectors);
    }

    bool some_component_was_limited_with_this_normal = false;

    // Inner lambda: apply minmod to one particular tensor
    const auto wrap_minmod = [&some_component_was_limited_with_this_normal,
                              &u_lin_buffer, &tci_buffer, &minmod_type,
                              &tvb_constant, &mesh, &element, &logical_coords,
                              &element_size, &neighbor_char_data](
                                 auto tag, const auto tensor) noexcept {
      const bool result =
          Limiters::Minmod_detail::minmod_impl<3, decltype(tag)>(
              &u_lin_buffer, &tci_buffer, tensor, minmod_type, tvb_constant,
              mesh, element, logical_coords, element_size, neighbor_char_data);
      some_component_was_limited_with_this_normal =
          result or some_component_was_limited_with_this_normal;
    };
    wrap_minmod(grmhd::ValenciaDivClean::Tags::VDivCleanMinus{},
                char_v_div_clean_minus);
    wrap_minmod(grmhd::ValenciaDivClean::Tags::VMinus{}, char_v_minus);
    wrap_minmod(grmhd::ValenciaDivClean::Tags::VMomentum{}, char_v_momentum);
    wrap_minmod(grmhd::ValenciaDivClean::Tags::VPlus{}, char_v_plus);
    wrap_minmod(grmhd::ValenciaDivClean::Tags::VDivCleanPlus{},
                char_v_div_clean_plus);
    return some_component_was_limited_with_this_normal;
  };

  return grmhd::ValenciaDivClean::Limiters::
      apply_limiter_to_characteristic_fields_in_all_directions(
          tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
          spatial_metric, mesh, equation_of_state,
          minmod_convert_neighbor_data_then_limit);
}

}  // namespace

namespace grmhd::ValenciaDivClean::Limiters {

Minmod::Minmod(
    const ::Limiters::MinmodType minmod_type,
    const grmhd::ValenciaDivClean::Limiters::VariablesToLimit vars_to_limit,
    const double tvb_constant, const bool apply_flattener,
    const bool disable_for_debugging) noexcept
    : minmod_type_(minmod_type),
      vars_to_limit_(vars_to_limit),
      tvb_constant_(tvb_constant),
      apply_flattener_(apply_flattener),
      disable_for_debugging_(disable_for_debugging),
      conservative_vars_minmod_(minmod_type_, tvb_constant_,
                                disable_for_debugging_) {
  ASSERT(tvb_constant >= 0.0, "The TVB constant must be non-negative.");
}

void Minmod::pup(PUP::er& p) noexcept {
  p | minmod_type_;
  p | vars_to_limit_;
  p | tvb_constant_;
  p | apply_flattener_;
  p | disable_for_debugging_;
  p | conservative_vars_minmod_;
}

void Minmod::package_data(
    const gsl::not_null<PackagedData*> packaged_data,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3>& tilde_s,
    const tnsr::I<DataVector, 3>& tilde_b, const Scalar<DataVector>& tilde_phi,
    const Mesh<3>& mesh, const std::array<double, 3>& element_size,
    const OrientationMap<3>& orientation_map) const noexcept {
  conservative_vars_minmod_.package_data(packaged_data, tilde_d, tilde_tau,
                                         tilde_s, tilde_b, tilde_phi, mesh,
                                         element_size, orientation_map);
}

template <size_t ThermodynamicDim>
bool Minmod::operator()(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3>& spatial_metric, const Mesh<3>& mesh,
    const Element<3>& element,
    const tnsr::I<DataVector, 3, Frame::Logical>& logical_coords,
    const std::array<double, 3>& element_size,
    const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<3>, ElementId<3>>, PackagedData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& neighbor_data)
    const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not modify input tensors
    return false;
  }

  // Checks for the post-timestep, pre-limiter ValenciaDivClean state:
#ifdef SPECTRE_DEBUG
  const double mean_tilde_d = mean_value(get(*tilde_d), mesh);
  ASSERT(mean_tilde_d > 0.0,
         "Positivity was violated on a cell-average level.");
  // TODO: Add other simple positivity checks?
#endif  // SPECTRE_DEBUG

  bool limiter_activated = false;

  if (vars_to_limit_ ==
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::Conserved) {
    limiter_activated = conservative_vars_minmod_(
        tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, mesh, element,
        logical_coords, element_size, neighbor_data);
  } else if (vars_to_limit_ == grmhd::ValenciaDivClean::Limiters::
                                   VariablesToLimit::NumericalCharacteristic) {
    limiter_activated = characteristic_minmod_impl(
        tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, minmod_type_,
        tvb_constant_, lapse, shift, spatial_metric, mesh, element,
        logical_coords, element_size, equation_of_state, neighbor_data);
  } else {
    ERROR(
        "No implementation of grmhd::ValenciaDivClean::Limiters::Minmod for "
        "variables: "
        << vars_to_limit_);
  }

  if (apply_flattener_) {
    const Scalar<DataVector> det_logical_to_inertial_jacobian{
        1.0 / get(det_inv_logical_to_inertial_jacobian)};
    const auto flattener_action = flatten_solution(
        tilde_d, tilde_tau, tilde_s, *tilde_b, sqrt_det_spatial_metric,
        spatial_metric, mesh, det_logical_to_inertial_jacobian);
    if (flattener_action != FlattenerAction::NoOp) {
      limiter_activated = true;
    }
  }

  // Checks for the post-limiter ValenciaDivClean state:
#ifdef SPECTRE_DEBUG
  ASSERT(min(get(*tilde_d)) > 0.0, "Bad density after limiting.");
  // TODO: Add other simple positivity checks?
#endif  // SPECTRE_DEBUG

  return limiter_activated;
}

bool operator==(const Minmod& lhs, const Minmod& rhs) noexcept {
  // No need to compare the conservative_vars_minmod_ member variable because
  // it is constructed from the other member variables.
  return lhs.minmod_type_ == rhs.minmod_type_ and
         lhs.vars_to_limit_ == rhs.vars_to_limit_ and
         lhs.tvb_constant_ == rhs.tvb_constant_ and
         lhs.apply_flattener_ == rhs.apply_flattener_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

bool operator!=(const Minmod& lhs, const Minmod& rhs) noexcept {
  return not(lhs == rhs);
}

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template bool Minmod::operator()(                                        \
      const gsl::not_null<Scalar<DataVector>*>,                            \
      const gsl::not_null<Scalar<DataVector>*>,                            \
      const gsl::not_null<tnsr::i<DataVector, 3>*>,                        \
      const gsl::not_null<tnsr::I<DataVector, 3>*>,                        \
      const gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&, \
      const Scalar<DataVector>&, const tnsr::I<DataVector, 3>&,            \
      const tnsr::ii<DataVector, 3>&, const Mesh<3>&, const Element<3>&,   \
      const tnsr::I<DataVector, 3, Frame::Logical>&,                       \
      const std::array<double, 3>&, const Scalar<DataVector>&,             \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>&,    \
      const std::unordered_map<                                            \
          std::pair<Direction<3>, ElementId<3>>, PackagedData,             \
          boost::hash<std::pair<Direction<3>, ElementId<3>>>>&)            \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2))

#undef INSTANTIATE
#undef THERMO_DIM

}  // namespace grmhd::ValenciaDivClean::Limiters
