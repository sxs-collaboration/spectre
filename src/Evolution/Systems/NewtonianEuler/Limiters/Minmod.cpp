// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Limiters/Minmod.hpp"

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
#include "Evolution/Systems/NewtonianEuler/Limiters/CharacteristicHelpers.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Flattener.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <size_t VolumeDim, size_t ThermodynamicDim>
bool characteristic_minmod_impl(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const Limiters::MinmodType minmod_type, const double tvb_constant,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const tnsr::I<DataVector, VolumeDim, Frame::ElementLogical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename NewtonianEuler::Limiters::Minmod<VolumeDim>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const bool compute_char_transformation_numerically) noexcept {
  // Storage for transforming neighbor_data into char variables
  using CharacteristicVarsMinmod =
      Limiters::Minmod<VolumeDim,
                       tmpl::list<NewtonianEuler::Tags::VMinus,
                                  NewtonianEuler::Tags::VMomentum<VolumeDim>,
                                  NewtonianEuler::Tags::VPlus>>;
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      typename CharacteristicVarsMinmod::PackagedData,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_char_data{};
  for (const auto& [key, data] : neighbor_data) {
    neighbor_char_data[key].element_size = data.element_size;
  }

  // Buffers for minmod limiter and TCI
  DataVector u_lin_buffer(mesh.number_of_grid_points());
  Limiters::Minmod_detail::BufferWrapper<VolumeDim> tci_buffer(mesh);

  // Outer lambda: wraps applying minmod to the NewtonianEuler characteristics
  // for one particular choice of characteristic decomposition
  const auto minmod_convert_neighbor_data_then_limit =
      [&u_lin_buffer, &tci_buffer, &minmod_type, &tvb_constant, &mesh, &element,
       &logical_coords, &element_size, &neighbor_data, &neighbor_char_data](
          const gsl::not_null<Scalar<DataVector>*> char_v_minus,
          const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> char_v_momentum,
          const gsl::not_null<Scalar<DataVector>*> char_v_plus,
          const Matrix& left_eigenvectors) noexcept -> bool {
    // Convert neighbor data to characteristics
    for (const auto& [key, data] : neighbor_data) {
      NewtonianEuler::Limiters::characteristic_fields(
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
          Limiters::Minmod_detail::minmod_impl<VolumeDim, decltype(tag)>(
              &u_lin_buffer, &tci_buffer, tensor, minmod_type, tvb_constant,
              mesh, element, logical_coords, element_size, neighbor_char_data);
      some_component_was_limited_with_this_normal =
          result or some_component_was_limited_with_this_normal;
    };
    wrap_minmod(NewtonianEuler::Tags::VMinus{}, char_v_minus);
    wrap_minmod(NewtonianEuler::Tags::VMomentum<VolumeDim>{}, char_v_momentum);
    wrap_minmod(NewtonianEuler::Tags::VPlus{}, char_v_plus);
    return some_component_was_limited_with_this_normal;
  };

  return NewtonianEuler::Limiters::
      apply_limiter_to_characteristic_fields_in_all_directions(
          mass_density_cons, momentum_density, energy_density, mesh,
          equation_of_state, minmod_convert_neighbor_data_then_limit,
          compute_char_transformation_numerically);
}

}  // namespace

namespace NewtonianEuler::Limiters {

template <size_t VolumeDim>
Minmod<VolumeDim>::Minmod(
    const ::Limiters::MinmodType minmod_type,
    const NewtonianEuler::Limiters::VariablesToLimit vars_to_limit,
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

template <size_t VolumeDim>
void Minmod<VolumeDim>::pup(PUP::er& p) noexcept {
  p | minmod_type_;
  p | vars_to_limit_;
  p | tvb_constant_;
  p | apply_flattener_;
  p | disable_for_debugging_;
  p | conservative_vars_minmod_;
}

template <size_t VolumeDim>
void Minmod<VolumeDim>::package_data(
    const gsl::not_null<PackagedData*> packaged_data,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, VolumeDim>& momentum_density,
    const Scalar<DataVector>& energy_density, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const OrientationMap<VolumeDim>& orientation_map) const noexcept {
  conservative_vars_minmod_.package_data(packaged_data, mass_density_cons,
                                         momentum_density, energy_density, mesh,
                                         element_size, orientation_map);
}

template <size_t VolumeDim>
template <size_t ThermodynamicDim>
bool Minmod<VolumeDim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const tnsr::I<DataVector, VolumeDim, Frame::ElementLogical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not modify input tensors
    return false;
  }

  // Checks for the post-timestep, pre-limiter NewtonianEuler state:
#ifdef SPECTRE_DEBUG
  const double mean_density = mean_value(get(*mass_density_cons), mesh);
  ASSERT(mean_density > 0.0,
         "Positivity was violated on a cell-average level.");
  if constexpr (ThermodynamicDim == 2) {
    const double mean_energy = mean_value(get(*energy_density), mesh);
    ASSERT(mean_energy > 0.0,
           "Positivity was violated on a cell-average level.");
  }
#endif  // SPECTRE_DEBUG

  bool limiter_activated = false;

  if (vars_to_limit_ == NewtonianEuler::Limiters::VariablesToLimit::Conserved) {
    limiter_activated = conservative_vars_minmod_(
        mass_density_cons, momentum_density, energy_density, mesh, element,
        logical_coords, element_size, neighbor_data);
  } else if (vars_to_limit_ ==
                 NewtonianEuler::Limiters::VariablesToLimit::Characteristic or
             vars_to_limit_ == NewtonianEuler::Limiters::VariablesToLimit::
                                   NumericalCharacteristic) {
    const bool compute_char_transformation_numerically =
        (vars_to_limit_ ==
         NewtonianEuler::Limiters::VariablesToLimit::NumericalCharacteristic);
    limiter_activated = characteristic_minmod_impl(
        mass_density_cons, momentum_density, energy_density, minmod_type_,
        tvb_constant_, mesh, element, logical_coords, element_size,
        equation_of_state, neighbor_data,
        compute_char_transformation_numerically);
  } else {
    ERROR(
        "No implementation of NewtonianEuler::Limiters::Minmod for variables: "
        << vars_to_limit_);
  }

  if (apply_flattener_) {
    const Scalar<DataVector> det_logical_to_inertial_jacobian{
        1.0 / get(det_inv_logical_to_inertial_jacobian)};
    const auto flattener_action = flatten_solution(
        mass_density_cons, momentum_density, energy_density, mesh,
        det_logical_to_inertial_jacobian, equation_of_state);
    if (flattener_action != FlattenerAction::NoOp) {
      limiter_activated = true;
    }
  }

  // Checks for the post-limiter NewtonianEuler state:
#ifdef SPECTRE_DEBUG
  ASSERT(min(get(*mass_density_cons)) > 0.0, "Bad density after limiting.");
  if constexpr (ThermodynamicDim == 2) {
    const auto specific_internal_energy = Scalar<DataVector>{
        get(*energy_density) / get(*mass_density_cons) -
        0.5 * get(dot_product(*momentum_density, *momentum_density)) /
            square(get(*mass_density_cons))};
    const auto pressure = equation_of_state.pressure_from_density_and_energy(
        *mass_density_cons, specific_internal_energy);
    ASSERT(min(get(pressure)) > 0.0, "Bad pressure after limiting.");
  }
#endif  // SPECTRE_DEBUG

  return limiter_activated;
}

template <size_t LocalDim>
bool operator==(const Minmod<LocalDim>& lhs,
                const Minmod<LocalDim>& rhs) noexcept {
  // No need to compare the conservative_vars_minmod_ member variable because
  // it is constructed from the other member variables.
  return lhs.minmod_type_ == rhs.minmod_type_ and
         lhs.vars_to_limit_ == rhs.vars_to_limit_ and
         lhs.tvb_constant_ == rhs.tvb_constant_ and
         lhs.apply_flattener_ == rhs.apply_flattener_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim>
bool operator!=(const Minmod<VolumeDim>& lhs,
                const Minmod<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                          \
  template class Minmod<DIM(data)>;                   \
  template bool operator==(const Minmod<DIM(data)>&,  \
                           const Minmod<DIM(data)>&); \
  template bool operator!=(const Minmod<DIM(data)>&, const Minmod<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                                                   \
  template bool Minmod<DIM(data)>::operator()(                                 \
      const gsl::not_null<Scalar<DataVector>*>,                                \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                    \
      const gsl::not_null<Scalar<DataVector>*>, const Mesh<DIM(data)>&,        \
      const Element<DIM(data)>&,                                               \
      const tnsr::I<DataVector, DIM(data), Frame::ElementLogical>&,            \
      const std::array<double, DIM(data)>&, const Scalar<DataVector>&,         \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>&,       \
      const std::unordered_map<                                                \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>, PackagedData, \
          boost::hash<                                                         \
              std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>&)        \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (1, 2))

#undef INSTANTIATE
#undef DIM
#undef THERMO_DIM

}  // namespace NewtonianEuler::Limiters
