// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Limiters/Weno.hpp"

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Structure/Direction.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Element.hpp"    // IWYU pragma: keep
#include "Domain/Structure/ElementId.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"                 // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Limiters/HwenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodTci.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/SimpleWenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/CharacteristicHelpers.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Flattener.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/KxrcfTci.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/VariablesToLimit.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/Weno.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t VolumeDim, size_t ThermodynamicDim>
bool characteristic_simple_weno_impl(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const double tvb_constant, const double neighbor_linear_weight,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename NewtonianEuler::Limiters::Weno<VolumeDim>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const bool compute_char_transformation_numerically) {
  // Storage for transforming neighbor_data into char variables
  using CharacteristicVarsWeno =
      Limiters::Weno<VolumeDim,
                     tmpl::list<NewtonianEuler::Tags::VMinus,
                                NewtonianEuler::Tags::VMomentum<VolumeDim>,
                                NewtonianEuler::Tags::VPlus>>;
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      typename CharacteristicVarsWeno::PackagedData,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_char_data{};
  for (const auto& [key, data] : neighbor_data) {
    neighbor_char_data[key].volume_data.initialize(
        mesh.number_of_grid_points());
    neighbor_char_data[key].mesh = data.mesh;
    neighbor_char_data[key].element_size = data.element_size;
  }

  // Buffers for TCI
  Limiters::Minmod_detail::BufferWrapper<VolumeDim> tci_buffer(mesh);
  const auto effective_neighbor_sizes =
      Limiters::Minmod_detail::compute_effective_neighbor_sizes(element,
                                                                neighbor_data);

  // Buffers for SimpleWeno extrapolated poly
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      intrp::RegularGrid<VolumeDim>,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      interpolator_buffer{};
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      modified_neighbor_solution_buffer{};

  // Outer lambda: wraps applying SimpleWeno to the NewtonianEuler
  // characteristics for one particular choice of characteristic decomposition
  const auto simple_weno_convert_neighbor_data_then_limit =
      [&tci_buffer, &interpolator_buffer, &modified_neighbor_solution_buffer,
       &tvb_constant, &neighbor_linear_weight, &mesh, &element, &element_size,
       &neighbor_data, &neighbor_char_data, &effective_neighbor_sizes](
          const gsl::not_null<Scalar<DataVector>*> char_v_minus,
          const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> char_v_momentum,
          const gsl::not_null<Scalar<DataVector>*> char_v_plus,
          const Matrix& left_eigenvectors) -> bool {
    // Convert neighbor data to characteristics
    for (const auto& [key, data] : neighbor_data) {
      NewtonianEuler::Limiters::characteristic_fields(
          make_not_null(&(neighbor_char_data[key].volume_data)),
          data.volume_data, left_eigenvectors);
      NewtonianEuler::Limiters::characteristic_fields(
          make_not_null(&(neighbor_char_data[key].means)), data.means,
          left_eigenvectors);
    }

    bool some_component_was_limited_with_this_normal = false;

    // Inner lambda: apply SimpleWeno to one particular tensor
    const auto wrap_minmod_tci_and_simple_weno =
        [&some_component_was_limited_with_this_normal, &tci_buffer,
         &interpolator_buffer, &modified_neighbor_solution_buffer,
         &tvb_constant, &neighbor_linear_weight, &mesh, &element, &element_size,
         &neighbor_char_data,
         &effective_neighbor_sizes](auto tag, const auto tensor) {
          for (size_t tensor_storage_index = 0;
               tensor_storage_index < tensor->size(); ++tensor_storage_index) {
            // Check TCI
            const auto effective_neighbor_means =
                Limiters::Minmod_detail::compute_effective_neighbor_means<
                    decltype(tag)>(tensor_storage_index, element,
                                   neighbor_char_data);
            const bool component_needs_limiting =
                Limiters::Tci::tvb_minmod_indicator(
                    make_not_null(&tci_buffer), tvb_constant,
                    (*tensor)[tensor_storage_index], mesh, element,
                    element_size, effective_neighbor_means,
                    effective_neighbor_sizes);

            if (component_needs_limiting) {
              if (modified_neighbor_solution_buffer.empty()) {
                // Allocate the neighbor solution buffers only if the limiter is
                // triggered. This reduces allocation when no limiting occurs.
                for (const auto& [neighbor, data] : neighbor_char_data) {
                  (void)data;
                  modified_neighbor_solution_buffer.insert(std::make_pair(
                      neighbor, DataVector(mesh.number_of_grid_points())));
                }
              }
              Limiters::Weno_detail::simple_weno_impl<decltype(tag)>(
                  make_not_null(&interpolator_buffer),
                  make_not_null(&modified_neighbor_solution_buffer), tensor,
                  neighbor_linear_weight, tensor_storage_index, mesh, element,
                  neighbor_char_data);
              some_component_was_limited_with_this_normal = true;
            }
          }
        };
    wrap_minmod_tci_and_simple_weno(NewtonianEuler::Tags::VMinus{},
                                    char_v_minus);
    wrap_minmod_tci_and_simple_weno(
        NewtonianEuler::Tags::VMomentum<VolumeDim>{}, char_v_momentum);
    wrap_minmod_tci_and_simple_weno(NewtonianEuler::Tags::VPlus{}, char_v_plus);
    return some_component_was_limited_with_this_normal;
  };

  return NewtonianEuler::Limiters::
      apply_limiter_to_characteristic_fields_in_all_directions(
          mass_density_cons, momentum_density, energy_density, mesh,
          equation_of_state, simple_weno_convert_neighbor_data_then_limit,
          compute_char_transformation_numerically);
}

template <size_t VolumeDim, size_t ThermodynamicDim>
bool characteristic_hweno_impl(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const double kxrcf_constant, const double neighbor_linear_weight,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const Scalar<DataVector>& det_logical_to_inertial_jacobian,
    const typename evolution::dg::Tags::NormalCovectorAndMagnitude<
        VolumeDim>::type& normals_and_magnitudes,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename NewtonianEuler::Limiters::Weno<VolumeDim>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const bool compute_char_transformation_numerically) {
  // Hweno checks TCI before limiting any tensors
  const bool cell_is_troubled = NewtonianEuler::Limiters::Tci::kxrcf_indicator(
      kxrcf_constant, *mass_density_cons, *momentum_density, *energy_density,
      mesh, element, element_size, det_logical_to_inertial_jacobian,
      normals_and_magnitudes, neighbor_data);
  if (not cell_is_troubled) {
    // No limiting is needed
    return false;
  }

  // Storage for transforming neighbor_data into char variables
  using CharacteristicVarsWeno =
      Limiters::Weno<VolumeDim,
                     tmpl::list<NewtonianEuler::Tags::VMinus,
                                NewtonianEuler::Tags::VMomentum<VolumeDim>,
                                NewtonianEuler::Tags::VPlus>>;
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      typename CharacteristicVarsWeno::PackagedData,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_char_data{};
  for (const auto& [key, data] : neighbor_data) {
    neighbor_char_data[key].volume_data.initialize(
        mesh.number_of_grid_points());
    neighbor_char_data[key].mesh = data.mesh;
    neighbor_char_data[key].element_size = data.element_size;
  }

  // Buffers for Hweno extrapolated poly
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      modified_neighbor_solution_buffer{};
  for (const auto& [neighbor, data] : neighbor_char_data) {
    (void)data;
    modified_neighbor_solution_buffer.insert(
        std::make_pair(neighbor, DataVector(mesh.number_of_grid_points())));
  }

  // Lambda wraps applying Hweno to the NewtonianEuler characteristics for one
  // particular choice of characteristic decomposition
  const auto hweno_convert_neighbor_data_then_limit =
      [&modified_neighbor_solution_buffer, &neighbor_linear_weight, &mesh,
       &element, &neighbor_data, &neighbor_char_data](
          const gsl::not_null<Scalar<DataVector>*> char_v_minus,
          const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> char_v_momentum,
          const gsl::not_null<Scalar<DataVector>*> char_v_plus,
          const Matrix& left_eigenvectors) -> bool {
    // Convert neighbor data to characteristics
    for (const auto& [key, data] : neighbor_data) {
      NewtonianEuler::Limiters::characteristic_fields(
          make_not_null(&(neighbor_char_data[key].volume_data)),
          data.volume_data, left_eigenvectors);
      NewtonianEuler::Limiters::characteristic_fields(
          make_not_null(&(neighbor_char_data[key].means)), data.means,
          left_eigenvectors);
    }

    ::Limiters::Weno_detail::hweno_impl<NewtonianEuler::Tags::VMinus>(
        make_not_null(&modified_neighbor_solution_buffer), char_v_minus,
        neighbor_linear_weight, mesh, element, neighbor_char_data);
    ::Limiters::Weno_detail::hweno_impl<
        NewtonianEuler::Tags::VMomentum<VolumeDim>>(
        make_not_null(&modified_neighbor_solution_buffer), char_v_momentum,
        neighbor_linear_weight, mesh, element, neighbor_char_data);
    ::Limiters::Weno_detail::hweno_impl<NewtonianEuler::Tags::VPlus>(
        make_not_null(&modified_neighbor_solution_buffer), char_v_plus,
        neighbor_linear_weight, mesh, element, neighbor_char_data);
    return true;  // all components were limited
  };

  NewtonianEuler::Limiters::
      apply_limiter_to_characteristic_fields_in_all_directions(
          mass_density_cons, momentum_density, energy_density, mesh,
          equation_of_state, hweno_convert_neighbor_data_then_limit,
          compute_char_transformation_numerically);
  return true;  // all components were limited
}

template <size_t VolumeDim>
bool conservative_hweno_impl(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const double kxrcf_constant, const double neighbor_linear_weight,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const Scalar<DataVector>& det_logical_to_inertial_jacobian,
    const typename evolution::dg::Tags::NormalCovectorAndMagnitude<
        VolumeDim>::type& normals_and_magnitudes,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename NewtonianEuler::Limiters::Weno<VolumeDim>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) {
  // Hweno checks TCI before limiting any tensors
  const bool cell_is_troubled = NewtonianEuler::Limiters::Tci::kxrcf_indicator(
      kxrcf_constant, *mass_density_cons, *momentum_density, *energy_density,
      mesh, element, element_size, det_logical_to_inertial_jacobian,
      normals_and_magnitudes, neighbor_data);
  if (not cell_is_troubled) {
    // No limiting is needed
    return false;
  }

  // Buffers for Hweno extrapolated poly
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      modified_neighbor_solution_buffer{};
  for (const auto& [neighbor, data] : neighbor_data) {
    (void)data;
    modified_neighbor_solution_buffer.insert(
        std::make_pair(neighbor, DataVector(mesh.number_of_grid_points())));
  }

  ::Limiters::Weno_detail::hweno_impl<NewtonianEuler::Tags::MassDensityCons>(
      make_not_null(&modified_neighbor_solution_buffer), mass_density_cons,
      neighbor_linear_weight, mesh, element, neighbor_data);
  ::Limiters::Weno_detail::hweno_impl<
      NewtonianEuler::Tags::MomentumDensity<VolumeDim>>(
      make_not_null(&modified_neighbor_solution_buffer), momentum_density,
      neighbor_linear_weight, mesh, element, neighbor_data);
  ::Limiters::Weno_detail::hweno_impl<NewtonianEuler::Tags::EnergyDensity>(
      make_not_null(&modified_neighbor_solution_buffer), energy_density,
      neighbor_linear_weight, mesh, element, neighbor_data);
  return true;  // all components were limited
}

}  // namespace

namespace NewtonianEuler::Limiters {

template <size_t VolumeDim>
Weno<VolumeDim>::Weno(
    const ::Limiters::WenoType weno_type,
    const NewtonianEuler::Limiters::VariablesToLimit vars_to_limit,
    const double neighbor_linear_weight,
    const std::optional<double> tvb_constant,
    const std::optional<double> kxrcf_constant, const bool apply_flattener,
    const bool disable_for_debugging, const Options::Context& context)
    : weno_type_(weno_type),
      vars_to_limit_(vars_to_limit),
      neighbor_linear_weight_(neighbor_linear_weight),
      tvb_constant_(tvb_constant),
      kxrcf_constant_(kxrcf_constant),
      apply_flattener_(apply_flattener),
      disable_for_debugging_(disable_for_debugging),
      conservative_vars_weno_(
          weno_type_, neighbor_linear_weight_,
          tvb_constant_.value_or(std::numeric_limits<double>::signaling_NaN()),
          disable_for_debugging_) {
  if (weno_type == ::Limiters::WenoType::Hweno) {
    if (tvb_constant.has_value() or not kxrcf_constant.has_value()) {
      PARSE_ERROR(context,
                  "The Hweno limiter uses the KXRCF TCI. The TvbConstant must "
                  "be set to 'None', and the KxrcfConstant must be set to a "
                  "non-negative value.");
    }
    if (kxrcf_constant.value() < 0.0) {
      PARSE_ERROR(context, "The KXRCF constant must be non-negative, but got: "
                               << kxrcf_constant.value());
    }
  } else {  // SimpleWeno
    if (not tvb_constant.has_value() or kxrcf_constant.has_value()) {
      PARSE_ERROR(context,
                  "The SimpleWeno limiter uses the TVB minmod TCI. The "
                  "TvbConstant must be set to a non-negative value, and the "
                  "KxrcfConstant must be set to 'None'.");
    }
    if (tvb_constant.value() < 0.0) {
      PARSE_ERROR(context, "The TVB constant must be non-negative, but got: "
                               << tvb_constant.value());
    }
  }
}

template <size_t VolumeDim>
// NOLINTNEXTLINE(google-runtime-references)
void Weno<VolumeDim>::pup(PUP::er& p) {
  p | weno_type_;
  p | vars_to_limit_;
  p | neighbor_linear_weight_;
  p | tvb_constant_;
  p | kxrcf_constant_;
  p | apply_flattener_;
  p | disable_for_debugging_;
}

template <size_t VolumeDim>
void Weno<VolumeDim>::package_data(
    const gsl::not_null<PackagedData*> packaged_data,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, VolumeDim>& momentum_density,
    const Scalar<DataVector>& energy_density, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const OrientationMap<VolumeDim>& orientation_map) const {
  conservative_vars_weno_.package_data(packaged_data, mass_density_cons,
                                       momentum_density, energy_density, mesh,
                                       element_size, orientation_map);
}

template <size_t VolumeDim>
template <size_t ThermodynamicDim>
bool Weno<VolumeDim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,
    const typename evolution::dg::Tags::NormalCovectorAndMagnitude<
        VolumeDim>::type& normals_and_magnitudes,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) const {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not modify input tensors
    return false;
  }

  // Enforce restrictions on h-refinement, p-refinement
  if (UNLIKELY(
          alg::any_of(element.neighbors(), [](const auto& direction_neighbors) {
            return direction_neighbors.second.size() != 1;
          }))) {
    ERROR("The Weno limiter does not yet support h-refinement");
    // Removing this limitation will require:
    // - Generalizing the computation of the modified neighbor solutions.
    // - Generalizing the WENO weighted sum for multiple neighbors in each
    //   direction.
  }
  alg::for_each(neighbor_data, [&mesh](const auto& neighbor_and_data) {
    if (UNLIKELY(neighbor_and_data.second.mesh != mesh)) {
      ERROR("The Weno limiter does not yet support p-refinement");
      // Removing this limitation will require generalizing the
      // computation of the modified neighbor solutions.
    }
  });

  // Checks for the post-timestep, pre-limiter NewtonianEuler state:
#ifdef SPECTRE_DEBUG
  const double mean_density = mean_value(get(*mass_density_cons), mesh);
  ASSERT(mean_density > 0.0,
         "Positivity was violated on a cell-average level.");
  if (ThermodynamicDim == 2) {
    const double mean_energy = mean_value(get(*energy_density), mesh);
    ASSERT(mean_energy > 0.0,
           "Positivity was violated on a cell-average level.");
  }
#endif  // SPECTRE_DEBUG

  bool limiter_activated = false;

  // Small possible optimization: only compute this if needed
  const Scalar<DataVector> det_logical_to_inertial_jacobian{
      1. / get(det_inv_logical_to_inertial_jacobian)};

  if (weno_type_ == ::Limiters::WenoType::Hweno) {
    if (vars_to_limit_ ==
            NewtonianEuler::Limiters::VariablesToLimit::Characteristic or
        vars_to_limit_ == NewtonianEuler::Limiters::VariablesToLimit::
                              NumericalCharacteristic) {
      const bool compute_char_transformation_numerically =
          (vars_to_limit_ ==
           NewtonianEuler::Limiters::VariablesToLimit::NumericalCharacteristic);
      // impl function handles specialized TCI + char transform
      limiter_activated = characteristic_hweno_impl(
          mass_density_cons, momentum_density, energy_density,
          kxrcf_constant_.value(), neighbor_linear_weight_, mesh, element,
          element_size, det_logical_to_inertial_jacobian,
          normals_and_magnitudes, equation_of_state, neighbor_data,
          compute_char_transformation_numerically);
    } else {
      // impl function handles specialized TCI
      limiter_activated = conservative_hweno_impl(
          mass_density_cons, momentum_density, energy_density,
          kxrcf_constant_.value(), neighbor_linear_weight_, mesh, element,
          element_size, det_logical_to_inertial_jacobian,
          normals_and_magnitudes, neighbor_data);
    }
  } else if (weno_type_ == ::Limiters::WenoType::SimpleWeno) {
    if (vars_to_limit_ ==
            NewtonianEuler::Limiters::VariablesToLimit::Characteristic or
        vars_to_limit_ == NewtonianEuler::Limiters::VariablesToLimit::
                              NumericalCharacteristic) {
      const bool compute_char_transformation_numerically =
          (vars_to_limit_ ==
           NewtonianEuler::Limiters::VariablesToLimit::NumericalCharacteristic);
      // impl function handles char transform
      limiter_activated = characteristic_simple_weno_impl(
          mass_density_cons, momentum_density, energy_density,
          tvb_constant_.value(), neighbor_linear_weight_, mesh, element,
          element_size, equation_of_state, neighbor_data,
          compute_char_transformation_numerically);
    } else {
      // Fall back to generic SimpleWeno
      limiter_activated = conservative_vars_weno_(
          mass_density_cons, momentum_density, energy_density, mesh, element,
          element_size, neighbor_data);
    }
  }

  if (apply_flattener_) {
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
    ASSERT(min(get(pressure)) > 0.0, "Bad energy after limiting.");
  }
#endif  // SPECTRE_DEBUG

  return limiter_activated;
}

template <size_t LocalDim>
bool operator==(const Weno<LocalDim>& lhs, const Weno<LocalDim>& rhs) {
  // No need to compare the conservative_vars_weno_ member variable because
  // it is constructed from the other member variables.
  return lhs.weno_type_ == rhs.weno_type_ and
         lhs.vars_to_limit_ == rhs.vars_to_limit_ and
         lhs.neighbor_linear_weight_ == rhs.neighbor_linear_weight_ and
         lhs.tvb_constant_ == rhs.tvb_constant_ and
         lhs.kxrcf_constant_ == rhs.kxrcf_constant_ and
         lhs.apply_flattener_ == rhs.apply_flattener_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim>
bool operator!=(const Weno<VolumeDim>& lhs, const Weno<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template class Weno<DIM(data)>;                                           \
  template bool operator==(const Weno<DIM(data)>&, const Weno<DIM(data)>&); \
  template bool operator!=(const Weno<DIM(data)>&, const Weno<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                                                   \
  template bool Weno<DIM(data)>::operator()(                                   \
      const gsl::not_null<Scalar<DataVector>*>,                                \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                    \
      const gsl::not_null<Scalar<DataVector>*>, const Mesh<DIM(data)>&,        \
      const Element<DIM(data)>&, const std::array<double, DIM(data)>&,         \
      const Scalar<DataVector>&,                                               \
      const typename evolution::dg::Tags::NormalCovectorAndMagnitude<DIM(      \
          data)>::type&,                                                       \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>&,       \
      const std::unordered_map<                                                \
          std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>, PackagedData, \
          boost::hash<                                                         \
              std::pair<Direction<DIM(data)>, ElementId<DIM(data)>>>>&) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (1, 2))

#undef INSTANTIATE
#undef DIM
#undef THERMO_DIM

}  // namespace NewtonianEuler::Limiters
