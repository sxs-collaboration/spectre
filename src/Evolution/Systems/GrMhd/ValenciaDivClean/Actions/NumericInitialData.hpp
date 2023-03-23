// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <variant>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd::ValenciaDivClean::Actions {

namespace detail {
namespace OptionTags {
template <typename Tag, typename IsRequired>
struct VarName {
  using tag = Tag;
  static constexpr bool is_required = IsRequired::value;
  static std::string name() { return db::tag_name<Tag>(); }
  using type = std::conditional_t<is_required, std::string,
                                  std::variant<double, std::string>>;
  static constexpr Options::String help =
      "Name of the variable in the volume data file. For optional variables "
      "you may instead specify a double that is used as a constant value "
      "on the entire grid.";
};
}  // namespace OptionTags

// These are the hydro variables that we support loading from volume
// data files
using required_primitive_vars =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::LowerSpatialFourVelocity<DataVector, 3>>;
using optional_primitive_vars =
    tmpl::list<hydro::Tags::ElectronFraction<DataVector>,
               hydro::Tags::MagneticField<DataVector, 3>>;
using primitive_vars_option_tags =
    tmpl::append<db::wrap_tags_in<OptionTags::VarName, required_primitive_vars,
                                  std::bool_constant<true>>,
                 db::wrap_tags_in<OptionTags::VarName, optional_primitive_vars,
                                  std::bool_constant<false>>>;
struct PrimitiveVarsOptions
    : tuples::tagged_tuple_from_typelist<primitive_vars_option_tags> {
  using Base = tuples::tagged_tuple_from_typelist<primitive_vars_option_tags>;
  static constexpr Options::String help =
      "Primitive hydro variables: 'RestMassDensity' and "
      "'LowerSpatialFourVelocity' (which is u_i = W * gamma_ij v^j). ";
  using options = primitive_vars_option_tags;
  using Base::TaggedTuple;
};

using all_numeric_vars =
    tmpl::append<required_primitive_vars, optional_primitive_vars>;

namespace OptionTags {
template <typename ImporterOptionsGroup>
struct NumericInitialDataVariables {
  static std::string name() { return "Variables"; }
  using type = PrimitiveVarsOptions;
  static constexpr Options::String help =
      "Set of initial data variables from which the Valencia evolution "
      "variables are computed.";
  using group = ImporterOptionsGroup;
};

template <typename ImporterOptionsGroup>
struct DensityCutoff {
  using type = double;
  static constexpr Options::String help =
      "Where the density is below this cutoff the fluid variables are set to "
      "vacuum (zero density, pressure, energy and velocity, unit Lorentz "
      "factor and enthalpy). "
      "During the evolution, atmosphere treatment will typically kick in and "
      "fix the value of the fluid variables in these regions. Therefore, "
      "it makes sense to set this density cutoff to the same value as the "
      "atmosphere density cutoff.";
  using group = ImporterOptionsGroup;
  static constexpr double lower_bound() { return 0.; }
};
}  // namespace OptionTags

namespace Tags {
template <typename ImporterOptionsGroup>
struct NumericInitialDataVariables : db::SimpleTag {
  using type = typename OptionTags::NumericInitialDataVariables<
      ImporterOptionsGroup>::type;
  using option_tags =
      tmpl::list<OptionTags::NumericInitialDataVariables<ImporterOptionsGroup>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) { return value; }
};

template <typename ImporterOptionsGroup>
struct DensityCutoff : db::SimpleTag {
  using type = double;
  using option_tags =
      tmpl::list<OptionTags::DensityCutoff<ImporterOptionsGroup>>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double value) { return value; }
};
}  // namespace Tags
}  // namespace detail

/*!
 * \brief Dispatch loading numeric initial data from files.
 *
 * Place this action before
 * grmhd::ValenciaDivClean::Actions::SetNumericInitialData in the action list.
 * See importers::Actions::ReadAllVolumeDataAndDistribute for details, which is
 * invoked by this action.
 *
 * \tparam ImporterOptionsGroup Option group in which options are placed.
 */
template <typename ImporterOptionsGroup>
struct ReadNumericInitialData {
  using const_global_cache_tags = tmpl::list<
      importers::Tags::FileGlob<ImporterOptionsGroup>,
      importers::Tags::Subgroup<ImporterOptionsGroup>,
      importers::Tags::ObservationValue<ImporterOptionsGroup>,
      importers::Tags::EnableInterpolation<ImporterOptionsGroup>,
      detail::Tags::NumericInitialDataVariables<ImporterOptionsGroup>,
      detail::Tags::DensityCutoff<ImporterOptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Select the subset of the available variables that we want to read from
    // the volume data file
    const auto& selected_initial_data_vars =
        get<detail::Tags::NumericInitialDataVariables<ImporterOptionsGroup>>(
            box);
    tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<importers::Tags::Selected, detail::all_numeric_vars>>
        selected_fields{};
    tmpl::for_each<detail::primitive_vars_option_tags>(
        [&selected_fields,
         &selected_initial_data_vars](const auto option_tag_v) {
          using option_tag =
              tmpl::type_from<std::decay_t<decltype(option_tag_v)>>;
          using tag = typename option_tag::tag;
          static constexpr bool is_required = option_tag::is_required;
          const auto& selected_dataset_name =
              get<option_tag>(selected_initial_data_vars);
          if constexpr (is_required) {
            // Always select required tags for import
            get<importers::Tags::Selected<tag>>(selected_fields) =
                selected_dataset_name;
          } else {
            // Only select optional tags for import if a dataset name was
            // specified
            if (std::holds_alternative<std::string>(selected_dataset_name)) {
              get<importers::Tags::Selected<tag>>(selected_fields) =
                  std::get<std::string>(selected_dataset_name);
            }
          }
        });
    // Dispatch loading the variables from the volume data file
    // - Not using `ckLocalBranch` here to make sure the simple action
    //   invocation is asynchronous.
    auto& reader_component = Parallel::get_parallel_component<
        importers::ElementDataReader<Metavariables>>(cache);
    Parallel::simple_action<importers::Actions::ReadAllVolumeDataAndDistribute<
        Metavariables::volume_dim, ImporterOptionsGroup,
        detail::all_numeric_vars, ParallelComponent>>(
        reader_component, std::move(selected_fields));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Receive numeric initial data loaded by
 * grmhd::ValenciaDivClean::Actions::ReadNumericInitialData.
 *
 * Place this action in the action list after
 * grmhd::ValenciaDivClean::Actions::ReadNumericInitialData to wait until the
 * data for this element has arrived, and then compute the remaining primitive
 * variables and store them in the DataBox to be used as initial data.
 *
 * This action modifies the tags listed in `hydro::grmhd_tags` in the DataBox
 * (i.e., the hydro primitives). It does not modify conservative variables, so
 * it relies on a primitive-to-conservative update in the action list before
 * the evolution can start.
 *
 * \requires This action requires that the (inverse) spatial metric is available
 * through the DataBox, so it should run after GR initial data has been loaded.
 *
 * \requires This action also requires an equation of state, which is retrieved
 * from the DataBox as `hydro::Tags::EquationOfStateBase`.
 *
 * \tparam ImporterOptionsGroup Option group in which options are placed.
 */
template <typename ImporterOptionsGroup>
struct SetNumericInitialData {
  static constexpr size_t Dim = 3;
  using inbox_tags =
      tmpl::list<importers::Tags::VolumeData<ImporterOptionsGroup,
                                             detail::all_numeric_vars>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox = tuples::get<importers::Tags::VolumeData<
        ImporterOptionsGroup, detail::all_numeric_vars>>(inboxes);
    // Using 0 for the temporal ID since we only read the volume data once, so
    // there's no need to keep track of the temporal ID.
    if (inbox.find(0_st) == inbox.end()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    auto numeric_initial_data = std::move(inbox.extract(0_st).mapped());

    const auto& selected_initial_data_vars =
        get<detail::Tags::NumericInitialDataVariables<ImporterOptionsGroup>>(
            box);
    const auto& eos = db::get<hydro::Tags::EquationOfStateBase>(box);
    const double density_cutoff =
        db::get<detail::Tags::DensityCutoff<ImporterOptionsGroup>>(box);

    db::mutate<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::ElectronFraction<DataVector>,
               hydro::Tags::SpecificInternalEnergy<DataVector>,
               hydro::Tags::SpatialVelocity<DataVector, 3>,
               hydro::Tags::MagneticField<DataVector, 3>,
               hydro::Tags::DivergenceCleaningField<DataVector>,
               hydro::Tags::LorentzFactor<DataVector>,
               hydro::Tags::Pressure<DataVector>,
               hydro::Tags::SpecificEnthalpy<DataVector>>(
        make_not_null(&box),
        [&numeric_initial_data, &selected_initial_data_vars, &eos,
         &density_cutoff](
            const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
            const gsl::not_null<Scalar<DataVector>*> electron_fraction,
            const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
            const gsl::not_null<tnsr::I<DataVector, 3>*> spatial_velocity,
            const gsl::not_null<tnsr::I<DataVector, 3>*> magnetic_field,
            const gsl::not_null<Scalar<DataVector>*> div_cleaning_field,
            const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
            const gsl::not_null<Scalar<DataVector>*> pressure,
            const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
            const tnsr::II<DataVector, 3>& inv_spatial_metric) {
          // Rest mass density from dataset
          *rest_mass_density =
              std::move(get<hydro::Tags::RestMassDensity<DataVector>>(
                  numeric_initial_data));
          const size_t num_points = get(*rest_mass_density).size();
          // Electron fraction from dataset or constant value
          const std::variant<double, std::string>& electron_fraction_selection =
              get<detail::OptionTags::VarName<
                  hydro::Tags::ElectronFraction<DataVector>,
                  std::bool_constant<false>>>(selected_initial_data_vars);
          if (std::holds_alternative<std::string>(
                  electron_fraction_selection)) {
            *electron_fraction =
                std::move(get<hydro::Tags::ElectronFraction<DataVector>>(
                    numeric_initial_data));
          } else {
            const double constant_electron_fraction =
                std::get<double>(electron_fraction_selection);
            destructive_resize_components(electron_fraction, num_points);
            get(*electron_fraction) = constant_electron_fraction;
          }
          // Velocity and Lorentz factor from u_i dataset
          // W = 1 + W^2 v_i v^i
          // where W v_i = u_i, so we first raise the index on W v_i with the
          // spatial metric and then compute its magnitude. We use
          // `spatial_velocity` as intermediate memory buffer for W v^i.
          const auto& u_i =
              get<hydro::Tags::LowerSpatialFourVelocity<DataVector, 3>>(
                  numeric_initial_data);
          raise_or_lower_index(spatial_velocity, u_i, inv_spatial_metric);
          dot_product(lorentz_factor, u_i, *spatial_velocity);
          get(*lorentz_factor) += 1.;
          for (size_t d = 0; d < 3; ++d) {
            spatial_velocity->get(d) /= get(*lorentz_factor);
          }
          // Specific internal energy, pressure, and enthalpy from EOS
          destructive_resize_components(specific_internal_energy, num_points);
          destructive_resize_components(pressure, num_points);
          destructive_resize_components(specific_enthalpy, num_points);
          for (size_t i = 0; i < num_points; ++i) {
            double& local_rest_mass_density = get(*rest_mass_density)[i];
            // Apply the EOS only where the density is above the cutoff, because
            // the EOS may be ill-defined at zero density. Also, the specific
            // enthalpy is ill-defined at zero density.
            if (local_rest_mass_density <= density_cutoff) {
              local_rest_mass_density = 0.;
              get(*specific_internal_energy)[i] = 0.;
              get(*pressure)[i] = 0.;
              get(*specific_enthalpy)[i] = 1.;
              // Also reset velocity and Lorentz factor below cutoff to be safe
              for (size_t d = 0; d < 3; ++d) {
                spatial_velocity->get(d)[i] = 0.;
              }
              get(*lorentz_factor)[i] = 1.;
            } else {
              get(*specific_internal_energy)[i] =
                  get(eos.specific_internal_energy_from_density(
                      Scalar<double>(local_rest_mass_density)));
              get(*pressure)[i] = get(eos.pressure_from_density(
                  Scalar<double>(local_rest_mass_density)));
              get(*specific_enthalpy)[i] =
                  get(hydro::relativistic_specific_enthalpy(
                      Scalar<double>(local_rest_mass_density),
                      Scalar<double>(get(*specific_internal_energy)[i]),
                      Scalar<double>(get(*pressure)[i])));
            }
          }
          // Magnetic field from dataset or constant value
          const std::variant<double, std::string>& magnetic_field_selection =
              get<detail::OptionTags::VarName<
                  hydro::Tags::MagneticField<DataVector, 3>,
                  std::bool_constant<false>>>(selected_initial_data_vars);
          if (std::holds_alternative<std::string>(magnetic_field_selection)) {
            *magnetic_field =
                std::move(get<hydro::Tags::MagneticField<DataVector, 3>>(
                    numeric_initial_data));
          } else {
            const double constant_magnetic_field =
                std::get<double>(magnetic_field_selection);
            if (constant_magnetic_field != 0.) {
              ERROR(
                  "Choose a magnetic field dataset or set it to zero. "
                  "Nonzero uniform magnetic fields cannot currently be chosen "
                  "in the input file. Generate a dataset for the nonzero "
                  "uniform magnetic field if you need to.");
            }
            destructive_resize_components(magnetic_field, num_points);
            std::fill(magnetic_field->begin(), magnetic_field->end(),
                      constant_magnetic_field);
          }
          // Divergence cleaning field
          destructive_resize_components(div_cleaning_field, num_points);
          get(*div_cleaning_field) = 0.;
        },
        db::get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>(
            box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace grmhd::ValenciaDivClean::Actions
