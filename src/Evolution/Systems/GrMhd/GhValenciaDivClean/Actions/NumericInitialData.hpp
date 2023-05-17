// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <variant>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/NumericInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Actions/NumericInitialData.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Options.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd::GhValenciaDivClean {

/*!
 * \brief Numeric initial data loaded from volume data files
 *
 * This class can be factory-created in the input file to start an evolution
 * from numeric initial data. It selects the set of GH variables to load from
 * the volume data file (ADM or GH variables), the set of hydro variables, and
 * allows to set constant values for some of the hydro variables. This class
 * mostly combines the `gh::NumericInitialData` and
 * `grmhd::ValenciaDivClean::NumericInitialData` classes.
 */
class NumericInitialData : public evolution::initial_data::InitialData {
 private:
  using GhNumericId = gh::NumericInitialData;
  using HydroNumericId = grmhd::ValenciaDivClean::NumericInitialData;

 public:
  using all_vars =
      tmpl::append<GhNumericId::all_vars, HydroNumericId::all_vars>;

  struct GhVariables : GhNumericId::Variables {};
  struct HydroVariables : HydroNumericId::Variables {};

  using options =
      tmpl::list<importers::OptionTags::FileGlob,
                 importers::OptionTags::Subgroup,
                 importers::OptionTags::ObservationValue,
                 importers::OptionTags::EnableInterpolation, GhVariables,
                 HydroVariables, HydroNumericId::DensityCutoff>;

  static constexpr Options::String help =
      "Numeric initial data loaded from volume data files";

  NumericInitialData() = default;
  NumericInitialData(const NumericInitialData& rhs) = default;
  NumericInitialData& operator=(const NumericInitialData& rhs) = default;
  NumericInitialData(NumericInitialData&& /*rhs*/) = default;
  NumericInitialData& operator=(NumericInitialData&& /*rhs*/) = default;
  ~NumericInitialData() = default;

  /// \cond
  explicit NumericInitialData(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NumericInitialData);
  /// \endcond

  std::unique_ptr<evolution::initial_data::InitialData> get_clone()
      const override {
    return std::make_unique<NumericInitialData>(*this);
  }

  NumericInitialData(
      std::string file_glob, std::string subfile_name,
      std::variant<double, importers::ObservationSelector> observation_value,
      bool enable_interpolation,
      typename GhNumericId::Variables::type gh_selected_variables,
      typename HydroNumericId::Variables::type hydro_selected_variables,
      double density_cutoff);

  const importers::ImporterOptions& importer_options() const {
    return gh_numeric_id_.importer_options();
  }

  const GhNumericId& gh_numeric_id() const { return gh_numeric_id_; }

  const HydroNumericId& hydro_numeric_id() const { return hydro_numeric_id_; }

  size_t volume_data_id() const;

  template <typename... AllTags>
  void select_for_import(
      const gsl::not_null<tuples::TaggedTuple<AllTags...>*> fields) const {
    gh_numeric_id_.select_for_import(fields);
    hydro_numeric_id_.select_for_import(fields);
  }

  template <typename... AllTags>
  void set_initial_data(
      const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
      const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
      const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi,
      const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      const gsl::not_null<Scalar<DataVector>*> electron_fraction,
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      const gsl::not_null<tnsr::I<DataVector, 3>*> spatial_velocity,
      const gsl::not_null<tnsr::I<DataVector, 3>*> magnetic_field,
      const gsl::not_null<Scalar<DataVector>*> div_cleaning_field,
      const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      const gsl::not_null<Scalar<DataVector>*> pressure,
      const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
      const gsl::not_null<tuples::TaggedTuple<AllTags...>*> numeric_data,
      const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const EquationsOfState::EquationOfState<true, 1>& equation_of_state)
      const {
    gh_numeric_id_.set_initial_data(spacetime_metric, pi, phi, numeric_data,
                                    mesh, inv_jacobian);
    const auto spatial_metric = gr::spatial_metric(*spacetime_metric);
    const auto inv_spatial_metric =
        determinant_and_inverse(spatial_metric).second;
    hydro_numeric_id_.set_initial_data(
        rest_mass_density, electron_fraction, specific_internal_energy,
        spatial_velocity, magnetic_field, div_cleaning_field, lorentz_factor,
        pressure, specific_enthalpy, numeric_data, inv_spatial_metric,
        equation_of_state);
  }

  void pup(PUP::er& p) override;

  friend bool operator==(const NumericInitialData& lhs,
                         const NumericInitialData& rhs);

 private:
  GhNumericId gh_numeric_id_{};
  HydroNumericId hydro_numeric_id_{};
};

namespace Actions {

/*!
 * \brief Dispatch loading numeric initial data from files.
 *
 * Place this action before
 * grmhd::GhValenciaDivClean::Actions::SetNumericInitialData in the action list.
 * See importers::Actions::ReadAllVolumeDataAndDistribute for details, which is
 * invoked by this action.
 */
struct ReadNumericInitialData {
  using const_global_cache_tags =
      tmpl::list<evolution::initial_data::Tags::InitialData>;

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
    const auto& initial_data = dynamic_cast<const NumericInitialData&>(
        db::get<evolution::initial_data::Tags::InitialData>(box));
    tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
        importers::Tags::Selected, NumericInitialData::all_vars>>
        selected_fields{};
    initial_data.select_for_import(make_not_null(&selected_fields));
    // Dispatch loading the variables from the volume data file
    // - Not using `ckLocalBranch` here to make sure the simple action
    //   invocation is asynchronous.
    auto& reader_component = Parallel::get_parallel_component<
        importers::ElementDataReader<Metavariables>>(cache);
    Parallel::simple_action<importers::Actions::ReadAllVolumeDataAndDistribute<
        3, NumericInitialData::all_vars, ParallelComponent>>(
        reader_component, initial_data.importer_options(),
        initial_data.volume_data_id(), std::move(selected_fields));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Receive numeric initial data loaded by
 * grmhd::GhValenciaDivClean::Actions::ReadNumericInitialData.
 *
 * Place this action in the action list after
 * grmhd::GhValenciaDivClean::Actions::ReadNumericInitialData to wait until the
 * data for this element has arrived, and then compute the GH variables and the
 * remaining primitive variables and store them in the DataBox to be used as
 * initial data.
 *
 * This action modifies the GH system tags (spacetime metric, pi, phi) and the
 * tags listed in `hydro::grmhd_tags` in the DataBox (i.e., the hydro
 * primitives). It does not modify conservative variables, so it relies on a
 * primitive-to-conservative update in the action list before the evolution can
 * start.
 *
 * \requires This action requires an equation of state, which is retrieved from
 * the DataBox as `hydro::Tags::EquationOfStateBase`.
 */
struct SetNumericInitialData {
  static constexpr size_t Dim = 3;
  using inbox_tags =
      tmpl::list<importers::Tags::VolumeData<NumericInitialData::all_vars>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox =
        tuples::get<importers::Tags::VolumeData<NumericInitialData::all_vars>>(
            inboxes);
    const auto& initial_data = dynamic_cast<const NumericInitialData&>(
        db::get<evolution::initial_data::Tags::InitialData>(box));
    const size_t volume_data_id = initial_data.volume_data_id();
    if (inbox.find(volume_data_id) == inbox.end()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    auto numeric_data = std::move(inbox.extract(volume_data_id).mapped());

    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& inv_jacobian =
        db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                              Frame::Inertial>>(box);
    const auto& equation_of_state =
        db::get<hydro::Tags::EquationOfStateBase>(box);

    db::mutate<gr::Tags::SpacetimeMetric<DataVector, 3>,
               gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>,
               hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::ElectronFraction<DataVector>,
               hydro::Tags::SpecificInternalEnergy<DataVector>,
               hydro::Tags::SpatialVelocity<DataVector, 3>,
               hydro::Tags::MagneticField<DataVector, 3>,
               hydro::Tags::DivergenceCleaningField<DataVector>,
               hydro::Tags::LorentzFactor<DataVector>,
               hydro::Tags::Pressure<DataVector>,
               hydro::Tags::SpecificEnthalpy<DataVector>>(
        make_not_null(&box),
        [&initial_data, &numeric_data, &mesh, &inv_jacobian,
         &equation_of_state](
            const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
            const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
            const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi,
            const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
            const gsl::not_null<Scalar<DataVector>*> electron_fraction,
            const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
            const gsl::not_null<tnsr::I<DataVector, 3>*> spatial_velocity,
            const gsl::not_null<tnsr::I<DataVector, 3>*> magnetic_field,
            const gsl::not_null<Scalar<DataVector>*> div_cleaning_field,
            const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
            const gsl::not_null<Scalar<DataVector>*> pressure,
            const gsl::not_null<Scalar<DataVector>*> specific_enthalpy) {
          initial_data.set_initial_data(
              spacetime_metric, pi, phi, rest_mass_density, electron_fraction,
              specific_internal_energy, spatial_velocity, magnetic_field,
              div_cleaning_field, lorentz_factor, pressure, specific_enthalpy,
              make_not_null(&numeric_data), mesh, inv_jacobian,
              equation_of_state);
        });

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions

}  // namespace grmhd::GhValenciaDivClean
