// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace gh {

/*!
 * \brief Compute initial GH variables from ADM variables
 *
 * - The spacetime metric is assembled from the spatial metric, lapse, and
 *   shift. See `gr::spacetime_metric` for details.
 * - Phi is set to the numerical derivative of the spacetime metric. This
 *   ensures that the 3-index constraint is initially satisfied.
 * - Pi is computed by choosing the time derivatives of lapse and shift to be
 *   zero. The `gh::gauges::SetPiFromGauge` mutator exists to override Pi later
 *   in the algorithm (it should be combined with this function).
 */
void initial_gh_variables_from_adm(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
    const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi,
    const tnsr::ii<DataVector, 3>& spatial_metric,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3>& extrinsic_curvature, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian);

namespace detail {
namespace OptionTags {
template <typename Tag>
struct VarName {
  using tag = Tag;
  static std::string name() { return db::tag_name<Tag>(); }
  using type = std::string;
  static constexpr Options::String help =
      "Name of the variable in the volume data file";
};
}  // namespace OptionTags

// These are the sets of variables that we support loading from volume data
// files:
// - ADM variables
using adm_vars =
    tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
               gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
               gr::Tags::ExtrinsicCurvature<DataVector, 3>>;
struct Adm : tuples::tagged_tuple_from_typelist<
                 db::wrap_tags_in<OptionTags::VarName, adm_vars>> {
  using Base = tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<OptionTags::VarName, adm_vars>>;
  static constexpr Options::String help =
      "ADM variables: 'Lapse', 'Shift', 'SpatialMetric' and "
      "'ExtrinsicCurvature'. The initial GH variables will be computed from "
      "these numeric fields, as well as their numeric spatial derivatives on "
      "the computational grid.";
  using options = db::wrap_tags_in<OptionTags::VarName, adm_vars>;
  using Base::TaggedTuple;
};
// - Generalized harmonic variables
using gh_vars = tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                           Tags::Pi<3, Frame::Inertial>>;
struct GeneralizedHarmonic
    : tuples::tagged_tuple_from_typelist<
          db::wrap_tags_in<OptionTags::VarName, gh_vars>> {
  using Base = tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<OptionTags::VarName, gh_vars>>;
  static constexpr Options::String help =
      "GH variables: 'SpacetimeMetric' and 'Pi'. These variables are "
      "used to set the initial data directly; Phi is then set to the numerical "
      "derivative of SpacetimeMetric, to enforce the 3-index constraint.";
  using options = db::wrap_tags_in<OptionTags::VarName, gh_vars>;
  using Base::TaggedTuple;
};

// Collect all variables that we support loading from volume data files.
// Remember to `tmpl::remove_duplicates` when adding overlapping sets of vars.
using all_numeric_vars = tmpl::append<adm_vars, gh_vars>;

namespace OptionTags {
template <typename ImporterOptionsGroup>
struct NumericInitialDataVariables {
  static std::string name() { return "Variables"; }
  // The user can supply any of these choices of variables in the input file
  using type = std::variant<Adm, GeneralizedHarmonic>;
  static constexpr Options::String help =
      "Set of initial data variables from which the generalized harmonic "
      "system variables are computed.";
  using group = ImporterOptionsGroup;
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
}  // namespace Tags
}  // namespace detail

namespace Actions {

/*!
 * \brief Dispatch loading numeric initial data from files.
 *
 * Place this action before gh::Actions::SetNumericInitialData
 * in the action list. See importers::Actions::ReadAllVolumeDataAndDistribute
 * for details, which is invoked by this action.
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
      detail::Tags::NumericInitialDataVariables<ImporterOptionsGroup>>;

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
    std::visit(
        [&selected_fields](const auto& vars) {
          // This lambda is invoked with the set of vars selected in the input
          // file, which map to the tensor names that should be read from the H5
          // file
          using selected_vars = std::decay_t<decltype(vars)>;
          // Get the mapped tensor name from the input file and select it in the
          // set of all possible vars.
          tmpl::for_each<typename selected_vars::tags_list>(
              [&selected_fields, &vars](const auto tag_v) {
                using tag = typename std::decay_t<decltype(tag_v)>::type::tag;
                get<importers::Tags::Selected<tag>>(selected_fields) =
                    get<detail::OptionTags::VarName<tag>>(vars);
              });
        },
        selected_initial_data_vars);
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
 * gh::Actions::ReadNumericInitialData.
 *
 * Place this action in the action list after
 * gh::Actions::ReadNumericInitialData to wait until the data
 * for this element has arrived, and then transform the data to GH variables and
 * store it in the DataBox to be used as initial data.
 *
 * This action modifies the following tags in the DataBox:
 * - gr::Tags::SpacetimeMetric<DataVector, 3>
 * - gh::Tags::Pi<3, Frame::Inertial>
 * - gh::Tags::Phi<3, Frame::Inertial>
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

    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& inv_jacobian =
        db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                              Frame::Inertial>>(box);

    const auto& selected_initial_data_vars =
        get<detail::Tags::NumericInitialDataVariables<ImporterOptionsGroup>>(
            box);
    if (std::holds_alternative<detail::GeneralizedHarmonic>(
            selected_initial_data_vars)) {
      // We have loaded the GH system variables from the file, so just move the
      // data for spacetime_metric and Pi into the DataBox directly, with no
      // conversion needed. Set Phi to the spatial derivative of the spacetime
      // metric to enforce the 3-index constraint.
      db::mutate<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 Tags::Pi<3, Frame::Inertial>, Tags::Phi<3, Frame::Inertial>>(
          make_not_null(&box),
          [&numeric_initial_data, &mesh, &inv_jacobian](
              const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
              const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
              const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi) {
            *spacetime_metric =
                std::move(get<gr::Tags::SpacetimeMetric<DataVector, 3>>(
                    numeric_initial_data));
            *pi = std::move(
                get<Tags::Pi<3, Frame::Inertial>>(numeric_initial_data));
            // Set Phi to the numerical spatial derivative of spacetime_metric
            partial_derivative(phi, *spacetime_metric, mesh, inv_jacobian);
          });
    } else if (std::holds_alternative<detail::Adm>(
                   selected_initial_data_vars)) {
      // We have loaded ADM variables from the file. Convert to GH variables.
      const auto& spatial_metric =
          get<gr::Tags::SpatialMetric<DataVector, 3>>(numeric_initial_data);
      const auto& lapse =
          get<gr::Tags::Lapse<DataVector>>(numeric_initial_data);
      const auto& shift =
          get<gr::Tags::Shift<DataVector, 3>>(numeric_initial_data);
      const auto& extrinsic_curvature =
          get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(
              numeric_initial_data);

      db::mutate<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 Tags::Pi<3, Frame::Inertial>, Tags::Phi<3, Frame::Inertial>>(
          make_not_null(&box), &initial_gh_variables_from_adm, spatial_metric,
          lapse, shift, extrinsic_curvature, mesh, inv_jacobian);
    } else {
      ERROR(
          "These initial data variables are not implemented yet. Please add "
          "an implementation to "
          "gh::Actions::SetNumericInitialData.");
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions
}  // namespace gh
