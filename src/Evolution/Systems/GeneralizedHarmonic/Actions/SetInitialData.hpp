// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <variant>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
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
template <size_t Dim>
void initial_gh_variables_from_adm(
    gsl::not_null<tnsr::aa<DataVector, Dim>*> spacetime_metric,
    gsl::not_null<tnsr::aa<DataVector, Dim>*> pi,
    gsl::not_null<tnsr::iaa<DataVector, Dim>*> phi,
    const tnsr::ii<DataVector, Dim>& spatial_metric,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::ii<DataVector, Dim>& extrinsic_curvature, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian);

/*!
 * \brief Numeric initial data loaded from volume data files
 *
 * This class can be factory-created in the input file to start an evolution
 * from numeric initial data. It selects the set of variables to load from
 * the volume data file (ADM or GH variables).
 */
class NumericInitialData : public evolution::initial_data::InitialData {
 public:
  /// Name of a variable in the volume data file
  template <typename Tag>
  struct VarName {
    using tag = Tag;
    static std::string name() { return db::tag_name<Tag>(); }
    using type = std::string;
    static constexpr Options::String help =
        "Name of the variable in the volume data file";
  };

  // These are the sets of variables that we support loading from volume data
  // files:
  // - ADM variables
  using adm_vars =
      tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3>>;
  struct AdmVars : tuples::tagged_tuple_from_typelist<
                       db::wrap_tags_in<VarName, adm_vars>> {
    static constexpr Options::String help =
        "ADM variables: 'Lapse', 'Shift', 'SpatialMetric' and "
        "'ExtrinsicCurvature'. The initial GH variables will be computed "
        "from these numeric fields, as well as their numeric spatial "
        "derivatives on the computational grid.";
    using options = tags_list;
    using TaggedTuple::TaggedTuple;
  };

  // - Generalized harmonic variables
  using gh_vars = tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                             Tags::Pi<DataVector, 3>>;
  struct GhVars
      : tuples::tagged_tuple_from_typelist<db::wrap_tags_in<VarName, gh_vars>> {
    static constexpr Options::String help =
        "GH variables: 'SpacetimeMetric' and 'Pi'. These variables are "
        "used to set the initial data directly; Phi is then set to the "
        "numerical derivative of SpacetimeMetric, to enforce the 3-index "
        "constraint.";
    using options = tags_list;
    using TaggedTuple::TaggedTuple;
  };

  // Collect all variables that we support loading from volume data files.
  // Remember to `tmpl::remove_duplicates` when adding overlapping sets of
  // vars.
  using all_vars = tmpl::append<adm_vars, gh_vars>;

  // Input-file options
  struct Variables {
    // The user can supply any of these choices of variables in the input
    // file
    using type = std::variant<AdmVars, GhVars>;
    static constexpr Options::String help =
        "Set of initial data variables from which the generalized harmonic "
        "system variables are computed.";
  };

  using options =
      tmpl::list<importers::OptionTags::FileGlob,
                 importers::OptionTags::Subgroup,
                 importers::OptionTags::ObservationValue,
                 importers::OptionTags::EnableInterpolation, Variables>;

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
      std::variant<AdmVars, GhVars> selected_variables);

  const importers::ImporterOptions& importer_options() const {
    return importer_options_;
  }

  const std::variant<AdmVars, GhVars>& selected_variables() const {
    return selected_variables_;
  }

  /*!
   * \brief Unique identifier for loading this volume data
   *
   * Involves a hash of the type name and the volume data file names.
   */
  size_t volume_data_id() const;

  /*!
   * \brief Selects which of the `fields` to import based on the choices in the
   * input-file options
   *
   * The `fields` are all datasets that are available to import, represented by
   * `importers::Tags::Selected<Tag>` tags. We select only those that we need by
   * setting their dataset name.
   */
  template <typename... AllTags>
  void select_for_import(
      const gsl::not_null<tuples::TaggedTuple<AllTags...>*> fields) const {
    // Select the subset of the available variables that we want to read from
    // the volume data file
    std::visit(
        [&fields](const auto& vars) {
          // This lambda is invoked with the set of vars selected in the input
          // file, which map to the tensor names that should be read from the H5
          // file
          using selected_vars = std::decay_t<decltype(vars)>;
          // Get the mapped tensor name from the input file and select it in the
          // set of all possible vars.
          tmpl::for_each<typename selected_vars::tags_list>(
              [&fields, &vars](const auto tag_v) {
                using tag = typename std::decay_t<decltype(tag_v)>::type::tag;
                get<importers::Tags::Selected<tag>>(*fields) =
                    get<VarName<tag>>(vars);
              });
        },
        selected_variables_);
  }

  /*!
   * \brief Set GH initial data given numeric data loaded from files
   *
   * The `numeric_data` contains the datasets selected above (and possibly
   * more). We either set the GH variables directly, or compute them from the
   * ADM variables.
   */
  template <typename... AllTags>
  void set_initial_data(
      const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
      const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
      const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi,
      const gsl::not_null<tuples::TaggedTuple<AllTags...>*> numeric_data,
      const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian) const {
    if (std::holds_alternative<NumericInitialData::GhVars>(
            selected_variables_)) {
      // We have loaded the GH system variables from the file, so just move the
      // data for spacetime_metric and Pi into the DataBox directly, with no
      // conversion needed. Set Phi to the spatial derivative of the spacetime
      // metric to enforce the 3-index constraint.
      *spacetime_metric = std::move(
          get<gr::Tags::SpacetimeMetric<DataVector, 3>>(*numeric_data));
      *pi = std::move(get<Tags::Pi<DataVector, 3>>(*numeric_data));
      // Set Phi to the numerical spatial derivative of spacetime_metric
      partial_derivative(phi, *spacetime_metric, mesh, inv_jacobian);
    } else if (std::holds_alternative<NumericInitialData::AdmVars>(
                   selected_variables_)) {
      // We have loaded ADM variables from the file. Convert to GH variables.
      const auto& spatial_metric =
          get<gr::Tags::SpatialMetric<DataVector, 3>>(*numeric_data);
      const auto& lapse = get<gr::Tags::Lapse<DataVector>>(*numeric_data);
      const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(*numeric_data);
      const auto& extrinsic_curvature =
          get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(*numeric_data);

      initial_gh_variables_from_adm(spacetime_metric, pi, phi, spatial_metric,
                                    lapse, shift, extrinsic_curvature, mesh,
                                    inv_jacobian);
    } else {
      ERROR(
          "These initial data variables are not implemented yet. Please add "
          "an implementation to gh::NumericInitialData.");
    }
  }

  void pup(PUP::er& p) override;

  friend bool operator==(const NumericInitialData& lhs,
                         const NumericInitialData& rhs);

 private:
  importers::ImporterOptions importer_options_;
  std::variant<AdmVars, GhVars> selected_variables_{};
};

namespace Actions {

/*!
 * \brief Dispatch loading numeric initial data from files or set analytic
 * initial data.
 *
 * Place this action before
 * gh::Actions::ReceiveNumericInitialData in the action list.
 * See importers::Actions::ReadAllVolumeDataAndDistribute for details, which is
 * invoked by this action.
 * Analytic initial data is set directly by this action and terminates the
 * phase.
 */
struct SetInitialData {
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
      const ParallelComponent* const parallel_component) {
    // Dispatch to the correct `apply` overload based on type of initial data
    using initial_data_classes =
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 evolution::initial_data::InitialData>;
    return call_with_dynamic_type<Parallel::iterable_action_return_t,
                                  initial_data_classes>(
        &db::get<evolution::initial_data::Tags::InitialData>(box),
        [&box, &cache, &parallel_component](const auto* const initial_data) {
          return apply(make_not_null(&box), *initial_data, cache,
                       parallel_component);
        });
  }

 private:
  // Numeric initial data
  template <typename DbTagsList, typename Metavariables,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      const gsl::not_null<db::DataBox<DbTagsList>*> /*box*/,
      const NumericInitialData& initial_data,
      Parallel::GlobalCache<Metavariables>& cache,
      const ParallelComponent* const /*meta*/) {
    // Select the subset of the available variables that we want to read from
    // the volume data file
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

  // "AnalyticData"-type initial data
  template <typename DbTagsList, typename InitialData, typename Metavariables,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      const gsl::not_null<db::DataBox<DbTagsList>*> box,
      const InitialData& initial_data,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ParallelComponent* const /*meta*/) {
    static constexpr size_t Dim = Metavariables::volume_dim;

    // Get ADM variables from analytic data / solution
    const auto& x =
        db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(*box);
    const auto adm_vars = evolution::Initialization::initial_data(
        initial_data, x, db::get<::Tags::Time>(*box),
        tmpl::list<gr::Tags::SpatialMetric<DataVector, Dim>,
                   gr::Tags::Lapse<DataVector>,
                   gr::Tags::Shift<DataVector, Dim>,
                   gr::Tags::ExtrinsicCurvature<DataVector, Dim>>{});
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<DataVector, Dim>>(adm_vars);
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(adm_vars);
    const auto& shift = get<gr::Tags::Shift<DataVector, Dim>>(adm_vars);
    const auto& extrinsic_curvature =
        get<gr::Tags::ExtrinsicCurvature<DataVector, Dim>>(adm_vars);

    // Compute GH vars from ADM vars
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(*box);
    const auto& inv_jacobian =
        db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                              Frame::Inertial>>(*box);
    db::mutate<gr::Tags::SpacetimeMetric<DataVector, Dim>,
               Tags::Pi<DataVector, Dim>, Tags::Phi<DataVector, Dim>>(
        &gh::initial_gh_variables_from_adm<Dim>, box, spatial_metric, lapse,
        shift, extrinsic_curvature, mesh, inv_jacobian);

    // No need to import numeric initial data, so we terminate the phase by
    // pausing the algorithm on this element
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

/*!
 * \brief Receive numeric initial data loaded by gh::Actions::SetInitialData.
 *
 * Place this action in the action list after
 * gh::Actions::SetInitialData to wait until the data
 * for this element has arrived, and then transform the data to GH variables and
 * store it in the DataBox to be used as initial data.
 *
 * This action modifies the following tags in the DataBox:
 * - gr::Tags::SpacetimeMetric<DataVector, 3>
 * - gh::Tags::Pi<DataVector, 3>
 * - gh::Tags::Phi<DataVector, 3>
 */
struct ReceiveNumericInitialData {
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
    const auto& volume_data_id = initial_data.volume_data_id();
    if (inbox.find(volume_data_id) == inbox.end()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    auto numeric_data = std::move(inbox.extract(volume_data_id).mapped());

    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& inv_jacobian =
        db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                              Frame::Inertial>>(box);

    db::mutate<gr::Tags::SpacetimeMetric<DataVector, 3>,
               Tags::Pi<DataVector, 3>, Tags::Phi<DataVector, 3>>(
        [&initial_data, &numeric_data, &mesh, &inv_jacobian](
            const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
            const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
            const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi) {
          initial_data.set_initial_data(spacetime_metric, pi, phi,
                                        make_not_null(&numeric_data), mesh,
                                        inv_jacobian);
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions
}  // namespace gh
