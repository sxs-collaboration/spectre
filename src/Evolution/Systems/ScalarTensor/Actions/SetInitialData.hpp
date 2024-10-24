// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <variant>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/NumericInitialData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Actions/SetInitialData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Actions/SetInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ScalarTensor {

/*!
 * \brief Numeric initial data loaded from volume data files
 */
class NumericInitialData : public evolution::initial_data::InitialData,
                           public evolution::NumericInitialData {
 private:
  using GhNumericId = gh::NumericInitialData;
  using ScalarNumericId = CurvedScalarWave::NumericInitialData;

 public:
  using all_vars =
      tmpl::append<GhNumericId::all_vars, ScalarNumericId::all_vars>;

  struct GhVariables : GhNumericId::Variables {};
  struct ScalarVariables : ScalarNumericId::Variables {};

  using options = tmpl::list<
      importers::OptionTags::FileGlob, importers::OptionTags::Subgroup,
      importers::OptionTags::ObservationValue,
      importers::OptionTags::EnableInterpolation, GhVariables, ScalarVariables>;

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
      typename ScalarNumericId::Variables::type hydro_selected_variables);

  const importers::ImporterOptions& importer_options() const {
    return gh_numeric_id_.importer_options();
  }

  const GhNumericId& gh_numeric_id() const { return gh_numeric_id_; }

  const ScalarNumericId& scalar_numeric_id() const {
    return scalar_numeric_id_;
  }

  size_t volume_data_id() const;

  template <typename... AllTags>
  void select_for_import(
      const gsl::not_null<tuples::TaggedTuple<AllTags...>*> fields) const {
    gh_numeric_id_.select_for_import(fields);
    scalar_numeric_id_.select_for_import(fields);
  }

  template <typename... AllTags>
  void set_initial_data(
      const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
      const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
      const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi,
      const gsl::not_null<Scalar<DataVector>*> psi_scalar,
      const gsl::not_null<Scalar<DataVector>*> pi_scalar,
      const gsl::not_null<tnsr::i<DataVector, 3>*> phi_scalar,
      const gsl::not_null<tuples::TaggedTuple<AllTags...>*> numeric_data,
      const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian) const {
    gh_numeric_id_.set_initial_data(spacetime_metric, pi, phi, numeric_data,
                                    mesh, inv_jacobian);
    scalar_numeric_id_.set_initial_data(psi_scalar, pi_scalar, phi_scalar,
                                        numeric_data, mesh, inv_jacobian);
  }

  void pup(PUP::er& p) override;

  friend bool operator==(const NumericInitialData& lhs,
                         const NumericInitialData& rhs);

 private:
  GhNumericId gh_numeric_id_{};
  ScalarNumericId scalar_numeric_id_{};
};

namespace Actions {

/*!
 * \brief Dispatch loading numeric initial data from files.
 *
 * Place this action before
 * ScalarTensor::Actions::SetNumericInitialData in the action list.
 * See importers::Actions::ReadAllVolumeDataAndDistribute for details, which is
 * invoked by this action.
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
  static constexpr size_t Dim = 3;

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

  //   // "AnalyticData"-type initial data
  //   template <typename DbTagsList, typename InitialData, typename
  //   Metavariables,
  //             typename ParallelComponent>
  //   static Parallel::iterable_action_return_t apply(
  //       const gsl::not_null<db::DataBox<DbTagsList>*> box,
  //       const InitialData& initial_data,
  //       Parallel::GlobalCache<Metavariables>& /*cache*/,
  //       const ParallelComponent* const /*meta*/) {
  //     // Get ADM + hydro variables from analytic data / solution
  //     const auto& [coords, mesh, inv_jacobian] = [&box]() {
  //       if constexpr (db::tag_is_retrievable_v<
  //                         evolution::dg::subcell::Tags::ActiveGrid,
  //                         db::DataBox<DbTagsList>>) {
  //         const bool on_subcell =
  //             db::get<evolution::dg::subcell::Tags::ActiveGrid>(*box) ==
  //             evolution::dg::subcell::ActiveGrid::Subcell;
  //         if (on_subcell) {
  //           return std::forward_as_tuple(
  //               db::get<evolution::dg::subcell::Tags::Coordinates<
  //                   Dim, Frame::Inertial>>(*box),
  //               db::get<evolution::dg::subcell::Tags::Mesh<Dim>>(*box),
  //               db::get<evolution::dg::subcell::fd::Tags::
  //                           InverseJacobianLogicalToInertial<Dim>>(*box));
  //         }
  //       }
  //       return std::forward_as_tuple(
  //           db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(*box),
  //           db::get<domain::Tags::Mesh<Dim>>(*box),
  //           db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
  //                                                 Frame::Inertial>>(*box));
  //     }();
  //     auto vars = evolution::Initialization::initial_data(
  //         initial_data, coords, db::get<::Tags::Time>(*box),
  //         tmpl::append<tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
  //                                 gr::Tags::Lapse<DataVector>,
  //                                 gr::Tags::Shift<DataVector, 3>,
  //                                 gr::Tags::ExtrinsicCurvature<DataVector,
  //                                 3>>,
  //                      hydro::grmhd_tags<DataVector>>{});
  //     const auto& spatial_metric =
  //         get<gr::Tags::SpatialMetric<DataVector, 3>>(vars);
  //     const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  //     const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(vars);
  //     const auto& extrinsic_curvature =
  //         get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(vars);

  //     // Compute GH vars from ADM vars
  //     db::mutate<gr::Tags::SpacetimeMetric<DataVector, 3>,
  //                gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>(
  //         &gh::initial_gh_variables_from_adm<3>, box, spatial_metric, lapse,
  //         shift, extrinsic_curvature, mesh, inv_jacobian);

  //     // Move hydro vars directly into the DataBox
  //     tmpl::for_each<hydro::grmhd_tags<DataVector>>(
  //         [&box, &vars](const auto tag_v) {
  //           using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
  //           Initialization::mutate_assign<tmpl::list<tag>>(
  //               box, std::move(get<tag>(vars)));
  //         });

  //     // No need to import numeric initial data, so we terminate the phase by
  //     // pausing the algorithm on this element
  //     return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  //   }
  };

  /*!
   * \brief Receive numeric initial data loaded by
   * ScalarTensor::Actions::SetInitialData.
   */
  struct ReceiveNumericInitialData {
    static constexpr size_t Dim = 3;
    using inbox_tags =
        tmpl::list<importers::Tags::VolumeData<NumericInitialData::all_vars>>;

    template <typename DbTagsList, typename... InboxTags,
              typename Metavariables, typename ActionList,
              typename ParallelComponent>
    static Parallel::iterable_action_return_t apply(
        db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
      auto& inbox = tuples::get<
          importers::Tags::VolumeData<NumericInitialData::all_vars>>(inboxes);
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

      db::mutate<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>,
                 CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                 CurvedScalarWave::Tags::Phi<3>>(
          [&initial_data, &numeric_data, &mesh, &inv_jacobian](
              const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
              const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
              const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi,
              const gsl::not_null<Scalar<DataVector>*> psi_scalar,
              const gsl::not_null<Scalar<DataVector>*> pi_scalar,
              const gsl::not_null<tnsr::i<DataVector, 3>*> phi_scalar) {
            initial_data.set_initial_data(spacetime_metric, pi, phi,

                                          psi_scalar, pi_scalar, phi_scalar,

                                          make_not_null(&numeric_data), mesh,
                                          inv_jacobian);
          },
          make_not_null(&box));

      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
  };

}  // namespace Actions

}  // namespace ScalarTensor
