// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DhGaugeParameters.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/DhGaugeParameters.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Options.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace GeneralizedHarmonic::gauges::Actions {
/*!
 * \brief Initialize the damped harmonic gauge, either with or without a rollon
 * function.
 *
 * See `GeneralizedHarmonic::gauges::damped_harmonic()` for details on the
 * condition.
 *
 * When the non-rollon gauge is initialized, Eq. 9 and 10 of
 * \cite Lindblom2005qh are used to compute the time derivatives for the lapse
 * and shift to compute the new \f$\Pi_{ab}\f$.
 *
 * \warning The rollon gauge does not support moving meshes.
 *
 * When the rollon gauge is used:
 *
 * DataBox:
 * - Uses:
 *   - `domain::Tags::Domain<Dim, Frame::Inertial>`
 *   - `domain::CoordinateMaps::Tags::CoordinateMap<Dim,
 *      Frame::Grid, Frame::Inertial>`
 *   - `domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
 *      Frame::Inertial>`
 *   - `domain::Tags::Mesh<Dim>`
 *   - `gr::Tags::SpacetimeMetric<Dim, Frame::Inertial>`
 *   - `GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>`
 *   - `GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>`
 * - Adds:
 *   - `GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Frame::Inertial>`
 *   - `GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<Dim,
 *      Frame::Inertial>`
 *   - `GeneralizedHarmonic::Tags::GaugeH<Dim, Frame::Inertial>`
 *   - `GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim, Frame::Inertial>`
 * - Removes: nothing
 * - Modifies: nothing
 *
 * When the non-rollon gauge is used:
 *
 * DataBox:
 * - Uses:
 *   - `Initialization::Tags::InitialTime`
 *   - `domain::Tags::ElementMap<Dim, Frame::Grid>`
 *   - `domain::CoordinateMaps::Tags::CoordinateMap<Dim,
 *      Frame::Grid, Frame::Inertial>`
 *   - `domain::Tags::Coordinates<Dim, Frame::ElementLogical>`
 *   - `domain::Tags::FunctionsOfTime`
 *   - `gr::Tags::SpacetimeMetric<Dim, Frame::Inertial>`
 *   - `GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>`
 *   - `GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>`
 * - Adds:
 *   - `GeneralizedHarmonic::Tags::GaugeH<Dim, Frame::Inertial>`
 *   - `GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim, Frame::Inertial>`
 * - Removes: nothing
 * - Modifies:
 *   - `GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <size_t Dim, bool UseRollon>
struct InitializeDampedHarmonic {
 private:
  template <typename Frame>
  struct DampedHarmonicRollonCompute
      : db::ComputeTag,
        ::Tags::Variables<tmpl::list<
            ::GeneralizedHarmonic::Tags::GaugeH<Dim, Frame>,
            ::GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim, Frame>>> {
    using base = ::Tags::Variables<tmpl::list<
        ::GeneralizedHarmonic::Tags::GaugeH<Dim, Frame>,
        ::GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim, Frame>>>;
    static std::string name() noexcept { return "DampedHarmonicRollonCompute"; }
    using argument_tags = tmpl::list<
        ::GeneralizedHarmonic::Tags::InitialGaugeH<Dim, Frame>,
        ::GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<Dim, Frame>,
        ::gr::Tags::Lapse<DataVector>,
        ::gr::Tags::Shift<Dim, Frame, DataVector>,
        ::gr::Tags::SpacetimeNormalOneForm<Dim, Frame, DataVector>,
        ::gr::Tags::SqrtDetSpatialMetric<DataVector>,
        ::gr::Tags::InverseSpatialMetric<Dim, Frame, DataVector>,
        ::gr::Tags::SpacetimeMetric<Dim, Frame, DataVector>,
        ::GeneralizedHarmonic::Tags::Pi<Dim, Frame>,
        ::GeneralizedHarmonic::Tags::Phi<Dim, Frame>, ::Tags::Time,
        domain::Tags::Coordinates<Dim, Frame>,
        ::GeneralizedHarmonic::gauges::Tags::DhGaugeParameters<true>>;
    using return_type = Variables<tmpl::list<
        ::GeneralizedHarmonic::Tags::GaugeH<Dim, Frame>,
        ::GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim, Frame>>>;

    static void function(
        gsl::not_null<return_type*> h_and_d4_h,
        const tnsr::a<DataVector, Dim, Frame>& gauge_h_init,
        const tnsr::ab<DataVector, Dim, Frame>& dgauge_h_init,
        const Scalar<DataVector>& lapse,
        const tnsr::I<DataVector, Dim, Frame>& shift,
        const tnsr::a<DataVector, Dim, Frame>& spacetime_unit_normal_one_form,
        const Scalar<DataVector>& sqrt_det_spatial_metric,
        const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
        const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
        const tnsr::aa<DataVector, Dim, Frame>& pi,
        const tnsr::iaa<DataVector, Dim, Frame>& phi, double time,
        const tnsr::I<DataVector, Dim, Frame>& coords,
        const GeneralizedHarmonic::gauges::DhGaugeParameters<true>&
            parameters) noexcept;
  };

  template <typename Frame>
  struct DampedHarmonicCompute
      : db::ComputeTag,
        ::Tags::Variables<tmpl::list<
            ::GeneralizedHarmonic::Tags::GaugeH<Dim, Frame>,
            ::GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim, Frame>>> {
    using base = ::Tags::Variables<tmpl::list<
        ::GeneralizedHarmonic::Tags::GaugeH<Dim, Frame>,
        ::GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim, Frame>>>;
    static std::string name() noexcept { return "DampedHarmonicCompute"; }
    using argument_tags = tmpl::list<
        ::gr::Tags::Lapse<DataVector>,
        ::gr::Tags::Shift<Dim, Frame, DataVector>,
        ::gr::Tags::SpacetimeNormalOneForm<Dim, Frame, DataVector>,
        ::gr::Tags::SqrtDetSpatialMetric<DataVector>,
        ::gr::Tags::InverseSpatialMetric<Dim, Frame, DataVector>,
        ::gr::Tags::SpacetimeMetric<Dim, Frame, DataVector>,
        ::GeneralizedHarmonic::Tags::Pi<Dim, Frame>,
        ::GeneralizedHarmonic::Tags::Phi<Dim, Frame>,
        domain::Tags::Coordinates<Dim, Frame>,
        ::GeneralizedHarmonic::gauges::Tags::DhGaugeParameters<false>>;
    using return_type = Variables<tmpl::list<
        ::GeneralizedHarmonic::Tags::GaugeH<Dim, Frame>,
        ::GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim, Frame>>>;

    static void function(
        gsl::not_null<return_type*> h_and_d4_h, const Scalar<DataVector>& lapse,
        const tnsr::I<DataVector, Dim, Frame>& shift,
        const tnsr::a<DataVector, Dim, Frame>& spacetime_unit_normal_one_form,
        const Scalar<DataVector>& sqrt_det_spatial_metric,
        const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
        const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
        const tnsr::aa<DataVector, Dim, Frame>& pi,
        const tnsr::iaa<DataVector, Dim, Frame>& phi,
        const tnsr::I<DataVector, Dim, Frame>& coords,
        const GeneralizedHarmonic::gauges::DhGaugeParameters<false>&
            parameters) noexcept;
  };

 public:
  using frame = Frame::Inertial;

  using const_global_cache_tags = tmpl::list<
      GeneralizedHarmonic::gauges::Tags::DhGaugeParameters<UseRollon>>;

  using simple_tags = tmpl::conditional_t<
      UseRollon,
      tmpl::list<
          GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>,
          GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<Dim, frame>>,
      tmpl::list<>>;

  using compute_tags =
      tmpl::conditional_t<UseRollon, DampedHarmonicRollonCompute<frame>,
                          DampedHarmonicCompute<frame>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if constexpr (UseRollon) {
      if (not db::get<domain::CoordinateMaps::Tags::CoordinateMap<
                  Metavariables::volume_dim, Frame::Grid, Frame::Inertial>>(box)
                  .is_identity()) {
        ERROR(
            "Cannot use the damped harmonic rollon gauge with a moving mesh "
            "because the rollon is not implemented for a moving mesh. The "
            "issue is that the initial H_a needs to be in the grid frame, and "
            "then transformed to the inertial frame at each time step. "
            "Transforming the spacetime derivative requires spacetime "
            "Hessians, which are not implemented for the maps and there is "
            "currently no plan to add them because we do not need them for "
            "anything else.");
      }
      const auto& inverse_jacobian = db::get<
          domain::Tags::InverseJacobian<Dim, Frame::ElementLogical, frame>>(
          box);

      auto [initial_gauge_h, initial_d4_gauge_h] = impl_rollon(
          db::get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
              box),
          db::get<GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>>(box),
          db::get<GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>(box),
          db::get<domain::Tags::Mesh<Dim>>(box), inverse_jacobian);

      // Finally, update gauge related quantities in the box
      Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 std::move(initial_gauge_h),
                                                 std::move(initial_d4_gauge_h));
      return std::make_tuple(std::move(box));
    } else {
      const double initial_time =
          db::get<::Initialization::Tags::InitialTime>(box);
      const auto inertial_coords = db::get<
          ::domain::CoordinateMaps::Tags::CoordinateMap<
              Metavariables::volume_dim, Frame::Grid, Frame::Inertial>>(box)(
          db::get<::domain::Tags::ElementMap<Metavariables::volume_dim,
                                             Frame::Grid>>(box)(
              db::get<domain::Tags::Coordinates<Dim, Frame::ElementLogical>>(
                  box)),
          initial_time, db::get<::domain::Tags::FunctionsOfTime>(box));

      db::mutate<GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>>(
          make_not_null(&box), &new_pi_from_gauge_h,
          db::get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
              box),
          db::get<GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>(box),
          inertial_coords,
          db::get<
              GeneralizedHarmonic::gauges::Tags::DhGaugeParameters<UseRollon>>(
              box));

      // Add gauge tags
      return std::make_tuple(std::move(box));
    }
  }

 private:
  static std::tuple<tnsr::a<DataVector, Dim, Frame::Inertial>,
                    tnsr::ab<DataVector, Dim, Frame::Inertial>>
  impl_rollon(
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
      const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian) noexcept;

  static void new_pi_from_gauge_h(
      gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*> pi,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const GeneralizedHarmonic::gauges::DhGaugeParameters<false>&
          parameters) noexcept;
};
}  // namespace GeneralizedHarmonic::gauges::Actions
