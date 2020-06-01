// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/OptionTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/Options.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace GeneralizedHarmonic::gauges::Actions {
/// Option tags for the gauges
namespace OptionTags {
/// The rollon start time
struct DampedHarmonicRollOnStart {
  using type = double;
  static std::string name() noexcept { return "RollOnStartTime"; }
  static constexpr OptionString help{
      "Simulation time to start rolling on the damped harmonic gauge"};
  using group = gauges::OptionTags::GaugeGroup;
};

/// The width of the Gaussian for the gauge rollon
struct DampedHarmonicRollOnWindow {
  using type = double;
  static std::string name() noexcept { return "RollOnTimeWindow"; }
  static constexpr OptionString help{
      "The width of the Gaussian that controls how quickly the gauge is "
      "rolled on."};
  using group = gauges::OptionTags::GaugeGroup;
};

/// The width of the Gaussian for the spatial decay of the damped harmonic
/// gauge.
template <typename Frame>
struct DampedHarmonicSpatialDecayWidth {
  using type = double;
  static std::string name() noexcept { return "SpatialDecayWidth"; }
  static constexpr OptionString help{
      "Spatial width of weight function used in the damped harmonic "
      "gauge."};
  using group = gauges::OptionTags::GaugeGroup;
};
}  // namespace OptionTags

template <size_t Dim>
struct InitializeDampedHarmonic {
 private:
  struct GaugeHRollOnStartTime : db::SimpleTag {
    using type = double;
    using option_tags = tmpl::list<OptionTags::DampedHarmonicRollOnStart>;

    static constexpr bool pass_metavariables = false;
    static double create_from_options(
        const double gauge_roll_on_start_time) noexcept {
      return gauge_roll_on_start_time;
    }
  };

  struct GaugeHRollOnTimeWindow : db::SimpleTag {
    using type = double;
    using option_tags = tmpl::list<OptionTags::DampedHarmonicRollOnWindow>;

    static constexpr bool pass_metavariables = false;
    static double create_from_options(
        const double gauge_roll_on_window) noexcept {
      return gauge_roll_on_window;
    }
  };

  template <typename Frame>
  struct GaugeHSpatialWeightDecayWidth : db::SimpleTag {
    using type = double;
    using option_tags =
        tmpl::list<OptionTags::DampedHarmonicSpatialDecayWidth<Frame>>;

    static constexpr bool pass_metavariables = false;
    static double create_from_options(
        const double gauge_spatial_decay_width) noexcept {
      return gauge_spatial_decay_width;
    }
  };

  template <typename Frame>
  struct DampedHarmonicRollonCompute : db::ComputeTag {
    static std::string name() noexcept { return "DampedHarmonicRollonCompute"; }
    using argument_tags =
        tmpl::list<Tags::InitialGaugeH<Dim, Frame>,
                   Tags::SpacetimeDerivInitialGaugeH<Dim, Frame>,
                   ::gr::Tags::Lapse<DataVector>,
                   ::gr::Tags::Shift<Dim, Frame, DataVector>,
                   ::gr::Tags::SpacetimeNormalOneForm<Dim, Frame, DataVector>,
                   ::gr::Tags::SqrtDetSpatialMetric<DataVector>,
                   ::gr::Tags::InverseSpatialMetric<Dim, Frame, DataVector>,
                   ::gr::Tags::SpacetimeMetric<Dim, Frame, DataVector>,
                   Tags::Pi<Dim, Frame>, Tags::Phi<Dim, Frame>, ::Tags::Time,
                   GaugeHRollOnStartTime, GaugeHRollOnTimeWindow,
                   domain::Tags::Coordinates<Dim, Frame>,
                   GaugeHSpatialWeightDecayWidth<Frame>>;
    using return_type =
        Variables<tmpl::list<Tags::GaugeH<Dim, Frame>,
                             Tags::SpacetimeDerivGaugeH<Dim, Frame>>>;

    static void function(
        gsl::not_null<return_type*> h_and_d4_h,
        const db::item_type<Tags::InitialGaugeH<Dim, Frame>>& gauge_h_init,
        const db::item_type<Tags::SpacetimeDerivInitialGaugeH<Dim, Frame>>&
            dgauge_h_init,
        const Scalar<DataVector>& lapse,
        const tnsr::I<DataVector, Dim, Frame>& shift,
        const tnsr::a<DataVector, Dim, Frame>& spacetime_unit_normal_one_form,
        const Scalar<DataVector>& sqrt_det_spatial_metric,
        const tnsr::II<DataVector, Dim, Frame>& inverse_spatial_metric,
        const tnsr::aa<DataVector, Dim, Frame>& spacetime_metric,
        const tnsr::aa<DataVector, Dim, Frame>& pi,
        const tnsr::iaa<DataVector, Dim, Frame>& phi, double time,
        double rollon_start_time, double rollon_width,
        const tnsr::I<DataVector, Dim, Frame>& coords, double sigma_r) noexcept;
  };

 public:
  using frame = Frame::Inertial;

  using const_global_cache_tags =
      tmpl::list<GaugeHRollOnStartTime, GaugeHRollOnTimeWindow,
                 GaugeHSpatialWeightDecayWidth<frame>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if (not db::get<domain::CoordinateMaps::Tags::CoordinateMap<
                Metavariables::volume_dim, Frame::Grid, Frame::Inertial>>(box)
                .is_identity()) {
      ERROR(
          "Cannot use the damped harmonic rollon gauge with a moving mesh "
          "because the rollon is not implemented for a moving mesh. The issue "
          "is that the initial H_a needs to be in the grid frame, and then "
          "transformed to the inertial frame at each time step. Transforming "
          "the spacetime derivative requires spacetime Hessians, which are not "
          "implemented for the maps and there is currently no plan to add them "
          "because we do not need them for anything else.");
    }
    const auto& inverse_jacobian =
        db::get<domain::Tags::InverseJacobian<Dim, Frame::Logical, frame>>(box);

    auto [initial_gauge_h, initial_d4_gauge_h] = impl(
        db::get<gr::Tags::SpacetimeMetric<Dim, Frame::Inertial, DataVector>>(
            box),
        db::get<GeneralizedHarmonic::Tags::Pi<Dim, Frame::Inertial>>(box),
        db::get<GeneralizedHarmonic::Tags::Phi<Dim, Frame::Inertial>>(box),
        db::get<domain::Tags::Mesh<Dim>>(box), inverse_jacobian);

    // Add gauge tags
    using compute_tags = db::AddComputeTags<DampedHarmonicRollonCompute<frame>>;

    // Finally, insert gauge related quantities to the box
    return std::make_tuple(
        Initialization::merge_into_databox<
            InitializeDampedHarmonic,
            db::AddSimpleTags<
                GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>,
                GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<Dim,
                                                                       frame>>,
            compute_tags>(std::move(box), std::move(initial_gauge_h),
                          std::move(initial_d4_gauge_h)));
  }

 private:
  static std::tuple<tnsr::a<DataVector, Dim, Frame::Inertial>,
                    tnsr::ab<DataVector, Dim, Frame::Inertial>>
  impl(const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
       const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
       const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
       const Mesh<Dim>& mesh,
       const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
           inverse_jacobian) noexcept;
};
}  // namespace GeneralizedHarmonic::gauges::Actions
