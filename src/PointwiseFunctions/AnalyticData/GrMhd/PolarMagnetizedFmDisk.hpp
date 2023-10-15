// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedFmDisk.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/SphericalTorus.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/Transform.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {
/*!
 * \brief Magnetized fluid disk orbiting a Kerr black hole in the Kerr-Schild
 * Cartesian coordinates, but in a toroidal mesh defined from a torus map
 * (see grmhd::AnalyticData::SphericalTorus).
 */
class PolarMagnetizedFmDisk
    : public virtual evolution::initial_data::InitialData,
      public MarkAsAnalyticData,
      public AnalyticDataBase {
 public:
  struct DiskParameters {
    using type = MagnetizedFmDisk;
    static constexpr Options::String help = "Parameters for the disk.";
  };

  struct TorusParameters {
    using type = grmhd::AnalyticData::SphericalTorus;
    static constexpr Options::String help =
        "Parameters for the evolution region.";
  };

  using options = tmpl::list<DiskParameters, TorusParameters>;

  static constexpr Options::String help =
      "Magnetized Fishbone-Moncrief disk in polar coordinates.";

  PolarMagnetizedFmDisk() = default;
  PolarMagnetizedFmDisk(const PolarMagnetizedFmDisk& /*rhs*/) = default;
  PolarMagnetizedFmDisk& operator=(const PolarMagnetizedFmDisk& /*rhs*/) =
      default;
  PolarMagnetizedFmDisk(PolarMagnetizedFmDisk&& /*rhs*/) = default;
  PolarMagnetizedFmDisk& operator=(PolarMagnetizedFmDisk&& /*rhs*/) = default;
  ~PolarMagnetizedFmDisk() override = default;

  PolarMagnetizedFmDisk(MagnetizedFmDisk fm_disk,
                        grmhd::AnalyticData::SphericalTorus torus_map);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit PolarMagnetizedFmDisk(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PolarMagnetizedFmDisk);
  /// \endcond

  /// The grmhd variables.
  ///
  /// \note The functions are optimized for retrieving the hydro variables
  /// before the metric variables.
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    // In this function, we label the coordinates this solution works
    // in with Frame::BlockLogical, and the coordinates the wrapped
    // solution uses Frame::Inertial.  This means the input and output
    // have to be converted to the correct label.

    const tnsr::I<DataType, 3> observation_coordinates(torus_map_(x));

    using dependencies = tmpl::map<
        tmpl::pair<gr::AnalyticSolution<3>::DerivShift<DataType>,
                   gr::Tags::Shift<DataType, 3, Frame::Inertial>>,
        tmpl::pair<gr::AnalyticSolution<3>::DerivSpatialMetric<DataType>,
                   gr::Tags::SpatialMetric<DataType, 3, Frame::Inertial>>>;
    using required_tags = tmpl::remove_duplicates<
        tmpl::remove<tmpl::list<Tags..., tmpl::at<dependencies, Tags>...>,
                     tmpl::no_such_type_>>;

    auto observation_data =
        fm_disk_.variables(observation_coordinates, required_tags{});

    const auto jacobian = torus_map_.jacobian(x);
    const auto inv_jacobian = torus_map_.inv_jacobian(x);

    const auto change_frame = [this, &inv_jacobian, &jacobian, &x](
                                  const auto& data, auto tag) {
      using Tag = decltype(tag);
      auto result =
          transform::to_different_frame(get<Tag>(data), jacobian, inv_jacobian);

      if constexpr (std::is_same_v<
                        Tag, gr::AnalyticSolution<3>::DerivShift<DataType>>) {
        const auto deriv_inv_jacobian =
            torus_map_.derivative_of_inv_jacobian(x);
        const auto& shift =
            get<gr::Tags::Shift<DataType, 3, Frame::Inertial>>(data);
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 3; ++k) {
              result.get(i, j) +=
                  deriv_inv_jacobian.get(j, k, i) * shift.get(k);
            }
          }
        }
      } else if constexpr (std::is_same_v<
                               Tag, gr::AnalyticSolution<3>::DerivSpatialMetric<
                                        DataType>>) {
        const auto hessian = torus_map_.hessian(x);
        const auto& spatial_metric =
            get<gr::Tags::SpatialMetric<DataType, 3, Frame::Inertial>>(data);
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            for (size_t k = j; k < 3; ++k) {
              for (size_t l = 0; l < 3; ++l) {
                for (size_t m = 0; m < 3; ++m) {
                  result.get(i, j, k) +=
                      (hessian.get(l, j, i) * jacobian.get(m, k) +
                       hessian.get(l, k, i) * jacobian.get(m, j)) *
                      spatial_metric.get(l, m);
                }
              }
            }
          }
        }
      } else if constexpr (std::is_same_v<
                               Tag, gr::Tags::SqrtDetSpatialMetric<DataType>>) {
        get(result) *= abs(get(determinant(jacobian)));
      }

      typename Tag::type result_with_replaced_frame{};
      std::copy(std::move_iterator(result.begin()),
                std::move_iterator(result.end()),
                result_with_replaced_frame.begin());
      return result_with_replaced_frame;
    };

    return {change_frame(observation_data, Tags{})...};
  }

  using equation_of_state_type = MagnetizedFmDisk::equation_of_state_type;
  const equation_of_state_type& equation_of_state() const {
    return fm_disk_.equation_of_state();
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const PolarMagnetizedFmDisk& lhs,
                         const PolarMagnetizedFmDisk& rhs);

  MagnetizedFmDisk fm_disk_;
  grmhd::AnalyticData::SphericalTorus torus_map_;
};

bool operator!=(const PolarMagnetizedFmDisk& lhs,
                const PolarMagnetizedFmDisk& rhs);
}  // namespace grmhd::AnalyticData
