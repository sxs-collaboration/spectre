// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Fluxes.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/M1Closure.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace RadiationTransport::M1Grey::BoundaryConditions {

/// \cond
template <typename NeutrinoSpeciesList>
class DirichletAnalytic;
/// \endcond

/*!
 * \brief Sets Dirichlet boundary conditions using the analytic solution or
 * analytic data.
 */
template <typename... NeutrinoSpecies>
class DirichletAnalytic<tmpl::list<NeutrinoSpecies...>> final
    : public BoundaryCondition<tmpl::list<NeutrinoSpecies...>> {
 public:
  /// \brief What analytic solution/data to prescribe.
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution/data to prescribe.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };

  using options = tmpl::list<AnalyticPrescription>;
  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions using either analytic solution or "
      "analytic data."};

  DirichletAnalytic() = default;
  DirichletAnalytic(DirichletAnalytic&&) = default;
  DirichletAnalytic& operator=(DirichletAnalytic&&) = default;
  DirichletAnalytic(const DirichletAnalytic&);
  DirichletAnalytic& operator=(const DirichletAnalytic&);
  ~DirichletAnalytic() override = default;

  explicit DirichletAnalytic(CkMigrateMessage* msg)
      : BoundaryCondition<tmpl::list<NeutrinoSpecies...>>(msg) {}

  explicit DirichletAnalytic(
      std::unique_ptr<evolution::initial_data::InitialData>
          analytic_prescription);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DirichletAnalytic);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<DirichletAnalytic>(*this);
  }

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<::Tags::Time>;

  std::optional<std::string> dg_ghost(
      const gsl::not_null<typename Tags::TildeE<
          Frame::Inertial, NeutrinoSpecies>::type*>... tilde_e,
      const gsl::not_null<typename Tags::TildeS<
          Frame::Inertial, NeutrinoSpecies>::type*>... tilde_s,

      const gsl::not_null<typename ::Tags::Flux<
          Tags::TildeE<Frame::Inertial, NeutrinoSpecies>, tmpl::size_t<3>,
          Frame::Inertial>::type*>... flux_tilde_e,
      const gsl::not_null<typename ::Tags::Flux<
          Tags::TildeS<Frame::Inertial, NeutrinoSpecies>, tmpl::size_t<3>,
          Frame::Inertial>::type*>... flux_tilde_s,

      const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<
          tnsr::I<DataVector, 3, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>& /*normal_covector*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& coords, double time) const;

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
};
}  // namespace RadiationTransport::M1Grey::BoundaryConditions
