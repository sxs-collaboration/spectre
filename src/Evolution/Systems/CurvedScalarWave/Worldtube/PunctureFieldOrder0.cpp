// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DynamicBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

void puncture_field_0(
    gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration, const double bh_mass) {
  const size_t grid_size = get<0>(centered_coords).size();
  result->initialize(grid_size);
  const double xp = particle_position[0];
  const double yp = particle_position[1];
  const double xpdot = particle_velocity[0];
  const double ypdot = particle_velocity[1];
  const double xpddot = particle_acceleration[0];
  const double ypddot = particle_acceleration[1];
  const double rp = get(magnitude(particle_position));
  const double rpdot = (xp * xpdot + yp * ypdot) / rp;

  const auto& Dx = get<0>(centered_coords);
  const auto& Dy = get<1>(centered_coords);
  // the particle is fixed in the xy-plane, so Dz = z
  const auto& z = get<2>(centered_coords);

  const double M = bh_mass;

  DynamicBuffer<DataVector> temps(43, grid_size);

  const double d_0 = rp * rp * rp;
  const double d_1 = 2.0 * M;
  const double d_2 = yp * ypdot;
  const double d_3 = d_2 + xp * xpdot;
  const double d_4 = 1.0 / d_0;
  const double d_5 = rp * rp;
  const double d_6 = 4.0 * rp;
  const double d_7 = M * d_3;
  const double d_8 = d_3 * d_3;
  const double d_9 = xpdot * xpdot;
  const double d_10 = ypdot * ypdot;
  const double d_11 = d_10 + d_9;
  const double d_12 = d_0 * (d_11 - 1.0) + d_1 * d_5 + d_1 * d_8 + d_6 * d_7;
  const double d_13 = 1.0 / d_12;
  const double d_14 = d_13 * d_4;
  const double d_15 = xp * xp;
  const double d_16 = yp * yp;
  const double d_17 = rp * rp * rp * rp * rp;
  const double d_18 = 1.0 / d_17;
  const double d_19 = 2.0 * rp;
  const double d_20 = d_13 * d_18;
  const double d_21 = d_17 * sqrt(rp);
  const double d_22 = -d_12;
  const double d_23 = d_22 * sqrt(d_22);
  const double d_24 = sqrt(d_22);
  const double d_25 = 1. / (rp * sqrt(rp));
  const double d_26 = d_24 * d_25 * rpdot;
  const double d_27 = d_1 * xp;
  const double d_28 = d_0 * xpdot + d_27 * d_3 + d_27 * rp;
  const double d_29 = 2.0 * yp;
  const double d_30 = M * d_29 * rp + d_0 * ypdot + d_29 * d_7;
  const double d_31 = rp * rp * rp * rp;
  const double d_32 = 1.0 / d_31;
  const double d_33 = 3.0 * rpdot;
  const double d_34 = M * d_33;
  const double d_35 = 1.0 / d_5;
  const double d_36 = xp * xpddot;
  const double d_37 = yp * ypddot;
  const double d_38 = 2.0 * (d_10 + d_36 + d_37 + d_9) - rpdot;
  const double d_39 = 2.0 * rpdot;
  const double d_40 = d_11 + d_36 + d_37;
  const double d_41 = d_1 * d_3 * d_4 * (-d_39 + d_40) - d_32 * d_34 * d_8 +
                      xpddot * xpdot + ypddot * ypdot;
  const double d_42 = 1. / sqrt(rp);
  const double d_43 = d_24 * d_42;
  const double d_44 = 3.0 * M;
  const double d_45 = d_23 * d_42;
  const double d_46 = 2.0 * d_12 * d_17;
  DataVector& dv_0 = temps.at(0);
  dv_0 = Dx * Dx;
  DataVector& dv_1 = temps.at(1);
  dv_1 = Dy * ypdot;
  DataVector& dv_2 = temps.at(2);
  dv_2 = Dx * xpdot + dv_1;
  DataVector& dv_3 = temps.at(3);
  dv_3 = Dx * xp;
  DataVector& dv_4 = temps.at(4);
  dv_4 = Dy * yp;
  DataVector& dv_5 = temps.at(5);
  dv_5 = dv_3 + dv_4;
  DataVector& dv_6 = temps.at(6);
  dv_6 = d_1 * dv_5;
  DataVector& dv_7 = temps.at(7);
  dv_7 = dv_6 * rp;
  DataVector& dv_8 = temps.at(8);
  dv_8 = d_0 * dv_2 + d_3 * dv_6 + dv_7;
  DataVector& dv_9 = temps.at(9);
  dv_9 = dv_8 * dv_8;
  DataVector& dv_10 = temps.at(10);
  dv_10 = d_14 * dv_9;
  DataVector& dv_11 = temps.at(11);
  dv_11 = dv_5 * dv_5;
  DataVector& dv_12 = temps.at(12);
  dv_12 = d_4 * dv_11;
  DataVector& dv_13 = temps.at(13);
  dv_13 = Dy * Dy;
  DataVector& dv_14 = temps.at(14);
  dv_14 = z * z;
  DataVector& dv_15 = temps.at(15);
  dv_15 = dv_13 + dv_14;
  DataVector& dv_16 = temps.at(16);
  dv_16 = d_1 * dv_12 + dv_0 - dv_10 + dv_15;
  DataVector& dv_17 = temps.at(17);
  dv_17 = 2.0 * dv_0;
  DataVector& dv_18 = temps.at(18);
  dv_18 = dv_3 * dv_4;
  DataVector& dv_19 = temps.at(19);
  dv_19 = -dv_0;
  DataVector& dv_20 = temps.at(20);
  dv_20 = 2.0 * dv_14;
  DataVector& dv_21 = temps.at(21);
  dv_21 = 2.0 * dv_13 + dv_20;
  DataVector& dv_22 = temps.at(22);
  dv_22 = dv_19 + dv_21;
  DataVector& dv_23 = temps.at(13);
  dv_23 = -dv_13;
  DataVector& dv_24 = temps.at(20);
  dv_24 = dv_17 + dv_20 + dv_23;
  DataVector& dv_25 = temps.at(23);
  dv_25 = d_15 * dv_22 + d_16 * dv_24 - 6.0 * dv_18;
  DataVector& dv_26 = temps.at(24);
  dv_26 = M * dv_5;
  DataVector& dv_27 = temps.at(19);
  dv_27 = dv_15 + dv_19;
  DataVector& dv_28 = temps.at(0);
  dv_28 = dv_0 + dv_14 + dv_23;
  DataVector& dv_29 = temps.at(18);
  dv_29 = d_15 * dv_27 + d_16 * dv_28 - 4.0 * dv_18;
  DataVector& dv_30 = temps.at(13);
  dv_30 = d_19 * dv_29 + d_3 * dv_25;
  DataVector& dv_31 = temps.at(14);
  dv_31 = dv_30 * dv_8;
  DataVector& dv_32 = temps.at(15);
  dv_32 = M * dv_31;
  DataVector& dv_33 = temps.at(25);
  dv_33 = -dv_25;
  DataVector& dv_34 = temps.at(26);
  dv_34 = d_12 * dv_5;
  DataVector& dv_35 = temps.at(27);
  dv_35 = dv_33 * dv_34;
  DataVector& dv_36 = temps.at(28);
  dv_36 = dv_31 + dv_35;
  DataVector& dv_37 = temps.at(29);
  dv_37 = sqrt(dv_16);
  DataVector& dv_38 = temps.at(30);
  dv_38 = M * dv_37;
  DataVector& dv_39 = temps.at(31);
  dv_39 = d_14 * dv_8;
  DataVector& dv_40 = temps.at(32);
  dv_40 = Dx - d_28 * dv_39 + d_4 * dv_6 * xp;
  DataVector& dv_41 = temps.at(31);
  dv_41 = Dy + d_29 * d_4 * dv_26 - d_30 * dv_39;
  DataVector& dv_42 = temps.at(33);
  dv_42 = Dx * xpddot + Dy * ypddot;
  DataVector& dv_43 = temps.at(34);
  dv_43 = d_1 * dv_2;
  DataVector& dv_44 = temps.at(6);
  dv_44 = d_33 * d_5 * dv_2 + d_39 * dv_26 + d_40 * dv_6;
  DataVector& dv_45 = temps.at(34);
  dv_45 = d_32 * (-d_13 * d_33 * dv_9 +
                  d_13 * dv_8 * rp *
                      (d_0 * dv_42 + d_3 * dv_43 + dv_43 * rp + dv_44) +
                  d_31 * dv_9 * (-M * d_35 * d_38 - d_41) / (d_12 * d_12) +
                  d_34 * dv_11 - dv_2 * dv_7) +
          dv_40 * xpdot + dv_41 * ypdot;
  DataVector& dv_46 = temps.at(2);
  dv_46 = d_44 * dv_36 / dv_37;
  DataVector& dv_47 = temps.at(1);
  dv_47 = -d_2 + dv_1 + xpdot * (Dx - xp);
  DataVector& dv_48 = temps.at(7);
  dv_48 = Dy * d_16;
  DataVector& dv_49 = temps.at(11);
  dv_49 = 2.0 * Dy;
  DataVector& dv_50 = temps.at(9);
  dv_50 = -d_15 * dv_49 + 3.0 * dv_3 * yp + dv_48;
  DataVector& dv_51 = temps.at(35);
  dv_51 = 3.0 * Dy;
  DataVector& dv_52 = temps.at(36);
  dv_52 = Dx * d_15;
  DataVector& dv_53 = temps.at(37);
  dv_53 = Dx * yp;
  DataVector& dv_54 = temps.at(38);
  dv_54 = 3.0 * dv_4;
  DataVector& dv_55 = temps.at(35);
  dv_55 =
      2.0 * (dv_50 * ypdot +
             xpdot * (dv_52 - dv_53 * (d_29 + dv_51) + xp * (dv_22 + dv_54)) -
             ypdot * (-dv_24 * yp + dv_3 * dv_51));
  DataVector& dv_56 = temps.at(22);
  dv_56 = d_1 * dv_47;
  DataVector& dv_57 = temps.at(7);
  dv_57 = -Dy * d_15 + d_29 * dv_3 + dv_48;
  DataVector& dv_58 = temps.at(4);
  dv_58 = 2.0 * dv_4;
  DataVector& dv_59 = temps.at(20);
  dv_59 = 0.5 / (dv_16 * dv_16);
  DataVector& dv_60 = temps.at(39);
  dv_60 = d_46 * dv_37;
  DataVector& dv_61 = temps.at(40);
  dv_61 = Dx * d_16;
  DataVector& dv_62 = temps.at(38);
  dv_62 = dv_52 + dv_54 * xp - 2.0 * dv_61;
  DataVector& dv_63 = temps.at(41);
  dv_63 = 2.0 * dv_34;
  DataVector& dv_64 = temps.at(25);
  dv_64 = d_12 * dv_33;
  DataVector& dv_65 = temps.at(42);
  dv_65 = d_20 * dv_59;

  get(get<CurvedScalarWave::Tags::Psi>(*result)) =
      0.5 *
      (4.0 * M * dv_12 - d_18 * dv_25 * dv_26 + d_20 * dv_32 - 2.0 * dv_10 +
       dv_17 + dv_21) /
      (dv_16 * sqrt(dv_16));
  get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(*result)) =
      -dv_59 / d_21 / d_23 *
      (-2.0 * d_21 * d_23 * dv_37 * dv_45 - 7.0 * d_26 * dv_36 * dv_38 +
       d_43 * dv_45 * dv_46 +
       dv_38 * (d_23 * d_25 * d_39 * dv_25 * dv_5 - d_26 * dv_31 +
                d_43 * dv_30 *
                    (d_0 * (-d_10 - d_9 + dv_42) + d_3 * dv_56 + dv_44 +
                     dv_56 * rp) +
                d_43 * dv_8 *
                    (d_3 * dv_55 + d_39 * dv_29 + d_40 * dv_25 +
                     d_6 * (dv_57 * ypdot +
                            xpdot * (dv_52 - dv_53 * (dv_49 + yp) +
                                     xp * (dv_27 + dv_58)) -
                            ypdot * (-dv_28 * yp + dv_3 * dv_49))) +
                d_45 * dv_25 * dv_47 + d_45 * dv_5 * dv_55 +
                2.0 * dv_31 * square(rp) * sqrt(rp) * (M * d_35 * d_38 + d_41) /
                    d_24));
  get<0>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      -dv_65 *
      (dv_38 *
           (-d_28 * dv_30 - dv_62 * dv_63 - dv_64 * xp +
            2.0 * dv_8 * (d_19 * (dv_52 + dv_58 * xp - dv_61) + d_3 * dv_62)) +
       dv_40 * dv_46 + dv_40 * dv_60);
  get<1>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      -dv_65 * (dv_38 * (-d_30 * dv_30 - dv_50 * dv_63 - dv_64 * yp +
                         2.0 * dv_8 * (d_19 * dv_57 + d_3 * dv_50)) +
                dv_41 * dv_46 + dv_41 * dv_60);
  get<2>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      0.5 * d_20 * z *
      (4.0 * M * dv_16 * (d_15 + d_16) * (-dv_34 + dv_8 * (d_3 + rp)) -
       d_44 * dv_35 - d_46 * dv_16 - 3.0 * dv_32) /
      (square(dv_16) * sqrt(dv_16));
}
}  // namespace CurvedScalarWave::Worldtube
