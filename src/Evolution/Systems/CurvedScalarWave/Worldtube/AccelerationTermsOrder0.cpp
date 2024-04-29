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

void acceleration_terms_0(
    gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration, const double ft,
    const double fx, const double fy, const double dt_ft, const double dt_fx,
    const double dt_fy, const double bh_mass) {
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
  const auto& z = get<2>(centered_coords);

  const double M = bh_mass;

  DynamicBuffer<DataVector> temps(56, grid_size);

  const double d_0 = rp * rp * rp;
  const double d_1 = 1.0 / d_0;
  const double d_2 = 2. * M;
  const double d_3 = d_1 * d_2;
  const double d_4 = xp * xpdot;
  const double d_5 = yp * ypdot;
  const double d_6 = d_4 + d_5;
  const double d_7 = rp * rp;
  const double d_8 = M * rp;
  const double d_9 = 4. * d_8;
  const double d_10 = d_6 * d_6;
  const double d_11 = xpdot * xpdot;
  const double d_12 = ypdot * ypdot;
  const double d_13 = d_11 + d_12;
  const double d_14 = d_0 * (d_13 - 1.) + d_10 * d_2 + d_2 * d_7 + d_6 * d_9;
  const double d_15 = 1.0 / d_14;
  const double d_16 = d_1 * d_15;
  const double d_17 = d_2 * yp;
  const double d_18 = d_0 + d_2 * (yp * yp);
  const double d_19 = d_0 * d_14;
  const double d_20 = rp * rp * rp * rp;
  const double d_21 = d_20 * ft;
  const double d_22 = rp * rp * rp * rp * rp * rp;
  const double d_23 = 0.5 * d_15 / d_22;
  const double d_24 = -d_14;
  const double d_25 = sqrt(d_24);
  const double d_26 = 1.0 / d_20;
  const double d_27 = 3. * rpdot;
  const double d_28 = M * d_27;
  const double d_29 = xp * xpddot;
  const double d_30 = yp * ypddot;
  const double d_31 = 2. * rpdot;
  const double d_32 = d_13 + d_29 + d_30;
  const double d_33 =
      -2. * M * (-d_11 - d_12 - d_29 - d_30 + 0.5 * rpdot) / d_7 -
      d_10 * d_26 * d_28 + d_3 * d_6 * (-d_31 + d_32) + xpddot * xpdot +
      ypddot * ypdot;
  const double d_34 = d_0 * xpdot;
  const double d_35 = d_2 * xp;
  const double d_36 = d_35 * rp;
  const double d_37 = d_34 + d_35 * d_6 + d_36;
  const double d_38 = d_0 * ypdot;
  const double d_39 = d_17 * rp;
  const double d_40 = d_17 * d_6 + d_38 + d_39;
  const double d_41 = d_27 * d_7;
  const double d_42 = 1. / sqrt(rp);
  const double d_43 = -d_5;
  const double d_44 = xp * xp;
  const double d_45 = d_35 * yp;
  const double d_46 = d_14 * d_21;
  const double d_47 = d_0 + d_2 * d_44;
  DataVector& dv_0 = temps.at(0);
  dv_0 = Dx * xp;
  DataVector& dv_1 = temps.at(1);
  dv_1 = Dy * yp;
  DataVector& dv_2 = temps.at(2);
  dv_2 = dv_0 + dv_1;
  DataVector& dv_3 = temps.at(3);
  dv_3 = dv_2 * dv_2;
  DataVector& dv_4 = temps.at(4);
  dv_4 = Dx * xpdot;
  DataVector& dv_5 = temps.at(5);
  dv_5 = Dy * ypdot;
  DataVector& dv_6 = temps.at(6);
  dv_6 = dv_4 + dv_5;
  DataVector& dv_7 = temps.at(7);
  dv_7 = d_2 * dv_2;
  DataVector& dv_8 = temps.at(8);
  dv_8 = dv_7 * rp;
  DataVector& dv_9 = temps.at(9);
  dv_9 = d_0 * dv_6 + d_6 * dv_7 + dv_8;
  DataVector& dv_10 = temps.at(10);
  dv_10 = dv_9 * dv_9;
  DataVector& dv_11 = temps.at(11);
  dv_11 = -d_16 * dv_10 + d_3 * dv_3 + Dx * Dx + Dy * Dy + z * z;
  DataVector& dv_12 = temps.at(12);
  dv_12 = dv_11 * sqrt(dv_11);
  DataVector& dv_13 = temps.at(13);
  dv_13 = dv_7 * xp;
  DataVector& dv_14 = temps.at(14);
  dv_14 = Dx * d_0 + dv_13;
  DataVector& dv_15 = temps.at(15);
  dv_15 = dv_10 * fx;
  DataVector& dv_16 = temps.at(16);
  dv_16 = dv_14 * dv_15;
  DataVector& dv_17 = temps.at(0);
  dv_17 = Dy * d_18 + d_17 * dv_0;
  DataVector& dv_18 = temps.at(17);
  dv_18 = dv_10 * fy;
  DataVector& dv_19 = temps.at(18);
  dv_19 = dv_17 * dv_18;
  DataVector& dv_20 = temps.at(19);
  dv_20 = dv_10 * ft;
  DataVector& dv_21 = temps.at(20);
  dv_21 = dv_14 * fx;
  DataVector& dv_22 = temps.at(21);
  dv_22 = d_19 * dv_11;
  DataVector& dv_23 = temps.at(22);
  dv_23 = dv_17 * fy;
  DataVector& dv_24 = temps.at(23);
  dv_24 = d_14 * dv_11;
  DataVector& dv_25 = temps.at(24);
  dv_25 = d_21 * dv_24;
  DataVector& dv_26 = temps.at(25);
  dv_26 = dv_21 * dv_22 + dv_22 * dv_23 + dv_25 * dv_7;
  DataVector& dv_27 = temps.at(26);
  dv_27 = dv_16 + dv_19 + dv_20 * dv_8 + dv_26;
  DataVector& dv_28 = temps.at(27);
  dv_28 = 1.0 / (dv_11 * dv_11);
  DataVector& dv_29 = temps.at(28);
  dv_29 = sqrt(dv_11);
  DataVector& dv_30 = temps.at(29);
  dv_30 = d_16 * dv_9;
  DataVector& dv_31 = temps.at(13);
  dv_31 = Dx + d_1 * dv_13 - d_37 * dv_30;
  DataVector& dv_32 = temps.at(29);
  dv_32 = Dy + d_1 * dv_7 * yp - d_40 * dv_30;
  DataVector& dv_33 = temps.at(30);
  dv_33 = Dx * xpddot + Dy * ypddot;
  DataVector& dv_34 = temps.at(31);
  dv_34 = d_2 * dv_6;
  DataVector& dv_35 = temps.at(32);
  dv_35 = M * dv_2;
  DataVector& dv_36 = temps.at(33);
  dv_36 = d_31 * dv_35;
  DataVector& dv_37 = temps.at(34);
  dv_37 = d_32 * dv_7 + d_41 * dv_6 + dv_36;
  DataVector& dv_38 = temps.at(31);
  dv_38 = -d_26 * (2. * M * dv_2 * dv_6 * rp + 3. * d_15 * dv_10 * rpdot -
                   d_15 * dv_9 * rp *
                       (d_0 * dv_33 + d_6 * dv_34 + dv_34 * rp + dv_37) +
                   d_20 * d_33 * dv_10 / (d_14 * d_14) - d_28 * dv_3) +
          dv_31 * xpdot + dv_32 * ypdot;
  DataVector& dv_39 = temps.at(3);
  dv_39 = 1.0 / dv_29;
  DataVector& dv_40 = temps.at(35);
  dv_40 = dv_14 * dt_fx;
  DataVector& dv_41 = temps.at(6);
  dv_41 = d_2 * (-d_4 + d_43 + dv_6);
  DataVector& dv_42 = temps.at(36);
  dv_42 = dv_41 * rp;
  DataVector& dv_43 = temps.at(37);
  dv_43 = dv_17 * dt_fy;
  DataVector& dv_44 = temps.at(1);
  dv_44 =
      Dx * d_41 +
      d_2 * (-d_44 * xpdot + dv_1 * xpdot + xp * (d_43 + 2. * dv_4 + dv_5)) -
      d_34;
  DataVector& dv_45 = temps.at(5);
  dv_45 = Dy * d_41 + d_17 * (d_43 + dv_4 + 2. * dv_5) +
          d_35 * (Dx * ypdot - xpdot * yp) - d_38;
  DataVector& dv_46 = temps.at(34);
  dv_46 = dv_9 * (d_0 * (-d_11 - d_12 + dv_33) + d_6 * dv_41 + dv_37 + dv_42);
  DataVector& dv_47 = temps.at(30);
  dv_47 = d_9 * dv_2 * ft;
  DataVector& dv_48 = temps.at(4);
  dv_48 = 2 * dv_21;
  DataVector& dv_49 = temps.at(38);
  dv_49 = 2 * dv_23;
  DataVector& dv_50 = temps.at(39);
  dv_50 = 6 * d_7 * dv_24 * rpdot;
  DataVector& dv_51 = temps.at(40);
  dv_51 = d_33 * dv_11;
  DataVector& dv_52 = temps.at(41);
  dv_52 = d_22 * dv_51;
  DataVector& dv_53 = temps.at(42);
  dv_53 = dv_20 * dv_29;
  DataVector& dv_54 = temps.at(43);
  dv_54 = dv_18 * dv_29;
  DataVector& dv_55 = temps.at(44);
  dv_55 = d_46 * dv_12;
  DataVector& dv_56 = temps.at(45);
  dv_56 = d_19 * dv_12;
  DataVector& dv_57 = temps.at(46);
  dv_57 = dv_56 * fy;
  DataVector& dv_58 = temps.at(47);
  dv_58 = dv_15 * dv_29;
  DataVector& dv_59 = temps.at(45);
  dv_59 = dv_56 * fx;
  DataVector& dv_60 = temps.at(9);
  dv_60 = dv_29 * dv_9;
  DataVector& dv_61 = temps.at(48);
  dv_61 = d_37 * dv_60;
  DataVector& dv_62 = temps.at(49);
  dv_62 = dv_31 * dv_39;
  DataVector& dv_63 = temps.at(50);
  dv_63 = 6. * d_8 * dv_2 * dv_20;
  DataVector& dv_64 = temps.at(51);
  dv_64 = d_46 * dv_29 * dv_7;
  DataVector& dv_65 = temps.at(16);
  dv_65 = 3. * dv_16;
  DataVector& dv_66 = temps.at(18);
  dv_66 = 3. * dv_19 + dv_63 + dv_65;
  DataVector& dv_67 = temps.at(52);
  dv_67 = d_19 * dv_29;
  DataVector& dv_68 = temps.at(53);
  dv_68 = dv_31 * dv_67;
  DataVector& dv_69 = temps.at(54);
  dv_69 = d_23 * dv_28;
  DataVector& dv_70 = temps.at(9);
  dv_70 = d_40 * dv_60;
  DataVector& dv_71 = temps.at(55);
  dv_71 = dv_32 * dv_39;
  DataVector& dv_72 = temps.at(52);
  dv_72 = dv_32 * dv_67;

  get(get<CurvedScalarWave::Tags::Psi>(*result)) = -d_23 * dv_27 / dv_12;
  get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(*result)) =
      -0.5 * dv_28 *
      (-3. * d_25 * d_42 * dv_27 * dv_38 * dv_39 +
       d_25 * d_42 * dv_29 *
           (2. * d_14 * dv_38 *
                (2. * M * d_20 * dv_2 * ft + d_0 * dv_14 * fx +
                 d_0 * dv_17 * fy) -
            d_20 * dv_24 * dv_7 * dt_ft - dv_10 * dv_40 - dv_10 * dv_43 -
            dv_10 * dv_8 * dt_ft - dv_15 * dv_44 - dv_18 * dv_45 -
            dv_20 * dv_36 - dv_20 * dv_42 - dv_21 * dv_50 -
            14 * dv_22 * dv_35 * ft * rpdot - dv_22 * dv_40 - dv_22 * dv_43 -
            dv_22 * dv_44 * fx - dv_22 * dv_45 * fy - dv_23 * dv_50 -
            dv_25 * dv_41 - 4 * dv_35 * dv_51 * ft * rp * square(cube(rp)) -
            dv_46 * dv_47 - dv_46 * dv_48 - dv_46 * dv_49 - dv_48 * dv_52 -
            dv_49 * dv_52) +
       9. * d_25 * dv_27 * dv_29 * rpdot / (rp * sqrt(rp)) -
       2. * d_33 * dv_27 * dv_29 * rp * rp * sqrt(rp) / d_25) /
      (d_24 * sqrt(d_24) * rp * square(square(rp)) * sqrt(rp));
  get<0>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      -dv_69 * (d_35 * dv_55 + d_36 * dv_53 + d_45 * dv_54 + d_45 * dv_57 +
                d_47 * dv_58 + d_47 * dv_59 - dv_21 * dv_68 - dv_23 * dv_68 -
                dv_31 * dv_64 + dv_47 * dv_61 + dv_48 * dv_61 + dv_49 * dv_61 -
                dv_62 * dv_66);
  get<1>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      -dv_69 * (d_17 * dv_55 + d_18 * dv_54 + d_18 * dv_57 + d_39 * dv_53 +
                d_45 * dv_58 + d_45 * dv_59 - dv_21 * dv_72 - dv_23 * dv_72 -
                dv_32 * dv_64 + dv_47 * dv_70 + dv_48 * dv_70 + dv_49 * dv_70 -
                dv_66 * dv_71);
  get<2>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      d_23 * z * (dv_26 + dv_66) / (dv_11 * dv_11 * sqrt(dv_11));
}
}  // namespace CurvedScalarWave::Worldtube
