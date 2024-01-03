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

// NOLINTNEXTLINE(google-readability-function-size, readability-function-size)
void puncture_field_1(
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

  DynamicBuffer<DataVector> temps(320, grid_size);

  const double d_0 = rp * rp * rp;
  const double d_1 = 2.0 * M;
  const double d_2 = xp * xpdot;
  const double d_3 = yp * ypdot;
  const double d_4 = d_2 + d_3;
  const double d_5 = 1.0 / d_0;
  const double d_6 = rp * rp;
  const double d_7 = 4.0 * M;
  const double d_8 = d_4 * d_4;
  const double d_9 = ypdot * ypdot;
  const double d_10 = xpdot * xpdot;
  const double d_11 = d_10 - 1;
  const double d_12 =
      d_0 * (d_11 + d_9) + d_1 * d_6 + d_1 * d_8 + d_4 * d_7 * rp;
  const double d_13 = 1.0 / d_12;
  const double d_14 = d_13 * d_5;
  const double d_15 = rp * rp * rp * rp * rp;
  const double d_16 = 1.0 / d_15;
  const double d_17 = xp * xp;
  const double d_18 = yp * yp;
  const double d_19 = 2.0 * rp;
  const double d_20 = rp * rp * rp * rp * rp * rp * rp * rp * rp * rp;
  const double d_21 = 1.0 / d_20;
  const double d_22 = rp * rp * rp * rp * rp * rp * rp;
  const double d_23 = 1.0 / (d_12 * d_12);
  const double d_24 = rp * rp * rp * rp * rp * rp;
  const double d_25 = rp * rp * rp * rp;
  const double d_26 = xp * yp;
  const double d_27 = 2.0 * xpdot;
  const double d_28 = d_27 * ypdot;
  const double d_29 = rp * rp * rp * rp * rp * rp * rp * rp * rp;
  const double d_30 = M * d_13;
  const double d_31 = 2.0 * yp;
  const double d_32 = M * d_31;
  const double d_33 = d_6 * ypdot;
  const double d_34 = 3.0 * d_3;
  const double d_35 = 3.0 * yp;
  const double d_36 = d_1 * ypdot;
  const double d_37 = -d_35 + d_36;
  const double d_38 = 8.0 * d_30;
  const double d_39 = rp * rp * rp * rp * rp * rp * rp * rp;
  const double d_40 = M * M;
  const double d_41 = 2.0 * d_40;
  const double d_42 = 3.0 * d_30 * d_6;
  const double d_43 = yp * yp * yp;
  const double d_44 = xp * xp * xp;
  const double d_45 = xp * xp * xp * xp;
  const double d_46 = yp * yp * yp * yp;
  const double d_47 = d_17 * d_18;
  const double d_48 = 4.0 * d_6;
  const double d_49 = d_17 * yp;
  const double d_50 = M * rp;
  const double d_51 = 2.0 * xp;
  const double d_52 = d_18 * xp;
  const double d_53 = 3.0 * d_44;
  const double d_54 = 8.0 * d_6;
  const double d_55 = 6.0 * d_25;
  const double d_56 = M * d_0;
  const double d_57 = d_0 * d_13;
  const double d_58 = 1.0 / d_6;
  const double d_59 = 1.0 / d_24;
  const double d_60 = M * d_4;
  const double d_61 = d_51 * d_60;
  const double d_62 = d_0 * xpdot + d_50 * d_51 + d_61;
  const double d_63 = d_32 * d_4;
  const double d_64 = d_0 * ypdot + d_32 * rp + d_63;
  const double d_65 = 1.0 / d_25;
  const double d_66 = 3.0 * rpdot;
  const double d_67 = ypddot * ypdot;
  const double d_68 = xp * xpddot;
  const double d_69 = yp * ypddot;
  const double d_70 = 2.0 * d_9;
  const double d_71 = 2.0 * rpdot;
  const double d_72 = d_10 + d_9;
  const double d_73 = d_68 + d_69 + d_72;
  const double d_74 = d_1 * d_4;
  const double d_75 =
      -M * d_58 * (-2 * d_10 - 2.0 * d_68 - 2.0 * d_69 - d_70 + rpdot) -
      3.0 * M * d_65 * d_8 * rpdot + d_5 * d_74 * (-d_71 + d_73) + d_67 +
      xpddot * xpdot;
  const double d_76 = -d_75;
  const double d_77 = d_6 * d_66;
  const double d_78 = M * d_16;
  const double d_79 = 36.0 * d_78;
  const double d_80 = -yp;
  const double d_81 = -d_9;
  const double d_82 = 4.0 * rp;
  const double d_83 = d_20 * rp;
  const double d_84 = 3.0 * M;
  const double d_85 = 12.0 * d_15;
  const double d_86 = d_23 * d_76;
  const double d_87 = d_18 * ypddot;
  const double d_88 = d_9 + 1;
  const double d_89 = d_71 * rp;
  const double d_90 = 2.0 * d_18;
  const double d_91 = ypdot * ypdot * ypdot;
  const double d_92 = M * d_69;
  const double d_93 = M * d_9;
  const double d_94 = d_18 * d_67;
  const double d_95 = d_18 * d_9;
  const double d_96 = 8.0 * d_29;
  const double d_97 = 2.0 * d_6;
  const double d_98 = 6.0 * M;
  const double d_99 = d_0 * rpdot;
  const double d_100 = d_13 * d_25;
  const double d_101 = 4.0 * yp;
  const double d_102 = rp * rpdot;
  const double d_103 = 2.0 * ypdot;
  const double d_104 = 3.0 * d_52;
  const double d_105 = 16.0 * d_40;
  const double d_106 = d_105 * d_14;
  const double d_107 = M * d_58;
  const double d_108 = 8.0 * d_107;
  const double d_109 = 1.0 / rp;
  const double d_110 = d_21 * d_40;
  const double d_111 = 45.0 * d_110;
  const double d_112 = 1.0 / d_29;
  const double d_113 = d_112 * d_84;
  const double d_114 = d_1 * d_25;
  const double d_115 = d_17 + d_18;
  const double d_116 = d_10 * (d_17 - d_90) + 2.0 * d_2 * (d_1 + d_34) -
                       ypdot * (d_103 * d_17 - yp * (d_3 + d_7));
  const double d_117 = d_4 + rp;
  DataVector& dv_0 = temps.at(0);
  dv_0 = Dx * xpdot;
  DataVector& dv_1 = temps.at(1);
  dv_1 = Dy * ypdot;
  DataVector& dv_2 = temps.at(2);
  dv_2 = dv_0 + dv_1;
  DataVector& dv_3 = temps.at(3);
  dv_3 = d_0 * dv_2;
  DataVector& dv_4 = temps.at(4);
  dv_4 = Dx * xp;
  DataVector& dv_5 = temps.at(5);
  dv_5 = Dy * yp;
  DataVector& dv_6 = temps.at(6);
  dv_6 = dv_4 + dv_5;
  DataVector& dv_7 = temps.at(7);
  dv_7 = d_1 * dv_6;
  DataVector& dv_8 = temps.at(8);
  dv_8 = d_4 * dv_7;
  DataVector& dv_9 = temps.at(9);
  dv_9 = dv_3 + dv_7 * rp + dv_8;
  DataVector& dv_10 = temps.at(10);
  dv_10 = dv_9 * dv_9;
  DataVector& dv_11 = temps.at(11);
  dv_11 = d_14 * dv_10;
  DataVector& dv_12 = temps.at(12);
  dv_12 = dv_6 * dv_6;
  DataVector& dv_13 = temps.at(13);
  dv_13 = d_1 * dv_12;
  DataVector& dv_14 = temps.at(14);
  dv_14 = Dx * Dx;
  DataVector& dv_15 = temps.at(15);
  dv_15 = Dy * Dy;
  DataVector& dv_16 = temps.at(16);
  dv_16 = z * z;
  DataVector& dv_17 = temps.at(17);
  dv_17 = dv_15 + dv_16;
  DataVector& dv_18 = temps.at(18);
  dv_18 = dv_14 + dv_17;
  DataVector& dv_19 = temps.at(19);
  dv_19 = d_5 * dv_13 - dv_11 + dv_18;
  DataVector& dv_20 = temps.at(20);
  dv_20 = sqrt(dv_19);
  DataVector& dv_21 = temps.at(21);
  dv_21 = 1.0 / dv_20;
  DataVector& dv_22 = temps.at(22);
  dv_22 = 6.0 * dv_5;
  DataVector& dv_23 = temps.at(23);
  dv_23 = dv_22 * dv_4;
  DataVector& dv_24 = temps.at(24);
  dv_24 = -dv_14;
  DataVector& dv_25 = temps.at(25);
  dv_25 = 2.0 * dv_15;
  DataVector& dv_26 = temps.at(26);
  dv_26 = 2.0 * dv_16;
  DataVector& dv_27 = temps.at(27);
  dv_27 = dv_25 + dv_26;
  DataVector& dv_28 = temps.at(28);
  dv_28 = dv_24 + dv_27;
  DataVector& dv_29 = temps.at(29);
  dv_29 = -dv_15;
  DataVector& dv_30 = temps.at(30);
  dv_30 = 2.0 * dv_14;
  DataVector& dv_31 = temps.at(31);
  dv_31 = dv_26 + dv_30;
  DataVector& dv_32 = temps.at(32);
  dv_32 = dv_29 + dv_31;
  DataVector& dv_33 = temps.at(33);
  dv_33 = d_17 * dv_28 + d_18 * dv_32 - dv_23;
  DataVector& dv_34 = temps.at(34);
  dv_34 = dv_33 * dv_6;
  DataVector& dv_35 = temps.at(35);
  dv_35 = 4.0 * dv_5;
  DataVector& dv_36 = temps.at(36);
  dv_36 = dv_17 + dv_24;
  DataVector& dv_37 = temps.at(37);
  dv_37 = dv_14 + dv_16;
  DataVector& dv_38 = temps.at(38);
  dv_38 = dv_29 + dv_37;
  DataVector& dv_39 = temps.at(39);
  dv_39 = d_17 * dv_36 + d_18 * dv_38 - dv_35 * dv_4;
  DataVector& dv_40 = temps.at(40);
  dv_40 = d_19 * dv_39;
  DataVector& dv_41 = temps.at(41);
  dv_41 = -dv_33;
  DataVector& dv_42 = temps.at(42);
  dv_42 = -d_4 * dv_41 + dv_40;
  DataVector& dv_43 = temps.at(43);
  dv_43 = d_13 * dv_42 * dv_9 - dv_34;
  DataVector& dv_44 = temps.at(44);
  dv_44 = 1.0 / dv_19;
  DataVector& dv_45 = temps.at(45);
  dv_45 = M * dv_44;
  DataVector& dv_46 = temps.at(46);
  dv_46 = 4.0 * dv_16;
  DataVector& dv_47 = temps.at(47);
  dv_47 = d_23 * dv_10;
  DataVector& dv_48 = temps.at(48);
  dv_48 = d_22 * dv_47;
  DataVector& dv_49 = temps.at(49);
  dv_49 = 8.0 * dv_16;
  DataVector& dv_50 = temps.at(50);
  dv_50 = M * dv_47;
  DataVector& dv_51 = temps.at(51);
  dv_51 = d_24 * dv_50;
  DataVector& dv_52 = temps.at(52);
  dv_52 = Dx * Dy;
  DataVector& dv_53 = temps.at(53);
  dv_53 = -d_26 * dv_16 + d_6 * dv_52;
  DataVector& dv_54 = temps.at(54);
  dv_54 = Dx * yp;
  DataVector& dv_55 = temps.at(55);
  dv_55 = Dy * xp;
  DataVector& dv_56 = temps.at(56);
  dv_56 = d_17 * dv_17;
  DataVector& dv_57 = temps.at(57);
  dv_57 = d_18 * dv_15 + dv_56;
  DataVector& dv_58 = temps.at(58);
  dv_58 = d_17 * dv_14;
  DataVector& dv_59 = temps.at(59);
  dv_59 = d_10 * dv_57 + d_9 * (d_18 * dv_37 + dv_58) + square(dv_54 - dv_55);
  DataVector& dv_60 = temps.at(60);
  dv_60 = -d_28 * dv_53 + dv_59;
  DataVector& dv_61 = temps.at(61);
  dv_61 = 8.0 * dv_47;
  DataVector& dv_62 = temps.at(62);
  dv_62 = M * dv_61;
  DataVector& dv_63 = temps.at(63);
  dv_63 = d_13 * dv_19;
  DataVector& dv_64 = temps.at(64);
  dv_64 = d_20 * dv_63;
  DataVector& dv_65 = temps.at(65);
  dv_65 = d_30 * dv_19;
  DataVector& dv_66 = temps.at(66);
  dv_66 = d_29 * dv_65;
  DataVector& dv_67 = temps.at(67);
  dv_67 = -dv_26;
  DataVector& dv_68 = temps.at(68);
  dv_68 = dv_14 + dv_67;
  DataVector& dv_69 = temps.at(69);
  dv_69 = dv_29 + dv_30;
  DataVector& dv_70 = temps.at(70);
  dv_70 = d_9 * dv_68 + dv_69;
  DataVector& dv_71 = temps.at(71);
  dv_71 = -dv_25;
  DataVector& dv_72 = temps.at(72);
  dv_72 = d_9 * dv_37;
  DataVector& dv_73 = temps.at(73);
  dv_73 = dv_72 * yp;
  DataVector& dv_74 = temps.at(74);
  dv_74 = yp * (d_7 * dv_37 * ypdot + dv_73 - yp * (dv_14 + dv_71));
  DataVector& dv_75 = temps.at(56);
  dv_75 = d_18 * (dv_15 + dv_67) + dv_56;
  DataVector& dv_76 = temps.at(75);
  dv_76 = d_10 * dv_75;
  DataVector& dv_77 = temps.at(52);
  dv_77 = dv_52 * (d_32 + d_33) - xp * (d_1 * dv_17 + d_34 * dv_16);
  DataVector& dv_78 = temps.at(76);
  dv_78 = d_27 * dv_77;
  DataVector& dv_79 = temps.at(77);
  dv_79 = 2.0 * Dy;
  DataVector& dv_80 = temps.at(78);
  dv_80 = dv_4 * dv_79;
  DataVector& dv_81 = temps.at(79);
  dv_81 = d_37 * dv_80;
  DataVector& dv_82 = temps.at(80);
  dv_82 = -dv_74 - dv_76 + dv_78 + dv_81;
  DataVector& dv_83 = temps.at(81);
  dv_83 = -d_17 * dv_70 + dv_82;
  DataVector& dv_84 = temps.at(82);
  dv_84 = 4.0 * dv_83;
  DataVector& dv_85 = temps.at(83);
  dv_85 = d_15 * dv_47;
  DataVector& dv_86 = temps.at(84);
  dv_86 = d_39 * dv_63;
  DataVector& dv_87 = temps.at(85);
  dv_87 = dv_41 * dv_6;
  DataVector& dv_88 = temps.at(86);
  dv_88 = -dv_42;
  DataVector& dv_89 = temps.at(87);
  dv_89 = d_13 * dv_9;
  DataVector& dv_90 = temps.at(88);
  dv_90 = dv_88 * dv_89;
  DataVector& dv_91 = temps.at(89);
  dv_91 = dv_87 - dv_90;
  DataVector& dv_92 = temps.at(90);
  dv_92 = dv_91 * dv_91;
  DataVector& dv_93 = temps.at(91);
  dv_93 = dv_6 * dv_6 * dv_6 * dv_6;
  DataVector& dv_94 = temps.at(92);
  dv_94 = d_41 * dv_93;
  DataVector& dv_95 = temps.at(93);
  dv_95 = 3.0 * dv_14;
  DataVector& dv_96 = temps.at(94);
  dv_96 = -dv_95;
  DataVector& dv_97 = temps.at(95);
  dv_97 = 4.0 * dv_15;
  DataVector& dv_98 = temps.at(96);
  dv_98 = dv_46 + dv_97;
  DataVector& dv_99 = temps.at(97);
  dv_99 = dv_96 + dv_98;
  DataVector& dv_100 = temps.at(98);
  dv_100 = 3.0 * dv_15;
  DataVector& dv_101 = temps.at(99);
  dv_101 = -dv_100;
  DataVector& dv_102 = temps.at(100);
  dv_102 = 4.0 * dv_14;
  DataVector& dv_103 = temps.at(101);
  dv_103 = dv_102 + dv_46;
  DataVector& dv_104 = temps.at(102);
  dv_104 = dv_101 + dv_103;
  DataVector& dv_105 = temps.at(103);
  dv_105 = -14 * Dx * Dy * xp * yp + d_17 * dv_99 + d_18 * dv_104;
  DataVector& dv_106 = temps.at(104);
  dv_106 = M * dv_12;
  DataVector& dv_107 = temps.at(105);
  dv_107 = dv_106 * rp;
  DataVector& dv_108 = temps.at(106);
  dv_108 = Dy * dv_38;
  DataVector& dv_109 = temps.at(107);
  dv_109 = 30.0 * d_43 * dv_108 * dv_4;
  DataVector& dv_110 = temps.at(108);
  dv_110 = Dx * d_44;
  DataVector& dv_111 = temps.at(109);
  dv_111 = dv_110 * dv_5;
  DataVector& dv_112 = temps.at(110);
  dv_112 = 30.0 * dv_111;
  DataVector& dv_113 = temps.at(111);
  dv_113 = dv_112 * dv_36;
  DataVector& dv_114 = temps.at(112);
  dv_114 = Dx * Dx * Dx * Dx;
  DataVector& dv_115 = temps.at(113);
  dv_115 = 2.0 * dv_114;
  DataVector& dv_116 = temps.at(114);
  dv_116 = 11.0 * dv_14;
  DataVector& dv_117 = temps.at(115);
  dv_117 = dv_115 - dv_116 * dv_17 + 2.0 * (dv_17 * dv_17);
  DataVector& dv_118 = temps.at(116);
  dv_118 = 11.0 * dv_15;
  DataVector& dv_119 = temps.at(117);
  dv_119 = -dv_46;
  DataVector& dv_120 = temps.at(118);
  dv_120 = dv_118 + dv_119;
  DataVector& dv_121 = temps.at(119);
  dv_121 = Dy * Dy * Dy * Dy;
  DataVector& dv_122 = temps.at(120);
  dv_122 = z * z * z * z;
  DataVector& dv_123 = temps.at(113);
  dv_123 = dv_115 - dv_118 * dv_16 + 2.0 * dv_121 + 2.0 * dv_122;
  DataVector& dv_124 = temps.at(121);
  dv_124 = 68.0 * dv_15;
  DataVector& dv_125 = temps.at(122);
  dv_125 = 7.0 * dv_16;
  DataVector& dv_126 = temps.at(123);
  dv_126 = dv_124 - dv_125;
  DataVector& dv_127 = temps.at(120);
  dv_127 = 4.0 * dv_122;
  DataVector& dv_128 = temps.at(124);
  dv_128 = 11.0 * dv_114 + 11.0 * dv_121 + dv_125 * dv_15 - dv_127;
  DataVector& dv_129 = temps.at(125);
  dv_129 = -dv_126 * dv_14 + dv_128;
  DataVector& dv_130 = temps.at(126);
  dv_130 = -d_45 * dv_117 - d_46 * (-dv_120 * dv_14 + dv_123) + d_47 * dv_129 +
           dv_109 + dv_113;
  DataVector& dv_131 = temps.at(127);
  dv_131 = Dy * d_45;
  DataVector& dv_132 = temps.at(128);
  dv_132 = 5.0 * dv_14;
  DataVector& dv_133 = temps.at(129);
  dv_133 = 5.0 * dv_16;
  DataVector& dv_134 = temps.at(130);
  dv_134 = dv_132 + dv_133;
  DataVector& dv_135 = temps.at(131);
  dv_135 = dv_134 + dv_29;
  DataVector& dv_136 = temps.at(132);
  dv_136 = d_46 * dv_79;
  DataVector& dv_137 = temps.at(133);
  dv_137 = 6.0 * dv_15;
  DataVector& dv_138 = temps.at(134);
  dv_138 = dv_137 + dv_24;
  DataVector& dv_139 = temps.at(135);
  dv_139 = dv_138 + dv_46;
  DataVector& dv_140 = temps.at(136);
  dv_140 = d_35 * dv_110;
  DataVector& dv_141 = temps.at(137);
  dv_141 = -34 * dv_14 + dv_49;
  DataVector& dv_142 = temps.at(138);
  dv_142 = dv_118 + dv_141;
  DataVector& dv_143 = temps.at(139);
  dv_143 = Dy * d_18;
  DataVector& dv_144 = temps.at(140);
  dv_144 = d_17 * dv_143;
  DataVector& dv_145 = temps.at(141);
  dv_145 = -9 * dv_15;
  DataVector& dv_146 = temps.at(142);
  dv_146 = dv_103 + dv_145;
  DataVector& dv_147 = temps.at(143);
  dv_147 = 3.0 * dv_4;
  DataVector& dv_148 = temps.at(144);
  dv_148 = d_43 * dv_146 * dv_147 + dv_142 * dv_144;
  DataVector& dv_149 = temps.at(145);
  dv_149 = dv_6 * dv_6 * dv_6;
  DataVector& dv_150 = temps.at(146);
  dv_150 = d_40 * dv_149;
  DataVector& dv_151 = temps.at(147);
  dv_151 = d_31 * dv_150;
  DataVector& dv_152 = temps.at(148);
  dv_152 = dv_143 * dv_4;
  DataVector& dv_153 = temps.at(149);
  dv_153 = dv_101 + dv_31;
  DataVector& dv_154 = temps.at(150);
  dv_154 = -dv_132 + dv_26 + dv_97;
  DataVector& dv_155 = temps.at(151);
  dv_155 = d_43 * dv_153 + d_49 * dv_154 + dv_110 * dv_79 - 12.0 * dv_152;
  DataVector& dv_156 = temps.at(152);
  dv_156 = d_50 * dv_6;
  DataVector& dv_157 = temps.at(153);
  dv_157 = dv_155 * dv_156;
  DataVector& dv_158 = temps.at(154);
  dv_158 = -dv_151 + dv_157;
  DataVector& dv_159 = temps.at(155);
  dv_159 = d_51 * dv_150;
  DataVector& dv_160 = temps.at(156);
  dv_160 = Dx * d_43;
  DataVector& dv_161 = temps.at(157);
  dv_161 = dv_160 * dv_79;
  DataVector& dv_162 = temps.at(158);
  dv_162 = Dx * d_17;
  DataVector& dv_163 = temps.at(159);
  dv_163 = 12.0 * dv_5;
  DataVector& dv_164 = temps.at(160);
  dv_164 = dv_162 * dv_163;
  DataVector& dv_165 = temps.at(161);
  dv_165 = dv_27 + dv_96;
  DataVector& dv_166 = temps.at(162);
  dv_166 = 5.0 * dv_15;
  DataVector& dv_167 = temps.at(163);
  dv_167 = dv_102 - dv_166 + dv_26;
  DataVector& dv_168 = temps.at(164);
  dv_168 = -d_44 * dv_165 - d_52 * dv_167 - dv_161 + dv_164;
  DataVector& dv_169 = temps.at(129);
  dv_169 = dv_133 + dv_166;
  DataVector& dv_170 = temps.at(162);
  dv_170 = 2.0 * Dx;
  DataVector& dv_171 = temps.at(165);
  dv_171 = d_45 * dv_170;
  DataVector& dv_172 = temps.at(166);
  dv_172 = Dx * d_46;
  DataVector& dv_173 = temps.at(167);
  dv_173 = 6.0 * dv_14;
  DataVector& dv_174 = temps.at(168);
  dv_174 = dv_173 + dv_29;
  DataVector& dv_175 = temps.at(169);
  dv_175 = -dv_174 - dv_46;
  DataVector& dv_176 = temps.at(170);
  dv_176 = d_43 * dv_55;
  DataVector& dv_177 = temps.at(171);
  dv_177 = -9 * dv_14;
  DataVector& dv_178 = temps.at(172);
  dv_178 = dv_177 + dv_98;
  DataVector& dv_179 = temps.at(173);
  dv_179 = -dv_178;
  DataVector& dv_180 = temps.at(174);
  dv_180 = -34 * dv_15 + dv_49;
  DataVector& dv_181 = temps.at(175);
  dv_181 = dv_116 + dv_180;
  DataVector& dv_182 = temps.at(176);
  dv_182 = d_18 * dv_162;
  DataVector& dv_183 = temps.at(177);
  dv_183 = d_53 * dv_179 * dv_5 + dv_171 * (-dv_169 - dv_24) + dv_172 * dv_32 +
           3.0 * dv_175 * dv_176 - dv_181 * dv_182;
  DataVector& dv_184 = temps.at(178);
  dv_184 = d_6 * dv_183 + dv_156 * dv_168 + dv_159;
  DataVector& dv_185 = temps.at(179);
  dv_185 = d_40 * dv_12;
  DataVector& dv_186 = temps.at(13);
  dv_186 = dv_13 * rp;
  DataVector& dv_187 = temps.at(180);
  dv_187 = d_54 * dv_12 - d_55 * dv_18 - d_56 * dv_18 + dv_185 + dv_186;
  DataVector& dv_188 = temps.at(181);
  dv_188 = d_19 * dv_6;
  DataVector& dv_189 = temps.at(182);
  dv_189 = dv_184 * xpdot + dv_187 * dv_188;
  DataVector& dv_190 = temps.at(183);
  dv_190 = dv_189 - ypdot * (d_6 * (-dv_131 * dv_28 + dv_135 * dv_136 +
                                    dv_139 * dv_140 + dv_148) +
                             dv_158);
  DataVector& dv_191 = temps.at(184);
  dv_191 = -d_57 * dv_190;
  DataVector& dv_192 = temps.at(8);
  dv_192 = d_5 * dv_8 + d_58 * dv_7 + dv_2;
  DataVector& dv_193 = temps.at(185);
  dv_193 = 4.0 * dv_192;
  DataVector& dv_194 = temps.at(186);
  dv_194 = -d_42 * dv_88 * dv_88 - d_48 * dv_130 - dv_105 * dv_107 +
           dv_191 * dv_193 + dv_94;
  DataVector& dv_195 = temps.at(82);
  dv_195 = d_22 * d_38 * dv_19 * dv_60 + d_25 * dv_60 * dv_62 + dv_194 * rp -
           9.0 * dv_45 * dv_92 - dv_46 * dv_48 - dv_46 * dv_64 + dv_49 * dv_51 +
           dv_49 * dv_66 - dv_84 * dv_85 - dv_84 * dv_86;
  DataVector& dv_196 = temps.at(187);
  dv_196 = M * dv_20;
  DataVector& dv_197 = temps.at(188);
  dv_197 = d_59 * dv_196;
  DataVector& dv_198 = temps.at(189);
  dv_198 = 72.0 * dv_91;
  DataVector& dv_199 = temps.at(190);
  dv_199 = M * dv_6;
  DataVector& dv_200 = temps.at(191);
  dv_200 = d_14 * dv_9;
  DataVector& dv_201 = temps.at(192);
  dv_201 = Dx + d_5 * d_51 * dv_199 - d_62 * dv_200;
  DataVector& dv_202 = temps.at(193);
  dv_202 = d_5 * dv_6;
  DataVector& dv_203 = temps.at(194);
  dv_203 = Dy + d_32 * dv_202 - d_64 * dv_200;
  DataVector& dv_204 = temps.at(195);
  dv_204 = d_66 * dv_106;
  DataVector& dv_205 = temps.at(196);
  dv_205 = d_76 * dv_47;
  DataVector& dv_206 = temps.at(197);
  dv_206 = Dx * xpddot;
  DataVector& dv_207 = temps.at(198);
  dv_207 = Dy * ypddot;
  DataVector& dv_208 = temps.at(199);
  dv_208 = dv_206 + dv_207;
  DataVector& dv_209 = temps.at(200);
  dv_209 = d_1 * dv_2;
  DataVector& dv_210 = temps.at(201);
  dv_210 = d_71 * dv_199 + d_73 * dv_7 + d_77 * dv_2;
  DataVector& dv_211 = temps.at(202);
  dv_211 = dv_89 * rp;
  DataVector& dv_212 = temps.at(200);
  dv_212 =
      -d_65 * (2.0 * M * dv_2 * dv_6 * rp + 3.0 * d_13 * dv_10 * rpdot -
               d_25 * dv_205 - dv_204 -
               dv_211 * (d_0 * dv_208 + d_4 * dv_209 + dv_209 * rp + dv_210)) +
      dv_201 * xpdot + dv_203 * ypdot;
  DataVector& dv_213 = temps.at(203);
  dv_213 = 24.0 * dv_20;
  DataVector& dv_214 = temps.at(204);
  dv_214 = dv_21 * dv_212;
  DataVector& dv_215 = temps.at(205);
  dv_215 = Dx - xp;
  DataVector& dv_216 = temps.at(206);
  dv_216 = Dy + d_80;
  DataVector& dv_217 = temps.at(207);
  dv_217 = dv_215 * xpdot + dv_216 * ypdot;
  DataVector& dv_218 = temps.at(208);
  dv_218 = dv_217 * rp;
  DataVector& dv_219 = temps.at(209);
  dv_219 = 3.0 * Dy;
  DataVector& dv_220 = temps.at(210);
  dv_220 = 3.0 * dv_5;
  DataVector& dv_221 = temps.at(211);
  dv_221 = 3.0 * Dx;
  DataVector& dv_222 = temps.at(212);
  dv_222 = xpdot * (dv_162 - dv_54 * (d_31 + dv_219) + xp * (dv_220 + dv_28)) +
           ypdot * (dv_143 - dv_55 * (d_51 + dv_221) + yp * (dv_147 + dv_32));
  DataVector& dv_223 = temps.at(213);
  dv_223 = dv_188 * dv_222;
  DataVector& dv_224 = temps.at(199);
  dv_224 = -d_10 + d_81 + dv_208;
  DataVector& dv_225 = temps.at(201);
  dv_225 = d_1 * dv_218 + d_74 * dv_217 + dv_210;
  DataVector& dv_226 = temps.at(214);
  dv_226 = d_13 * dv_88;
  DataVector& dv_227 = temps.at(215);
  dv_227 = 2.0 * d_25 * dv_9;
  DataVector& dv_228 = temps.at(216);
  dv_228 = dv_79 + yp;
  DataVector& dv_229 = temps.at(217);
  dv_229 = 2.0 * dv_5;
  DataVector& dv_230 = temps.at(212);
  dv_230 = 2.0 * d_4 * dv_222 +
           d_82 * (xpdot * (dv_162 - dv_228 * dv_54 + xp * (dv_229 + dv_36)) +
                   ypdot * (dv_143 - dv_55 * (dv_170 + xp) +
                            yp * (dv_38 + 2.0 * dv_4)));
  DataVector& dv_231 = temps.at(218);
  dv_231 = -dv_195;
  DataVector& dv_232 = temps.at(219);
  dv_232 = dv_196 * 1.0 / d_83;
  DataVector& dv_233 = temps.at(220);
  dv_233 = 20.0 * dv_16;
  DataVector& dv_234 = temps.at(221);
  dv_234 = dv_16 * rpdot;
  DataVector& dv_235 = temps.at(53);
  dv_235 = -dv_53;
  DataVector& dv_236 = temps.at(59);
  dv_236 = d_28 * dv_235 + dv_59;
  DataVector& dv_237 = temps.at(222);
  dv_237 = 16.0 * dv_236;
  DataVector& dv_238 = temps.at(223);
  dv_238 = d_76 * dv_10 * 1.0 / (d_12 * d_12 * d_12);
  DataVector& dv_239 = temps.at(224);
  dv_239 = 16.0 * dv_238;
  DataVector& dv_240 = temps.at(223);
  dv_240 = 32.0 * M * dv_238;
  DataVector& dv_241 = temps.at(225);
  dv_241 = d_0 * dv_224 + dv_225;
  DataVector& dv_242 = temps.at(226);
  dv_242 = d_23 * dv_9;
  DataVector& dv_243 = temps.at(227);
  dv_243 = dv_241 * dv_242;
  DataVector& dv_244 = temps.at(228);
  dv_244 = 16.0 * dv_16;
  DataVector& dv_245 = temps.at(229);
  dv_245 = M * dv_243;
  DataVector& dv_246 = temps.at(80);
  dv_246 = d_17 * (-d_9 * dv_68 + dv_15 - dv_30) + dv_82;
  DataVector& dv_247 = temps.at(68);
  dv_247 = dv_246 * rpdot;
  DataVector& dv_248 = temps.at(230);
  dv_248 = d_86 * dv_19;
  DataVector& dv_249 = temps.at(231);
  dv_249 = M * dv_248;
  DataVector& dv_250 = temps.at(232);
  dv_250 = dv_17 * xp * (xpdot * xpdot * xpdot);
  DataVector& dv_251 = temps.at(233);
  dv_251 = ypdot * (Dy * (-d_17 + d_6) - dv_143 + dv_17 * yp);
  DataVector& dv_252 = temps.at(234);
  dv_252 = dv_4 * yp;
  DataVector& dv_253 = temps.at(235);
  dv_253 = dv_14 * yp;
  DataVector& dv_254 = temps.at(236);
  dv_254 = Dy * dv_4;
  DataVector& dv_255 = temps.at(237);
  dv_255 = Dy * d_17;
  DataVector& dv_256 = temps.at(238);
  dv_256 = -dv_255;
  DataVector& dv_257 = temps.at(239);
  dv_257 = d_9 * dv_162;
  DataVector& dv_258 = temps.at(240);
  dv_258 = -d_6 * (-d_81 - dv_207) + d_89 * dv_1;
  DataVector& dv_259 = temps.at(73);
  dv_259 =
      16.0 *
      (d_10 * dv_251 + dv_250 +
       xpdot * (Dx * (-dv_258 - yp * (Dy + d_88 * yp)) - dv_257 +
                dv_57 * xpddot + xp * (Dy * (Dy + yp) + d_69 * dv_16 + dv_72)) +
       ypdot * (d_87 * dv_14 + d_87 * dv_16 + dv_235 * xpddot + dv_252 +
                dv_253 - dv_254 + dv_256 + dv_58 * ypddot + dv_73));
  DataVector& dv_260 = temps.at(57);
  dv_260 = 8.0 * dv_19;
  DataVector& dv_261 = temps.at(53);
  dv_261 = dv_162 * (d_9 + 2.0);
  DataVector& dv_262 = temps.at(241);
  dv_262 = dv_219 + yp;
  DataVector& dv_263 = temps.at(69);
  dv_263 =
      M * d_70 * dv_4 - d_1 * dv_207 * dv_4 +
      d_10 * (d_1 * (dv_17 + dv_5) + dv_251) - d_17 * d_67 * dv_26 +
      d_17 * dv_1 - d_3 * dv_14 - d_3 * dv_147 + d_3 * dv_25 + d_67 * dv_58 -
      d_90 * dv_1 + d_91 * dv_16 * yp + d_91 * dv_253 + d_92 * dv_26 +
      d_92 * dv_30 + d_93 * dv_26 + d_93 * dv_30 + d_94 * dv_14 + d_94 * dv_16 +
      dv_1 * dv_147 + dv_250 - dv_77 * xpddot +
      xpdot *
          (Dx * (-d_36 * dv_228 - d_95 - dv_258 + dv_262 * yp) - dv_261 +
           dv_75 * xpddot +
           xp * (-d_1 * dv_1 + d_35 * dv_16 * ypddot - dv_220 + dv_69 + dv_72));
  DataVector& dv_264 = temps.at(143);
  dv_264 = -dv_212;
  DataVector& dv_265 = temps.at(235);
  dv_265 = d_13 * dv_264;
  DataVector& dv_266 = temps.at(72);
  dv_266 = d_30 * dv_264;
  DataVector& dv_267 = temps.at(240);
  dv_267 = -dv_39;
  DataVector& dv_268 = temps.at(56);
  dv_268 = -dv_28;
  DataVector& dv_269 = temps.at(23);
  dv_269 = d_17 * dv_268 - d_18 * dv_32 + dv_23;
  DataVector& dv_270 = temps.at(32);
  dv_270 = d_19 * dv_267 + d_4 * dv_269;
  DataVector& dv_271 = temps.at(52);
  dv_271 = dv_270 * dv_270;
  DataVector& dv_272 = temps.at(131);
  dv_272 = -dv_135;
  DataVector& dv_273 = temps.at(144);
  dv_273 = dv_131 * dv_268 - dv_136 * dv_272 + dv_139 * dv_140 + dv_148;
  DataVector& dv_274 = temps.at(154);
  dv_274 = d_6 * dv_273 + dv_158;
  DataVector& dv_275 = temps.at(182);
  dv_275 = -dv_189 + dv_274 * ypdot;
  DataVector& dv_276 = temps.at(132);
  dv_276 = d_57 * dv_193;
  DataVector& dv_277 = temps.at(136);
  dv_277 = dv_4 * dv_5;
  DataVector& dv_278 = temps.at(56);
  dv_278 = -d_17 * dv_99 - d_18 * dv_104 + 14.0 * dv_277;
  DataVector& dv_279 = temps.at(115);
  dv_279 = d_45 * dv_117 + d_46 * (-dv_120 * dv_14 + dv_123) - d_47 * dv_129 -
           dv_109 - dv_113;
  DataVector& dv_280 = temps.at(92);
  dv_280 = d_48 * dv_279 + dv_107 * dv_278 + dv_94;
  DataVector& dv_281 = temps.at(107);
  dv_281 = -d_42 * dv_271 + dv_275 * dv_276 + dv_280;
  DataVector& dv_282 = temps.at(111);
  dv_282 = dv_19 * dv_19;
  DataVector& dv_283 = temps.at(118);
  dv_283 = 1.0 / dv_282;
  DataVector& dv_284 = temps.at(125);
  dv_284 = dv_269 * dv_6 - dv_270 * dv_89;
  DataVector& dv_285 = temps.at(205);
  dv_285 = -dv_215 * xpdot - dv_216 * ypdot;
  DataVector& dv_286 = temps.at(113);
  dv_286 = d_97 * dv_281 * dv_44;
  DataVector& dv_287 = temps.at(135);
  dv_287 = d_13 * dv_270;
  DataVector& dv_288 = temps.at(240);
  dv_288 = d_73 * dv_269 - dv_230 + 2.0 * dv_267 * rpdot;
  DataVector& dv_289 = temps.at(232);
  dv_289 = d_1 * dv_285;
  DataVector& dv_290 = temps.at(233);
  dv_290 = -dv_102;
  DataVector& dv_291 = temps.at(58);
  dv_291 = 11.0 * dv_16;
  DataVector& dv_292 = temps.at(216);
  dv_292 = Dx * d_45;
  DataVector& dv_293 = temps.at(242);
  dv_293 = dv_292 * (dv_118 + dv_290 + dv_291);
  DataVector& dv_294 = temps.at(116);
  dv_294 = dv_103 - dv_118;
  DataVector& dv_295 = temps.at(36);
  dv_295 = Dy * dv_36;
  DataVector& dv_296 = temps.at(121);
  dv_296 = -dv_124 + dv_125;
  DataVector& dv_297 = temps.at(243);
  dv_297 = 22.0 * dv_14 + dv_296;
  DataVector& dv_298 = temps.at(244);
  dv_298 = d_17 * dv_54;
  DataVector& dv_299 = temps.at(112);
  dv_299 = 4.0 * dv_114;
  DataVector& dv_300 = temps.at(245);
  dv_300 = 45.0 * dv_5;
  DataVector& dv_301 = temps.at(246);
  dv_301 = 22.0 * dv_15;
  DataVector& dv_302 = temps.at(247);
  dv_302 = 15.0 * dv_5;
  DataVector& dv_303 = temps.at(248);
  dv_303 = -dv_116 + dv_98;
  DataVector& dv_304 = temps.at(24);
  dv_304 = dv_100 + dv_16 + dv_24;
  DataVector& dv_305 = temps.at(249);
  dv_305 = dv_101 + dv_37;
  DataVector& dv_306 = temps.at(250);
  dv_306 = 15.0 * dv_4;
  DataVector& dv_307 = temps.at(251);
  dv_307 = Dy * Dy * Dy;
  DataVector& dv_308 = temps.at(252);
  dv_308 = 68.0 * dv_5;
  DataVector& dv_309 = temps.at(253);
  dv_309 = dv_217 * dv_6;
  DataVector& dv_310 = temps.at(145);
  dv_310 = d_41 * dv_149;
  DataVector& dv_311 = temps.at(179);
  dv_311 = 6.0 * dv_185;
  DataVector& dv_312 = temps.at(254);
  dv_312 = dv_199 * rpdot;
  DataVector& dv_313 = temps.at(255);
  dv_313 = M * dv_218;
  DataVector& dv_314 = temps.at(256);
  dv_314 = Dx * ypdot;
  DataVector& dv_315 = temps.at(257);
  dv_315 = Dy * xpdot;
  DataVector& dv_316 = temps.at(258);
  dv_316 = -dv_97;
  DataVector& dv_317 = temps.at(134);
  dv_317 = -ypdot * (dv_138 + dv_26);
  DataVector& dv_318 = temps.at(259);
  dv_318 = 12.0 * dv_14;
  DataVector& dv_319 = temps.at(260);
  dv_319 = 12.0 * dv_16;
  DataVector& dv_320 = temps.at(261);
  dv_320 = 4.0 * Dy;
  DataVector& dv_321 = temps.at(130);
  dv_321 = dv_101 + dv_134;
  DataVector& dv_322 = temps.at(262);
  dv_322 = dv_137 + dv_46 + dv_96;
  DataVector& dv_323 = temps.at(137);
  dv_323 = dv_141 + 33.0 * dv_15;
  DataVector& dv_324 = temps.at(138);
  dv_324 = dv_142 * dv_79;
  DataVector& dv_325 = temps.at(263);
  dv_325 = dv_145 + dv_318 + dv_46;
  DataVector& dv_326 = temps.at(264);
  dv_326 = Dx * d_18;
  DataVector& dv_327 = temps.at(265);
  dv_327 = 12.0 * dv_15;
  DataVector& dv_328 = temps.at(266);
  dv_328 = dv_177 + dv_327 + dv_46;
  DataVector& dv_329 = temps.at(169);
  dv_329 = dv_175 * dv_219;
  DataVector& dv_330 = temps.at(99);
  dv_330 = yp * (dv_101 + dv_173 + dv_46);
  DataVector& dv_331 = temps.at(129);
  dv_331 = 2.0 * d_45 * (dv_169 + dv_96);
  DataVector& dv_332 = temps.at(168);
  dv_332 = dv_174 + dv_26;
  DataVector& dv_333 = temps.at(174);
  dv_333 = 33.0 * dv_14 + dv_180;
  DataVector& dv_334 = temps.at(267);
  dv_334 = (1.0 / 24.0) * dv_283;
  DataVector& dv_335 = temps.at(210);
  dv_335 = -d_18 * dv_170 + dv_162 + dv_220 * xp;
  DataVector& dv_336 = temps.at(190);
  dv_336 = d_16 * dv_199;
  DataVector& dv_337 = temps.at(268);
  dv_337 = dv_213 * dv_336;
  DataVector& dv_338 = temps.at(269);
  dv_338 = dv_33 * xp;
  DataVector& dv_339 = temps.at(270);
  dv_339 = d_16 * dv_196;
  DataVector& dv_340 = temps.at(271);
  dv_340 = 12.0 * dv_339;
  DataVector& dv_341 = temps.at(239);
  dv_341 = -d_6 * dv_1 * xpdot + d_88 * dv_326 + dv_257 - dv_5 * xp;
  DataVector& dv_342 = temps.at(272);
  dv_342 = sqrt(dv_19) * dv_19;
  DataVector& dv_343 = temps.at(273);
  dv_343 = d_106 * dv_342;
  DataVector& dv_344 = temps.at(53);
  dv_344 = Dx * d_95 - d_37 * dv_55 + dv_261 - yp * (d_1 * dv_315 + dv_54) +
           ypdot * (-d_6 * dv_315 + d_7 * dv_54);
  DataVector& dv_345 = temps.at(274);
  dv_345 = d_108 * d_13 * dv_342;
  DataVector& dv_346 = temps.at(275);
  dv_346 = d_62 * dv_242;
  DataVector& dv_347 = temps.at(276);
  dv_347 = d_5 * dv_196 * dv_49;
  DataVector& dv_348 = temps.at(277);
  dv_348 = d_105 * dv_20;
  DataVector& dv_349 = temps.at(278);
  dv_349 = dv_346 * dv_348;
  DataVector& dv_350 = temps.at(279);
  dv_350 = d_65 * dv_16;
  DataVector& dv_351 = temps.at(280);
  dv_351 = d_105 * dv_47;
  DataVector& dv_352 = temps.at(281);
  dv_352 = d_59 * dv_20 * dv_351;
  DataVector& dv_353 = temps.at(282);
  dv_353 = -dv_326;
  DataVector& dv_354 = temps.at(283);
  dv_354 = d_19 * (d_51 * dv_5 + dv_162 + dv_353) + d_4 * dv_335;
  DataVector& dv_355 = temps.at(284);
  dv_355 = d_78 * dv_213 * dv_89;
  DataVector& dv_356 = temps.at(285);
  dv_356 = dv_339 * dv_61;
  DataVector& dv_357 = temps.at(286);
  dv_357 = d_13 * dv_201;
  DataVector& dv_358 = temps.at(187);
  dv_358 = dv_196 * dv_46;
  DataVector& dv_359 = temps.at(287);
  dv_359 = d_109 * d_40 * dv_49;
  DataVector& dv_360 = temps.at(288);
  dv_360 = dv_20 * dv_359;
  DataVector& dv_361 = temps.at(289);
  dv_361 = dv_201 * dv_21;
  DataVector& dv_362 = temps.at(290);
  dv_362 = d_79 * dv_361;
  DataVector& dv_363 = temps.at(291);
  dv_363 = d_59 * dv_60;
  DataVector& dv_364 = temps.at(292);
  dv_364 = d_5 * dv_319 * dv_50;
  DataVector& dv_365 = temps.at(293);
  dv_365 = 24.0 * dv_16;
  DataVector& dv_366 = temps.at(294);
  dv_366 = d_40 * dv_47;
  DataVector& dv_367 = temps.at(295);
  dv_367 = d_65 * dv_365 * dv_366;
  DataVector& dv_368 = temps.at(60);
  dv_368 = d_14 * d_40 * dv_60;
  DataVector& dv_369 = temps.at(296);
  dv_369 = 8.0 * dv_20 * dv_368;
  DataVector& dv_370 = temps.at(79);
  dv_370 = d_17 * dv_70 + dv_74 + dv_76 - dv_78 - dv_81;
  DataVector& dv_371 = temps.at(270);
  dv_371 = 8.0 * dv_339 * dv_370;
  DataVector& dv_372 = temps.at(294);
  dv_372 = 24.0 * dv_363 * dv_366;
  DataVector& dv_373 = temps.at(20);
  dv_373 = d_7 * dv_20;
  DataVector& dv_374 = temps.at(75);
  dv_374 = d_58 * dv_370;
  DataVector& dv_375 = temps.at(74);
  dv_375 = dv_373 * dv_374;
  DataVector& dv_376 = temps.at(76);
  dv_376 = 12.0 * d_78 * dv_47;
  DataVector& dv_377 = temps.at(272);
  dv_377 = 1.0 / dv_342;
  DataVector& dv_378 = temps.at(70);
  dv_378 = 2.0 * dv_89;
  DataVector& dv_379 = temps.at(297);
  dv_379 = 18.0 * d_110 * dv_21;
  DataVector& dv_380 = temps.at(298);
  dv_380 = -dv_294;
  DataVector& dv_381 = temps.at(299);
  dv_381 = 48.0 * d_0 * dv_12;
  DataVector& dv_382 = temps.at(300);
  dv_382 = d_4 * dv_311;
  DataVector& dv_383 = temps.at(301);
  dv_383 = dv_170 * dv_5 + xp * (dv_17 + dv_95);
  DataVector& dv_384 = temps.at(302);
  dv_384 = dv_1 * dv_172;
  DataVector& dv_385 = temps.at(303);
  dv_385 = dv_162 * (17.0 * d_3 + d_98);
  DataVector& dv_386 = temps.at(304);
  dv_386 = M * dv_173;
  DataVector& dv_387 = temps.at(305);
  dv_387 = dv_137 - 15.0 * dv_14 + dv_26;
  DataVector& dv_388 = temps.at(306);
  dv_388 = M * dv_137;
  DataVector& dv_389 = temps.at(167);
  dv_389 = -15 * dv_15 + dv_173 + dv_26;
  DataVector& dv_390 = temps.at(236);
  dv_390 = d_43 * dv_254;
  DataVector& dv_391 = temps.at(20);
  dv_391 = d_112 * dv_373;
  DataVector& dv_392 = temps.at(307);
  dv_392 = -d_17 * dv_79 + d_35 * dv_4 + dv_143;
  DataVector& dv_393 = temps.at(308);
  dv_393 = dv_41 * yp;
  DataVector& dv_394 = temps.at(309);
  dv_394 = -d_10 * dv_143 + d_33 * dv_0 + dv_252 - dv_255 * (d_10 + 1.0);
  DataVector& dv_395 = temps.at(237);
  dv_395 = Dx * d_6 * xpdot * ypdot - d_11 * dv_255 -
           xp * (Dx * d_35 - d_1 * dv_314 + d_7 * dv_315) -
           yp * (-d_1 * dv_0 + d_10 * dv_5 + dv_229);
  DataVector& dv_396 = temps.at(226);
  dv_396 = d_64 * dv_242;
  DataVector& dv_397 = temps.at(277);
  dv_397 = dv_348 * dv_396;
  DataVector& dv_398 = temps.at(310);
  dv_398 = d_19 * (d_31 * dv_4 + dv_143 + dv_256) + d_4 * dv_392;
  DataVector& dv_399 = temps.at(311);
  dv_399 = d_64 * dv_226;
  DataVector& dv_400 = temps.at(312);
  dv_400 = d_13 * dv_203;
  DataVector& dv_401 = temps.at(313);
  dv_401 = d_79 * dv_87;
  DataVector& dv_402 = temps.at(314);
  dv_402 = dv_203 * dv_21;
  DataVector& dv_403 = temps.at(315);
  dv_403 = d_79 * dv_90;
  DataVector& dv_404 = temps.at(90);
  dv_404 = d_111 * dv_92;
  DataVector& dv_405 = temps.at(186);
  dv_405 = d_113 * dv_194;
  DataVector& dv_406 = temps.at(316);
  dv_406 = d_44 * dv_54;
  DataVector& dv_407 = temps.at(37);
  dv_407 = dv_80 + yp * (dv_100 + dv_37);
  DataVector& dv_408 = temps.at(98);
  dv_408 = 12.0 * dv_1;
  DataVector& dv_409 = temps.at(78);
  dv_409 = 4.0 * Dx;
  DataVector& dv_410 = temps.at(317);
  dv_410 = 48.0 * d_115 * dv_19;
  DataVector& dv_411 = temps.at(318);
  dv_411 = d_13 * dv_282;
  DataVector& dv_412 = temps.at(280);
  dv_412 = dv_19 * dv_351;
  DataVector& dv_413 = temps.at(319);
  dv_413 = d_117 * dv_89;

  get(get<CurvedScalarWave::Tags::Psi>(*result)) =
      dv_21 *
      (0.5 * d_16 * dv_43 * dv_45 - 1.0 / 24.0 * d_21 * dv_195 * dv_45 + 1.0);
  get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(*result)) =
      dv_334 *
      (d_21 * d_84 * dv_214 * dv_231 + d_79 * dv_214 * dv_91 -
       dv_197 * dv_198 * rpdot +
       12.0 * dv_197 *
           (d_23 * d_75 * dv_227 * dv_88 + d_71 * dv_90 -
            dv_211 * (-d_71 * dv_39 + d_73 * dv_41 - dv_230) + dv_218 * dv_41 -
            dv_223 - dv_226 * rp * (d_0 * dv_224 + dv_225) + dv_87 * rpdot) +
       dv_212 * dv_213 - 14.0 * dv_231 * dv_232 * rpdot -
       dv_232 *
           (M * dv_259 * dv_85 + d_15 * dv_237 * dv_245 +
            d_20 * dv_16 * dv_240 + d_20 * dv_244 * dv_266 +
            64.0 * d_22 * dv_236 * dv_65 * rpdot + d_22 * dv_244 * dv_245 -
            8.0 * d_24 * dv_243 * dv_246 + d_24 * dv_263 * dv_61 +
            d_25 * dv_237 * dv_50 * rpdot - d_29 * dv_239 * dv_246 +
            d_39 * dv_236 * dv_240 + d_39 * dv_237 * dv_266 -
            d_39 * dv_243 * dv_49 + d_39 * dv_259 * dv_65 +
            18.0 * d_50 * dv_264 * dv_283 * (dv_284 * dv_284) +
            d_82 * dv_281 * rpdot - d_83 * dv_16 * dv_239 +
            d_83 * dv_237 * dv_249 - d_83 * dv_265 * dv_49 -
            d_85 * dv_247 * dv_47 - d_86 * dv_246 * dv_260 * d_83 * rp -
            d_96 * dv_246 * dv_265 + d_96 * dv_263 * dv_63 -
            dv_233 * dv_48 * rpdot + 32.0 * dv_234 * dv_51 -
            44.0 * dv_234 * dv_64 + 80.0 * dv_234 * dv_66 +
            dv_244 * dv_249 * d_83 * rp * rp - 36.0 * dv_247 * dv_86 -
            dv_248 * dv_49 * d_83 * rp * rp * rp - dv_264 * dv_286 -
            18.0 * dv_284 * dv_45 *
                (d_13 * dv_270 * dv_9 * rpdot - d_86 * dv_227 * dv_270 -
                 dv_211 * dv_288 + dv_217 * dv_269 * rp - dv_223 -
                 dv_241 * dv_287 * rp + 2.0 * dv_269 * dv_6 * rpdot) +
            dv_286 * (3.0 * d_13 * d_65 * dv_10 * rpdot - d_5 * dv_285 * dv_7 -
                      d_65 * dv_204 - dv_2 - dv_200 * dv_241 - dv_205) +
            rp *
                (-d_0 * d_98 * dv_287 * dv_288 +
                 d_100 * dv_193 *
                     (-d_19 * dv_187 * dv_217 - d_71 * dv_187 * dv_6 -
                      dv_184 * xpddot -
                      dv_188 * (-M * d_77 * dv_18 + d_1 * dv_3 +
                                16.0 * d_102 * dv_12 + 12.0 * d_25 * dv_2 +
                                d_41 * dv_309 + 16.0 * d_6 * dv_309 +
                                d_7 * dv_218 * dv_6 + d_71 * dv_106 -
                                24.0 * d_99 * dv_18) +
                      dv_274 * ypddot +
                      xpdot *
                          (6.0 * d_40 * dv_12 * dv_285 * xp -
                           d_6 * (xpdot *
                                      (d_43 * (dv_329 - dv_332 * yp) -
                                       d_44 * dv_170 *
                                           (20.0 * dv_15 + dv_233 + dv_290 +
                                            27.0 * dv_5) +
                                       d_49 * (-9 * Dy * dv_178 + dv_333 * yp) +
                                       d_90 * dv_4 * (18.0 * Dy * yp - dv_181) +
                                       dv_331) +
                                  ypdot *
                                      (20.0 * Dx * dv_131 +
                                       d_104 * (dv_329 + dv_330) +
                                       d_43 * dv_170 * (dv_103 + dv_5 + dv_71) +
                                       d_53 * (Dy * dv_179 + dv_328 * yp) -
                                       2.0 * dv_298 * (dv_181 + 34.0 * dv_5))) -
                           d_89 * dv_183 -
                           dv_156 *
                               (d_103 * (-d_26 * (dv_167 + 5.0 * dv_5) +
                                         d_44 * dv_79 + 6.0 * dv_162 * dv_216 +
                                         dv_326 * (-d_80 - dv_219)) +
                                xpdot * (8.0 * Dx * dv_262 * xp * yp -
                                         3.0 * d_17 * (dv_165 + dv_35) +
                                         d_18 * (2.0 * Dy * yp - dv_167) -
                                         6.0 * dv_110)) -
                           dv_168 * dv_312 - dv_168 * dv_313 - dv_310 * xpdot) +
                      ypdot *
                          (d_6 *
                               (d_43 *
                                    (-d_103 * (dv_272 * dv_320 + dv_321 * yp) +
                                     dv_0 * (-27 * dv_15 + dv_318 + dv_319 -
                                             20.0 * dv_5)) -
                                d_44 *
                                    (3.0 * dv_314 *
                                         (dv_119 - dv_137 + dv_14 + dv_163) +
                                     xpdot * (d_35 * dv_322 + dv_28 * dv_320)) +
                                d_45 * (-dv_0 * dv_79 - dv_317) +
                                d_49 * (dv_0 * (54.0 * dv_15 + 36.0 * dv_16 +
                                                dv_177 + dv_308) +
                                        ypdot * (-dv_323 * yp + dv_324)) +
                                d_52 * (9.0 * dv_314 * (dv_146 + dv_22) +
                                        xpdot * (-d_35 * dv_325 + dv_324))) +
                           d_89 * dv_273 + dv_155 * dv_312 + dv_155 * dv_313 +
                           dv_156 * (d_17 * (2.0 * dv_0 * (dv_219 + 5.0 * yp) +
                                             ypdot * (-dv_132 - dv_316 -
                                                      8.0 * dv_5 - dv_67)) +
                                     d_18 * (-4 * dv_0 * dv_262 +
                                             3.0 * ypdot * (dv_153 + dv_229)) +
                                     d_31 * xp *
                                         (6.0 * dv_314 * (-d_80 - dv_79) +
                                          xpdot * (dv_154 + dv_22)) -
                                     2.0 * d_44 * (dv_314 + dv_315)) +
                           dv_285 * dv_311 * yp - dv_310 * ypdot)) +
                 4.0 * d_100 * dv_275 *
                     (2.0 * M * d_5 * d_73 * dv_6 - d_4 * d_5 * dv_289 -
                      d_4 * d_65 * d_98 * dv_6 * rpdot - d_58 * dv_289 -
                      d_7 * dv_202 * rpdot - d_72 + dv_206 + dv_207) +
                 d_13 * d_99 * dv_193 * dv_275 +
                 8.0 * d_22 * d_86 * dv_192 * dv_275 -
                 d_24 * d_86 * d_98 * dv_271 + dv_280 * rpdot +
                 rp * (8.0 * d_102 * dv_279 +
                       d_54 *
                           (xpdot *
                                (d_44 * (-dv_14 *
                                             (22.0 * dv_16 + dv_300 + dv_301) +
                                         dv_17 * (dv_302 + dv_98) + dv_299) +
                                 d_52 * (15.0 * Dy * yp * (-dv_15 + dv_16) -
                                         dv_128 + dv_14 * (dv_126 + dv_300)) -
                                 dv_160 * (15.0 * dv_108 + dv_294 * yp) +
                                 dv_293 +
                                 dv_298 * (-45 * dv_295 + dv_297 * yp)) -
                            ypdot *
                                (-d_18 * dv_306 *
                                     (-dv_219 * dv_38 + dv_305 * yp) -
                                 d_43 *
                                     (-d_101 * dv_307 + 4.0 * dv_121 + dv_127 +
                                      dv_14 * (-dv_301 + dv_49 + 11.0 * dv_5) -
                                      dv_16 * dv_301 + dv_291 * dv_5 + dv_299) +
                                 d_49 * (-dv_125 * dv_5 + dv_128 +
                                         dv_14 * (dv_296 + dv_308) -
                                         22.0 * dv_307 * yp) +
                                 15.0 * dv_110 * (dv_295 - dv_304 * yp) +
                                 dv_131 * dv_303)) +
                       dv_106 * dv_278 * rpdot + 8.0 * dv_150 * dv_217 -
                       dv_186 *
                           (xpdot * (3.0 * dv_162 - dv_54 * (7.0 * Dy + d_101) +
                                     xp * (7.0 * dv_5 + dv_99)) +
                            ypdot *
                                (d_18 * dv_219 - dv_55 * (7.0 * Dx + 4.0 * xp) +
                                 yp * (dv_104 + 7.0 * dv_4))) +
                       dv_218 * dv_278 * dv_7))));
  get<0>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      -dv_334 *
      (d_111 * dv_201 * dv_377 * (dv_43 * dv_43) -
       d_113 * dv_361 *
           (2.0 * d_40 * dv_93 - d_42 * dv_42 * dv_42 - d_48 * dv_130 -
            dv_105 * dv_107 - dv_190 * dv_276) -
       d_13 * d_62 * dv_340 * dv_42 + dv_201 * dv_213 - dv_201 * dv_369 -
       dv_335 * dv_337 + dv_338 * dv_340 - dv_34 * dv_362 + dv_341 * dv_343 +
       dv_341 * dv_352 + dv_344 * dv_345 + dv_344 * dv_356 - dv_346 * dv_347 +
       dv_346 * dv_371 + dv_349 * dv_350 + dv_349 * dv_363 + dv_354 * dv_355 +
       dv_357 * dv_358 - dv_357 * dv_360 - dv_357 * dv_375 + dv_361 * dv_364 -
       dv_361 * dv_367 - dv_361 * dv_370 * dv_376 - dv_361 * dv_372 +
       dv_362 * dv_42 * dv_89 -
       dv_379 * dv_43 *
           (d_13 * d_62 * dv_42 + 2.0 * dv_335 * dv_6 - dv_338 -
            dv_354 * dv_378) -
       dv_391 *
           (3.0 * M * d_13 * d_6 * dv_354 * dv_88 +
            M * dv_6 * rp * (d_44 * dv_165 + d_52 * dv_167 + dv_161 - dv_164) +
            2.0 * d_6 *
                (d_44 * dv_302 * (dv_17 + dv_96) + dv_172 * dv_380 +
                 15.0 * dv_176 * (dv_16 + dv_29 + dv_95) + dv_182 * dv_297 +
                 dv_293) -
            dv_159 - dv_191 * (d_107 * d_51 + d_5 * d_61 + xpdot) -
            dv_89 *
                (d_114 * dv_383 +
                 d_50 * (-d_44 * (-d_3 * dv_387 + dv_386) -
                         d_52 * (-d_3 * dv_389 + dv_388) + 4.0 * dv_1 * dv_292 -
                         dv_229 * dv_385 + 4.0 * dv_384 +
                         xpdot * (d_45 * (dv_177 + dv_27) + d_46 * dv_25 +
                                  d_47 * (-17 * dv_15 + dv_26 + dv_318) -
                                  dv_112 + 12.0 * dv_390)) +
                 d_6 * (-d_104 * (M * dv_97 - d_3 * dv_325) -
                        d_53 * (M * dv_102 - d_3 * dv_322) + dv_1 * dv_171 -
                        dv_35 * dv_385 + 20.0 * dv_384 +
                        xpdot * (-d_46 * dv_332 + d_47 * dv_333 -
                                 54.0 * dv_111 + dv_331 + 36.0 * dv_390)) +
                 d_85 * dv_383 - dv_381 * xp - dv_382 * xp)));
  get<1>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      dv_334 *
      (-dv_203 * dv_213 + dv_203 * dv_369 - dv_203 * dv_377 * dv_404 +
       dv_337 * dv_392 + dv_340 * dv_393 - dv_340 * dv_399 + dv_343 * dv_394 +
       dv_345 * dv_395 + dv_347 * dv_396 - dv_350 * dv_397 + dv_352 * dv_394 -
       dv_355 * dv_398 + dv_356 * dv_395 - dv_358 * dv_400 + dv_360 * dv_400 -
       dv_363 * dv_397 - dv_364 * dv_402 + dv_367 * dv_402 - dv_371 * dv_396 +
       dv_372 * dv_402 + dv_375 * dv_400 - dv_376 * dv_402 * dv_83 +
       dv_379 * dv_91 *
           (-dv_378 * dv_398 + 2.0 * dv_392 * dv_6 + dv_393 - dv_399) -
       dv_391 *
           (-d_6 * d_84 * dv_226 * dv_398 +
            d_97 *
                (Dy * d_45 * dv_303 - Dy * d_46 * (dv_116 + dv_291 + dv_316) -
                 d_43 * dv_305 * dv_306 -
                 dv_144 * (dv_125 - 68.0 * dv_14 + dv_301) -
                 15.0 * dv_304 * dv_406) +
            dv_151 - dv_157 + dv_191 * (d_32 * d_58 + d_5 * d_63 + ypdot) +
            dv_89 *
                (d_114 * dv_407 +
                 d_50 *
                     (-d_43 * (-d_3 * (dv_145 + dv_31) + dv_388) +
                      d_45 * dv_30 * ypdot -
                      d_49 * (-d_3 * (-17 * dv_14 + dv_26 + dv_327) + dv_386) -
                      6.0 * dv_152 * (d_1 + 5.0 * d_3) + dv_406 * dv_408 +
                      xpdot * (d_43 * dv_389 * xp + d_44 * dv_387 * yp +
                               dv_131 * dv_409 - 34.0 * dv_143 * dv_162 +
                               dv_172 * dv_320)) +
                 d_6 * (-d_104 *
                            (Dx * dv_79 * (9.0 * d_3 + d_7) - dv_330 * xpdot) +
                        d_35 * d_44 * (Dx * dv_408 + dv_328 * xpdot) +
                        2.0 * d_43 *
                            (Dy * (-Dy * d_98 + dv_0 * yp) + d_3 * dv_321) +
                        d_45 * (20.0 * Dy * dv_0 + dv_317) -
                        d_49 * (-d_3 * dv_323 +
                                dv_409 * (M * dv_221 + 17.0 * dv_5 * xpdot))) +
                 d_85 * dv_407 - dv_381 * yp - dv_382 * yp)) -
       dv_401 * dv_402 + dv_402 * dv_403 + dv_402 * dv_405);
  get<2>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      -1.0 / 24.0 * z *
      (-M * dv_260 *
           (3.0 * d_115 * d_117 * d_13 * d_50 * (d_4 * dv_33 + dv_40) +
            d_115 * (dv_106 +
                     rp * (-d_17 * dv_303 + d_18 * dv_380 + 30.0 * dv_277)) -
            dv_378 * (d_115 * d_60 * dv_6 +
                      d_115 * rp *
                          (xpdot * (5.0 * dv_162 + dv_22 * xp + dv_353) +
                           ypdot * (5.0 * dv_143 + 6.0 * dv_252 + dv_256)) +
                      d_55 * dv_6 + d_56 * dv_6)) *
           1.0 / d_39 +
       d_105 * d_109 * dv_411 + d_106 * d_8 * dv_282 + d_108 * d_116 * dv_411 +
       d_110 * d_115 * dv_198 * (-dv_413 + dv_6) +
       d_116 * d_78 * dv_19 * dv_61 - d_38 * dv_282 + 48.0 * d_5 * dv_106 -
       d_5 * dv_19 * dv_62 + d_59 * d_8 * dv_412 + d_65 * dv_412 -
       d_7 * dv_374 * dv_63 - d_78 * dv_410 * dv_413 - 24.0 * dv_11 +
       24.0 * dv_14 + 24.0 * dv_15 - dv_260 * dv_368 + dv_336 * dv_410 -
       dv_359 * dv_63 + dv_364 + dv_365 - dv_367 - dv_370 * dv_376 - dv_372 +
       dv_401 - dv_403 + dv_404 * dv_44 - dv_405 + dv_46 * dv_65) /
      (square(dv_19) * sqrt(dv_19));
}
}  // namespace CurvedScalarWave::Worldtube
