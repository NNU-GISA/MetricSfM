// ObjectSfM - Object Based Structure-from-Motion.
// Copyright (C) 2018  Ohio State University, CEGE, GDA group
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "orientation/relative_pose_from_fundamental_matrix.h"

#include <Eigen/LU>
#include <Eigen/SVD>
#include <iostream>

namespace objectsfm {

	bool RelativePoseFromFundamentalMatrix::ReltivePoseFromFMatrix(const Eigen::Matrix3d& F, const std::vector<Eigen::Vector2d>& pts1, const std::vector<Eigen::Vector2d>& pts2, 
		double &f1, double &f2, Eigen::Matrix3d& R, Eigen::Vector3d &t)
	{
		// Only consider fundamental matrices that we can decompose focal lengths from.
		if (!FocalLengthFromFMatrix(F.data(), f1, f2))
		{
			return false;
		}

		// Given the focal length and the fundamental matrix, calculate the essential matrix
		Eigen::Matrix3d E;
		EMatrixFromFMatrix(F.data(), f1, f2, E.data());

		// normalize the corresponding points
		int num = pts1.size();
		std::vector<Eigen::Vector2d> pts1_homo(num), pts2_homo(num);
		for (size_t i = 0; i < num; i++)
		{
			pts1_homo[i] = pts1[i] / f1;
			pts2_homo[i] = pts2[i] / f2;
		}

		// obtain R and t from the essential matrix
		RelativePoseFromEssentialMatrix::ReltivePoseFromEMatrix(E, pts1_homo, pts2_homo, R, t);

		return true;
	}

	// Decompose the fundmental matrix and recover focal lengths f1, f2 such that 
	// diag([f2 f2 1]) F diag[f1 f1 1]) is a valid essential matrix. Besed on the
	// paper "Extraction of Focal Lengths from the Fundamental Matrix"-by Hartley
	bool RelativePoseFromFundamentalMatrix::FocalLengthFromFMatrix(const double *fmatrix, double &focal_length1, double &focal_length2)
	{
		Eigen::Map<const Eigen::Matrix3d> F(fmatrix);

		// The two epipoles e1 and e2 are determined by solving the equations F e1 = 0 and Ft e2 = 0.
		const Eigen::Vector3d epipole1 = F.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
		const Eigen::Vector3d epipole2 = F.transpose().jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
		if (epipole1.x() == 0 || epipole2.x() == 0)
		{
			return false;
		}

		// Find the rotation that takes epipole1 to (e_0, 0, e_2) and the
		// epipole2 to (e'_0, 0, e'_2). If we form a rotation matrix:
		// R = [ cos x  -sin x  0 ]
		//     [ sin x   cos x  0 ]
		//     [ 0       0      1 ]
		// then we can solve for the angle x such that R * e1 = (e_1, 0, e_3).
		// We can solve this simply be investigating the second row and noting that
		// e1(0) * sin x + e2 * cos x = 0.
		const double theta1 = atan2(-epipole1(1), epipole1(0));
		const double theta2 = atan2(-epipole2(1), epipole2(0));

		Eigen::Matrix3d rotation1, rotation2;
		rotation1 << cos(theta1), -sin(theta1), 0,
			sin(theta1), cos(theta1), 0,
			0, 0, 1;
		rotation2 << cos(theta2), -sin(theta2), 0,
			sin(theta2), cos(theta2), 0,
			0, 0, 1;

		// subsequently, correct the fundamental matrix to reflect this change
		// the fundamental matrix is now of the form:
		// F = [ e'_2   0    0   ] [ a b a ] [ e_2   0     0  ]
		//     [ 0      1    0   ] [ c d c ] [ 0     1     0  ]
		//     [ 0      0  -e'_1 ] [ a b a ] [ 0     0   -e_1 ]
		const Eigen::Matrix3d F_rotated = rotation2 * F * rotation1.transpose();

		// then, try to obtain the [ a b c d] matrix
		const Eigen::Vector3d epipole1_rotated = rotation1 * epipole1;
		const Eigen::Vector3d epipole2_rotated = rotation2 * epipole2;

		Eigen::Matrix3d factorized_matrix = Eigen::DiagonalMatrix<double, 3>(epipole2_rotated(2), 1, -epipole2_rotated(0)).inverse()
			* F_rotated
			* Eigen::DiagonalMatrix<double, 3>(epipole1_rotated(2), 1, -epipole1_rotated(0)).inverse();

		const double a = factorized_matrix(0, 0);
		const double b = factorized_matrix(0, 1);
		const double c = factorized_matrix(1, 0);
		const double d = factorized_matrix(1, 1);

		// finally, the focal lengths of two images can be calculabted as
		const double focal_length1_sq =
			(-a * c * epipole1_rotated(0) * epipole1_rotated(0)) /
			(a * c * epipole1_rotated(2) * epipole1_rotated(2) + b * d);
		const double focal_length2_sq =
			(-a * b * epipole2_rotated(0) * epipole2_rotated(0)) /
			(a * b * epipole2_rotated(2) * epipole2_rotated(2) + c * d);
		if (focal_length1_sq < 0 || focal_length2_sq < 0)
		{
			return false;
		}

		focal_length1 = std::sqrt(focal_length1_sq);
		focal_length2 = std::sqrt(focal_length2_sq);

		return true;
	}

	void RelativePoseFromFundamentalMatrix::EMatrixFromFMatrix(const double *fmatrix, double &focal_length1, double &focal_length2, double *ematrix)
	{
		// Given the fundamental matrix and the focal length, under the assumpation that
		// the principle point is in the center of the image, the essential matrix can be
		// easily calculate as: E = K2.t() * F * K1

		const Eigen::Map<const Eigen::Matrix3d> F(fmatrix);
		Eigen::Map<Eigen::Matrix3d> E(ematrix);
		E = Eigen::DiagonalMatrix<double, 3>(focal_length2, focal_length2, 1.0)
			* F
			* Eigen::DiagonalMatrix<double, 3>(focal_length1, focal_length1, 1.0);
	}

}  // namespace objectsfm
