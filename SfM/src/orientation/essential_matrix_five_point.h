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

#ifndef OBJECTSFM_CAMERA_FIVE_POINT_ESSENTIAL_MATRIX_H_
#define OBJECTSFM_CAMERA_FIVE_POINT_ESSENTIAL_MATRIX_H_

#include <vector>
#include <Eigen/Core>

namespace objectsfm {
	class EssentialMatrixFivePoints
	{
	public:
		EssentialMatrixFivePoints() {};
		~EssentialMatrixFivePoints() {};

		// get the best estimated essential matrix
		static bool FivePointEssentialMatrixRANSAC(const std::vector<Eigen::Vector2d>& image1_points,
			const std::vector<Eigen::Vector2d>& image2_points,
			Eigen::Matrix3d &E);

		// given corresponding points, get 10 essential matrix hypotheses
		static bool FivePointEssentialMatrix(const std::vector<Eigen::Vector2d>& image1_points,
			const std::vector<Eigen::Vector2d>& image2_points,
			std::vector<Eigen::Matrix3d>* essential_matrices);

	private:
		static Eigen::Matrix<double, 1, 10> MultiplyDegOnePoly(const Eigen::RowVector4d& a, const Eigen::RowVector4d& b);

		static Eigen::Matrix<double, 1, 20> MultiplyDegTwoDegOnePoly(const Eigen::Matrix<double, 1, 10>& a, const Eigen::RowVector4d& b);

		static Eigen::Matrix<double, 1, 10> EETranspose(const Eigen::Matrix<double, 1, 4> null_space[3][3], int i, int j);

		static Eigen::Matrix<double, 9, 20> GetTraceConstraint(const Eigen::Matrix<double, 1, 4> null_space[3][3]);

		static Eigen::Matrix<double, 1, 20> GetDeterminantConstraint(const Eigen::Matrix<double, 1, 4> null_space[3][3]);

		static Eigen::Matrix<double, 10, 20> BuildConstraintMatrix(const Eigen::Matrix<double, 1, 4> null_space[3][3]);

		static double Error(const Eigen::Matrix3d& E, const std::vector<Eigen::Vector2d>& image1_points, 
			const std::vector<Eigen::Vector2d>& image2_points);
	};

}// namespace objectsfm

#endif  // OBJECTSFM_CAMERA_FIVE_POINT_ESSENTIAL_MATRIX_H_



