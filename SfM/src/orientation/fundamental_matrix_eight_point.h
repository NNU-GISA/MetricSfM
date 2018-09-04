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

#ifndef OBJECTSFM_CAMERA_EIGHT_POINT_FUNDAMENTAL_MATRIX_H_
#define OBJECTSFM_CAMERA_EIGHT_POINT_FUNDAMENTAL_MATRIX_H_

#include <vector>
#include <Eigen/Core>

namespace objectsfm {

	//  estimate the fundamental matrix that makes pts2^t * F * pts1 = 0
	class FundamentalMatrixEightPoint
	{
	public:
		FundamentalMatrixEightPoint() {};
		~FundamentalMatrixEightPoint() {};

		// get the best estimated fundamental matrix
		static bool NormalizedEightPointFundamentalMatrixRANSAC(std::vector<Eigen::Vector2d>& pts1, std::vector<Eigen::Vector2d>& pts2, Eigen::Matrix3d &F);

		// given corresponding points, get the fundamental matrix
		static bool NormalizedEightPointFundamentalMatrix(std::vector<Eigen::Vector2d>& pts1, std::vector<Eigen::Vector2d>& pts2, Eigen::Matrix3d &F);

	private:
		static bool NormalizeImagePoints(const std::vector<Eigen::Vector2d>& image_points,
			std::vector<Eigen::Vector2d>* normalized_image_points,
			Eigen::Matrix3d* normalization_matrix);

		static double Error(const Eigen::Matrix3d& F, const std::vector<Eigen::Vector2d>& pts1,
			const std::vector<Eigen::Vector2d>& pts2);
	};

}  // namespace objectsfm

#endif  // OBJECTSFM_CAMERA_EIGHT_POINT_FUNDAMENTAL_MATRIX_H_
