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

#ifndef OBJECTSFM_GRAPH_RELATIVE_POSE_FROM_FUNDAMENTAL_H_
#define OBJECTSFM_GRAPH_RELATIVE_POSE_FROM_FUNDAMENTAL_H_

#include <vector>
#include <Eigen/Core>

#include "orientation/relative_pose_from_essential_matrix.h"

namespace objectsfm {

	class RelativePoseFromFundamentalMatrix
	{
	public:
		RelativePoseFromFundamentalMatrix() {};
		~RelativePoseFromFundamentalMatrix() {};

		static bool ReltivePoseFromFMatrix(const Eigen::Matrix3d& F, const std::vector<Eigen::Vector2d>& pts1, const std::vector<Eigen::Vector2d>& pts2,
			double &f1, double &f2, Eigen::Matrix3d& R, Eigen::Vector3d &t);

	private:
		static bool FocalLengthFromFMatrix(const double *fmatrix, double &focal_length1, double &focal_length2);

		static void EMatrixFromFMatrix(const double *fmatrix, double &focal_length1, double &focal_length2, double *ematrix);
	};
}
#endif //OBJECTSFM_GRAPH_RELATIVE_POSE_FROM_FUNDAMENTAL_H_