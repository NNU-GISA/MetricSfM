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

#ifndef OBJECTSFM_GRAPH_RELATIVE_POSE_FROM_ESSENTIAL_H_
#define OBJECTSFM_GRAPH_RELATIVE_POSE_FROM_ESSENTIAL_H_

#ifndef MAX_
#define MAX_(a,b) ( ((a)>(b)) ? (a):(b) )
#endif // !MAX

#include <vector>
#include <Eigen/Core>

namespace objectsfm {

	class RelativePoseFromEssentialMatrix
	{
	public:
		RelativePoseFromEssentialMatrix() {};
		~RelativePoseFromEssentialMatrix() {};

		static bool ReltivePoseFromEMatrix(const Eigen::Matrix3d& E, std::vector<Eigen::Vector2d> &pts1, std::vector<Eigen::Vector2d> &pts2, Eigen::Matrix3d& R, Eigen::Vector3d &t);

	private:
		static void DecomposeEssentialMatrix(const Eigen::Matrix3d& E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2, Eigen::Vector3d &t);

		static bool IsTriangulatedPointInFrontOfCameras(const Eigen::Vector3d& pt1, const Eigen::Vector3d& pt2, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
	};

}
#endif //OBJECTSFM_CAMERA_RELATIVE_POSE_FROM_ESSENTIAL_H_