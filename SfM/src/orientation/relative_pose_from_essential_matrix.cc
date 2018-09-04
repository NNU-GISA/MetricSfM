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

#include "orientation/relative_pose_from_essential_matrix.h"

#include <Eigen/LU>
#include <Eigen/SVD>

namespace objectsfm {

	// based on the properties of the essential matrix on Chapter 9.6.1 in "Multiple View
	// Geometry in Computer Vision" by Hartley, "a 3x3 matrix is an essential matrix if 
	// and only if two of its singular values are equal and the third is zero". E = [t]x * R
	// Define S = [t]x, so S is a skew-symmetric matrix. If E is essentisl matrix, then
	// E should be E = U diag(1, 1, 0) V.t(). Decompose E = UZWV.t() = UZUU.t()WV.t(), where 
	// W = [0 1 0     Z = [0 1 0 
	//     -1 0 0         -1 0 0
	//      0 0 1]         0 0 0]
	// thus that WZ = diag(1, 1, 0) and S = UZU is skew-symmetric, U.t()WV.t() is a rotation 
	// matrix

	bool RelativePoseFromEssentialMatrix::ReltivePoseFromEMatrix(const Eigen::Matrix3d& E, std::vector<Eigen::Vector2d> &pts1, std::vector<Eigen::Vector2d> &pts2, 
		Eigen::Matrix3d& R, Eigen::Vector3d &t)
	{
		// decompose the essential matrix
		Eigen::Matrix3d R1, R2;
		Eigen::Vector3d translation;
		DecomposeEssentialMatrix(E, R1, R2, t);

		// verify 
		std::vector<Eigen::Matrix3d> RHyps(4);
		RHyps[0] = R1;
		RHyps[1] = R1;
		RHyps[2] = R2;
		RHyps[3] = R2;

		std::vector<Eigen::Vector3d> tHyps(4);
		tHyps[0] = -RHyps[0].transpose() * t;
		tHyps[1] = -RHyps[1].transpose() * -t;
		tHyps[2] = -RHyps[2].transpose() * t;
		tHyps[3] = -RHyps[3].transpose() * -t;

		std::vector<int> pts_num(4, 0);
		for (int i = 0; i < pts1.size(); ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				Eigen::Vector3d pt1(pts1[i][0], pts1[i][1], 1.0);
				Eigen::Vector3d pt2(pts2[i][0], pts2[i][1], 1.0);
				if (IsTriangulatedPointInFrontOfCameras(pt1, pt2, RHyps[j], tHyps[j]))
				{
					pts_num[j] ++;
					break;
				}
			}
		}
		int max_num = MAX_(MAX_(pts_num[0], pts_num[1]), MAX_(pts_num[2], pts_num[3]));
		for (int i = 0; i < 4; ++i)
		{
			if (pts_num[i] == max_num)
			{
				R = RHyps[i];
				t = tHyps[i];
				return true;
			}
		}

		return false;
	}

	void RelativePoseFromEssentialMatrix::DecomposeEssentialMatrix(const Eigen::Matrix3d& E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2, Eigen::Vector3d &t)
	{
		Eigen::Matrix3d w;
		w << 0, 1, 0, -1, 0, 0, 0, 0, 1;

		const Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d V = svd.matrixV();
		if (U.determinant() < 0)
		{
			U.col(2) *= -1.0;
		}

		if (V.determinant() < 0) {
			V.col(2) *= -1.0;
		}

		// Possible configurations.
		R1 = U * w * V.transpose();
		R2 = U * w.transpose() * V.transpose();
		t = U.col(2).normalized();
	}

	bool RelativePoseFromEssentialMatrix::IsTriangulatedPointInFrontOfCameras(const Eigen::Vector3d& pt1, const Eigen::Vector3d& pt2, const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
	{
		const Eigen::Vector3d c = -R.transpose()*t;

		const Eigen::Vector3d dir1 = pt1;
		const Eigen::Vector3d dir2 = R.transpose() * pt2;

		const double dir1_sq = dir1.squaredNorm();
		const double dir2_sq = dir2.squaredNorm();
		const double dir1_dir2 = dir1.dot(dir2);
		const double dir1_pos = dir1.dot(c);
		const double dir2_pos = dir2.dot(c);

		return (dir2_sq * dir1_pos - dir1_dir2 * dir2_pos > 0 && dir1_dir2 * dir1_pos - dir1_sq * dir2_pos > 0);
	}

} //objectsfm