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

#include "orientation/absolute_pose_estimation.h"

#include "orientation/absolute_pose_via_epnpf.h"
#include "orientation/absolute_pose_via_epnp.h"
#include "orientation/absolute_pose_via_p4pf.h"

#ifndef MIN_
#define MIN_(a,b) ( ((a)<(b)) ? (a):(b) )
#endif // !MIN

namespace objectsfm {
	bool AbsolutePoseEstimation::AbsolutePoseWithoutFocalLength(std::vector<Eigen::Vector3d>& pts_w, 
		std::vector<Eigen::Vector2d>& pts_2d, double & f, RTPose & pose_absolute, std::vector<double> &errors, double &avg_error)
	{
		double f_ratio_min = 0.5;
		double f_ratio_max = 4.00;
		double error = 0.0;
		AbsolutePoseEPNPF::EPNPF(pts_w, pts_2d, f, f_ratio_min, f_ratio_max, pose_absolute, error);
		Error(pts_w, pts_2d, f, pose_absolute, errors, error);

		avg_error = error;

		return true;
	}

	bool AbsolutePoseEstimation::AbsolutePoseWithFocalLength(std::vector<Eigen::Vector3d>& pts_w, 
		std::vector<Eigen::Vector2d>& pts_2d, double f, RTPose & pose_absolute, std::vector<double> &errors, double &avg_error)
	{
		int num_pts_per_iter = 5;
		int max_iter = 100;
		double error = 0.0;

		AbsolutePoseEPNP epnp;
		epnp.EPNPRansac(pts_w, pts_2d, f, max_iter, pose_absolute, error);

		// calculate error
		Error(pts_w, pts_2d, f, pose_absolute, errors, error);

		avg_error = error;

		return true;
	}

	bool AbsolutePoseEstimation::AbsolutePoseWithRotation(std::vector<Eigen::Vector3d>& pts_w, 
		std::vector<Eigen::Vector2d>& pts_2d, Eigen::Matrix3d R, double & f, RTPose & pose_absolute, 
		std::vector<double> &errors, double &avg_error)
	{
		return false;
	}

	void AbsolutePoseEstimation::Error(const std::vector<Eigen::Vector3d>& pts_w, const std::vector<Eigen::Vector2d>& pts_2d, 
		double f, RTPose pose, std::vector<double>& errors, double & error)
	{
		// calculate the projection matrix P
		Eigen::Matrix<double, 3, 4> transformation_matrix;
		transformation_matrix.block<3, 3>(0, 0) = pose.R;
		transformation_matrix.col(3) = pose.t;
		Eigen::Matrix3d camera_matrix = Eigen::DiagonalMatrix<double, 3>(f, f, 1.0);
		Eigen::Matrix<double, 3, 4> projection_matrices = camera_matrix * transformation_matrix;

		// The reprojected point is computed as Xc = P * Xw 
		int num = pts_w.size();
		errors.resize(num, 1000.0);
		error = 0.0;
		int count_inliers = 0;
		for (size_t i = 0; i < num; i++)
		{
			Eigen::Vector4d pt_w(pts_w[i](0), pts_w[i](1), pts_w[i](2), 1.0);
			Eigen::Vector3d pt_c = projection_matrices * pt_w;
			const Eigen::Vector2d pt_c_2d(pt_c(0) / pt_c(2), pt_c(1) / pt_c(2));

			double error_i = (pt_c_2d - pts_2d[i]).norm();
			if (abs(error_i) < 10.0)  // to get rid of gross error
			{
				errors[i] = error_i;
				error += error_i * error_i;
				count_inliers++;
			}
		}

		if (count_inliers == 0)
		{
			error = 10000.0;
			return;
		}
		error = std::sqrt(error / count_inliers);
	}
}  // namespace objectsfm

