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

#include "absolute_pose_via_epnpf.h"

#include <iostream>
#include <cstring>

#include "orientation/absolute_pose_via_epnp.h"

namespace objectsfm {

	AbsolutePoseEPNPF::AbsolutePoseEPNPF(void)
	{
	}

	AbsolutePoseEPNPF::~AbsolutePoseEPNPF()
	{
	}

	void AbsolutePoseEPNPF::EPNPF(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d, 
		double & f, double f_ratio_min, double f_ratio_max,
		RTPose & pose_absolute, double &error)
	{
		int num_pts_per_iter = 5;
		int max_iter = 100;
		double f_init = f;
		double f_ratio_step = 0.01;

		int num_pts = pts_w.size();
		int num_sample = (f_ratio_max - f_ratio_min) / f_ratio_step;

		error = 1000000.0;
		for (size_t i = 0; i < num_sample; i++)
		{
			double f_i = (f_ratio_min + i * f_ratio_step)*f_init;
			RTPose pose_absolute_i;

			AbsolutePoseEPNP epnp;
			double error_i = 0.0;
			epnp.EPNPRansac(pts_w, pts_2d, f_i, max_iter, pose_absolute_i, error_i);

			if (error_i < error)
			{
				error = error_i;
				f = f_i;
				pose_absolute = pose_absolute_i;
			}
		}
	}

	double AbsolutePoseEPNPF::Error(const std::vector<Eigen::Vector3d>& pts_w, const std::vector<Eigen::Vector2d>& pts_2d, double f, RTPose pose)
	{
		// calculate the projection matrix P
		Eigen::Matrix<double, 3, 4> transformation_matrix;
		transformation_matrix.block<3, 3>(0, 0) = pose.R;
		transformation_matrix.col(3) = pose.t;
		Eigen::Matrix3d camera_matrix = Eigen::DiagonalMatrix<double, 3>(f, f, 1.0);
		Eigen::Matrix<double, 3, 4> projection_matrices = camera_matrix * transformation_matrix;

		// The reprojected point is computed as Xc = P * Xw 
		int num = pts_w.size();
		double mse = 0.0;
		int count_outliers = 0;
		for (size_t i = 0; i < num; i++)
		{
			Eigen::Vector4d pt_w(pts_w[i](0), pts_w[i](1), pts_w[i](2), 1.0);
			Eigen::Vector3d pt_c = projection_matrices * pt_w;
			const Eigen::Vector2d pt_c_2d(pt_c(0) / pt_c(2), pt_c(1) / pt_c(2));

			double error_i = (pt_c_2d - pts_2d[i]).squaredNorm();
			if (abs(error_i) < 50*50)  // to get rid of gross error
			{
				mse += error_i;
			}
			else
			{
				count_outliers++;
			}
		}

		return mse/(num-count_outliers);
	}
}  // namespace objectsfm
