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

#include "orientation/absolute_pose_via_p4pf.h"

#include <Eigen/SVD>
#include <Eigen/LU>

#include "utils/basic_funcs.h"
#include "orientation/absolute_pose_via_p4pf_helper.h"

#ifndef MIN_
#define MIN_(a,b) ( ((a)<(b)) ? (a):(b) )
#endif // !MIN

namespace objectsfm {

	double AbsolutePoseP4PF::P4PFRANSAC(std::vector<Eigen::Vector3d>& pts_w, std::vector<Eigen::Vector2d>& pts_2d, double & f, RTPose & pose)
	{
		// 
		int num = pts_w.size();
		if (num < 4)
		{
			return false;
		}

		int iter_max = 50;

		double f_best = 0.0;
		RTPose pose_best;
		double error_best = 1000000000.0;
		int iter = 0;
		srand(1);
		while (iter <= iter_max)
		{
			std::vector<Eigen::Vector3d> pts_w_partial;
			std::vector<Eigen::Vector2d> pts_2d_partial;
			std::vector<int> idxs;
			while (idxs.size() != 4)
			{
				int idx = std::rand() % pts_w.size();
				bool isin = false;
				for (size_t i = 0; i < idxs.size(); i++)
				{
					if (idx == idxs[i])
					{
						isin = true;
						break;
					}
				}
				if (!isin)
				{
					idxs.push_back(idx);
					pts_w_partial.push_back(pts_w[idx]);
					pts_2d_partial.push_back(pts_2d[idx]);
				}
			}

			std::vector<double> f_hyps;
			std::vector<RTPose> pose_hyps;
			if (!P4PF(pts_w_partial, pts_2d_partial, f_hyps, pose_hyps))
			{
				continue;
			}

			for (size_t i = 0; i < f_hyps.size(); i++)
			{
				double error_i = Error(pts_w, pts_2d, f_hyps[i], pose_hyps[i]);
				if (error_i < error_best)
				{
					error_best = error_i;
					f_best = f_hyps[i];
					pose_best = pose_hyps[i];
				}
			}
			iter++;
		}

		f = f_best;
		pose = pose_best;

		return std::sqrt(error_best);
	}

	bool AbsolutePoseP4PF::P4PF(std::vector<Eigen::Vector3d>& pts_w, std::vector<Eigen::Vector2d>& pts_2d, std::vector<double>& f_hyps, std::vector<RTPose>& pose_hyps)
	{
		Eigen::Map<const Eigen::Matrix<double, 2, 4> > features(pts_2d[0].data());
		Eigen::Map<const Eigen::Matrix<double, 3, 4> > world_points(pts_w[0].data());

		// Normalize the points such that the mean = 0, variance = sqrt(2.0).
		const Eigen::Vector3d mean_world_point = world_points.rowwise().mean();
		Eigen::Matrix<double, 3, 4> world_point_normalized = world_points.colwise() - mean_world_point;
		const double world_point_variance = world_point_normalized.colwise().norm().mean();
		world_point_normalized /= world_point_variance;

		// Scale 2D data so variance = sqrt(2.0).
		const double features_variance = features.colwise().norm().mean();
		Eigen::Matrix<double, 2, 4> features_normalized = features / features_variance;

		// Precompute monomials.
		const double glab = (world_point_normalized.col(0) -
			world_point_normalized.col(1)).squaredNorm();
		const double glac = (world_point_normalized.col(0) -
			world_point_normalized.col(2)).squaredNorm();
		const double glad = (world_point_normalized.col(0) -
			world_point_normalized.col(3)).squaredNorm();
		const double glbc = (world_point_normalized.col(1) -
			world_point_normalized.col(2)).squaredNorm();
		const double glbd = (world_point_normalized.col(1) -
			world_point_normalized.col(3)).squaredNorm();
		const double glcd = (world_point_normalized.col(2) -
			world_point_normalized.col(3)).squaredNorm();

		if (glab * glac * glad * glbc * glbd * glcd < 1e-15) 
		{
			return false;
		}

		// Call the helper function.
		std::vector<double> focal_length;
		std::vector<Eigen::Vector3d> depths;

		FourPointFocalLengthHelper(glab, glac, glad, glbc, glbd, glcd, features_normalized, &focal_length, &depths);
		if (focal_length.size() == 0) 
		{
			return -1;
		}

		// Get the rotation and translation.
		for (int i = 0; i < focal_length.size(); i++) 
		{
			// Create world points in camera coordinate system.
			Eigen::Matrix<double, 3, 4> adjusted_world_points;
			adjusted_world_points.block<2, 4>(0, 0) = features_normalized;
			adjusted_world_points.row(2).setConstant(focal_length[i]);
			adjusted_world_points.col(1) *= depths[i].x();
			adjusted_world_points.col(2) *= depths[i].y();
			adjusted_world_points.col(3) *= depths[i].z();

			// Fix the scale.
			Eigen::Matrix<double, 6, 1> d;
			d(0) = sqrt(glab / (adjusted_world_points.col(0) -
				adjusted_world_points.col(1)).squaredNorm());
			d(1) = sqrt(glac / (adjusted_world_points.col(0) -
				adjusted_world_points.col(2)).squaredNorm());
			d(2) = sqrt(glad / (adjusted_world_points.col(0) -
				adjusted_world_points.col(3)).squaredNorm());
			d(3) = sqrt(glbc / (adjusted_world_points.col(1) -
				adjusted_world_points.col(2)).squaredNorm());
			d(4) = sqrt(glbd / (adjusted_world_points.col(1) -
				adjusted_world_points.col(3)).squaredNorm());
			d(5) = sqrt(glcd / (adjusted_world_points.col(2) -
				adjusted_world_points.col(3)).squaredNorm());

			const double gta = d.mean();

			adjusted_world_points *= gta;

			// Get the transformation by aligning the points.
			Eigen::Matrix3d rotation;
			Eigen::Vector3d translation;
			GetRigidTransform(world_point_normalized, adjusted_world_points, false, &rotation, &translation);
			translation = world_point_variance * translation - rotation * mean_world_point;
			RTPose pose_temp;
			pose_temp.R = rotation;
			pose_temp.t = translation;
			pose_hyps.push_back(pose_temp);

			focal_length[i] *= features_variance;
			f_hyps.push_back(focal_length[i]);
		}
		
		return f_hyps.size();
	}


	void AbsolutePoseP4PF::GetRigidTransform(const Eigen::Matrix<double, 3, 4>& points1, const Eigen::Matrix<double, 3, 4>& points2, const bool left_handed_coordinates, Eigen::Matrix3d * rotation, Eigen::Vector3d * translation)
	{
		// Move the centroid to th origin.
		const Eigen::Vector3d mean_points1 = points1.rowwise().mean();
		const Eigen::Vector3d mean_points2 = points2.rowwise().mean();

		const Eigen::Matrix<double, 3, 4> points1_shifted = points1.colwise() - mean_points1;
		const Eigen::Matrix<double, 3, 4> points2_shifted = points2.colwise() - mean_points2;

		// Normalize to unit size.
		const Eigen::Matrix<double, 3, 4> points1_normalized =
			points1_shifted.colwise().normalized();
		const Eigen::Matrix<double, 3, 4> points2_normalized =
			points2_shifted.colwise().normalized();

		// Compute the necessary rotation from the difference in points.
		Eigen::Matrix3d rotation_diff = points2_normalized * points1_normalized.transpose();
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotation_diff, Eigen::ComputeFullU | Eigen::ComputeFullV);

		Eigen::Matrix3d s = Eigen::Matrix3d::Zero();
		s(0, 0) = svd.singularValues()(0) < 0 ? -1.0 : 1.0;
		s(1, 1) = svd.singularValues()(1) < 0 ? -1.0 : 1.0;
		Eigen::Matrix3d temp = svd.matrixU() * svd.matrixV().transpose();
		const double sign = temp.determinant() < 0 ? -1.0 : 1.0;

		if (left_handed_coordinates) 
		{
			s(2, 2) = -sign;
		}
		else 
		{
			s(2, 2) = sign;
		}
		*rotation = svd.matrixU() * s * svd.matrixV().transpose();
		*translation = -*rotation * mean_points1 + mean_points2;
	}

	double AbsolutePoseP4PF::Error(std::vector<Eigen::Vector3d>& pts_w, std::vector<Eigen::Vector2d>& pts_2d, double & f, RTPose & pose)
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
			if (abs(error_i) < 1000.0)  // to get rid of gross error
			{
				mse += error_i;
			}
			else
			{
				count_outliers++;
			}
		}

		return mse / (num - count_outliers);
	}

}  // namespace objectsfm
