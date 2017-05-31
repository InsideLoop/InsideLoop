//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_KMEANS_CLUSTERING_0_H
#define IL_KMEANS_CLUSTERING_0_H

#include <cstddef>

namespace il {

double kmeans_clustering_0(std::size_t nb_point, std::size_t nb_cluster,
                           std::size_t nb_iteration);
double kmeans_clustering_1(std::size_t nb_point, std::size_t nb_cluster,
                           std::size_t nb_iteration);
double kmeans_clustering_2(std::ptrdiff_t nb_point, std::ptrdiff_t nb_cluster,
                           std::ptrdiff_t nb_iteration);
double kmeans_clustering_3(std::ptrdiff_t nb_point, std::ptrdiff_t nb_cluster,
                           std::ptrdiff_t nb_iteration);
double kmeans_clustering_4(int nb_point, int nb_cluster, int nb_iteration);
double kmeans_clustering_5(int nb_point, int nb_cluster, int nb_iteration);
double kmeans_clustering_6(int nb_point, int nb_cluster, int nb_iteration);

double kmeans_clustering_il(int nb_point, int nb_cluster, int nb_iteration);

}  // namespace il

#endif  // IL_KMEANS_CLUSTERING_0_H
