//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_KMEANS_CLUSTERING_H
#define IL_KMEANS_CLUSTERING_H

#include <il/Array2C.h>
#include <il/Array2D.h>

namespace il {

il::Array2C<float> kmeans_clustering_0(const il::Array2C<float>& point,
                                       int nb_cluster, int nb_iteration);
il::Array2D<float> kmeans_clustering_1(const il::Array2D<float>& point,
                                       int nb_cluster, int nb_iteration);
il::Array2D<float> kmeans_clustering_2(const il::Array2D<float>& point,
                                       int nb_cluster, int nb_iteration);
il::Array2D<float> kmeans_clustering_3(const il::Array2D<float>& point,
                                       int nb_cluster, int nb_iteration);
}  // namespace il

#endif  // IL_KMEANS_CLUSTERING_H
