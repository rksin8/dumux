// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*****************************************************************************
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 3 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
/*!
 * \file
 * \ingroup StaggeredDiscretization
 * \copydoc Dumux::FreeFlowStaggeredSubControlVolumeFace
 */
#ifndef DUMUX_DISCRETIZATION_STAGGERED_FREE_FLOW_SUBCONTROLVOLUMEFACE_HH
#define DUMUX_DISCRETIZATION_STAGGERED_FREE_FLOW_SUBCONTROLVOLUMEFACE_HH

#include <array>
#include <utility>
#include <dune/common/fvector.hh>
#include <dune/geometry/type.hh>
#include <dune/geometry/multilineargeometry.hh>

#include <dumux/common/indextraits.hh>
#include <dumux/common/typetraits/isvalid.hh>
#include <dumux/discretization/subcontrolvolumefacebase.hh>
#include <dumux/discretization/staggered/subcontrolvolumeface.hh>
#include <dumux/discretization/staggered/freeflow/staggeredgeometryhelper.hh>

#include <typeinfo>

namespace Dumux {

#ifndef DOXYGEN
namespace Detail {
// helper struct detecting if the container class storing the scvf's corners has a resize function
// for g++ > 5.3, this can be replaced by a lambda
struct hasResize
{
    template<class Container>
    auto operator()(Container&& c)
    -> decltype(c.resize(1))
    {}
};
} // end namespace Detail
#endif

/*!
 * \ingroup StaggeredDiscretization
 * \brief Default traits class to be used for the sub-control volume faces
 *        for the free-flow staggered finite volume scheme
 * \tparam GridView the type of the grid view
 * \tparam upwindSchemeOrder the order of the upwind scheme
 */
template<class GridView, int upwindSchemeOrder>
struct FreeFlowStaggeredDefaultScvfGeometryTraits
{
    using GridIndexType = typename IndexTraits<GridView>::GridIndex;
    using LocalIndexType = typename IndexTraits<GridView>::LocalIndex;
    using Scalar = typename GridView::ctype;
    using PairData = typename FreeFlowStaggeredGeometryHelper<GridView, upwindSchemeOrder>::PairData;
    using AxisData = typename FreeFlowStaggeredGeometryHelper<GridView, upwindSchemeOrder>::AxisData;


    using Grid = typename GridView::Grid;
    static constexpr int dim = Grid::dimension;
    static constexpr int dimWorld = Grid::dimensionworld;

    // we use geometry traits that use static corner vectors to and a fixed geometry type
    template <class ct>
    struct ScvfMLGTraits : public Dune::MultiLinearGeometryTraits<ct>
    {
        // we use static vectors to store the corners as we know
        // the number of corners in advance (2^(dim-1) corners (1<<(dim-1))
        template< int mydim, int cdim >
        struct CornerStorage
        {
            using Type = std::array< Dune::FieldVector< ct, cdim >, (1<<(dim-1)) >;
        };
    };

    using Geometry = Dune::MultiLinearGeometry<Scalar, dim-1, dimWorld, ScvfMLGTraits<Scalar> >;
    using CornerStorage = typename ScvfMLGTraits<Scalar>::template CornerStorage<dim-1, dimWorld>::Type;
    using GlobalPosition = typename CornerStorage::value_type;
};

/*!
 * \ingroup StaggeredDiscretization
 * \brief Class for a sub control volume face in the staggered method, i.e a part of the boundary
 *        of a sub control volume we compute fluxes on. This is a specialization for free flow models.
 */
template<class GV,
         int upwindSchemeOrder,
         class T = FreeFlowStaggeredDefaultScvfGeometryTraits<GV, upwindSchemeOrder>>
class FreeFlowStaggeredSubControlVolumeFace
: public SubControlVolumeFaceBase<FreeFlowStaggeredSubControlVolumeFace<GV, upwindSchemeOrder, T>, T>
{
    using ThisType = FreeFlowStaggeredSubControlVolumeFace<GV, upwindSchemeOrder, T>;
    using ParentType = SubControlVolumeFaceBase<ThisType, T>;
    using Geometry = typename T::Geometry;
    using GridIndexType = typename IndexTraits<GV>::GridIndex;
    using LocalIndexType = typename IndexTraits<GV>::LocalIndex;
    using CornerStorage = typename T::CornerStorage;

    using PairData = typename T::PairData;
    using AxisData = typename T::AxisData;

    using Scalar = typename T::Scalar;
    static const int dim = GV::dimension;

    static constexpr int numPairs = 2 * (dim - 1);

    static constexpr bool useHigherOrder = upwindSchemeOrder > 1;

public:
    using GlobalPosition = typename T::GlobalPosition;

    //! State the traits public and thus export all types
    using Traits = T;

    // The default constructor
    FreeFlowStaggeredSubControlVolumeFace() = default;

    //! Constructor with intersection
    template <class Intersection, class GeometryHelper>
    FreeFlowStaggeredSubControlVolumeFace(const Intersection& is,
                                          const typename Intersection::Geometry& isGeometry,
                                          GridIndexType scvfIndex,
                                          const std::vector<GridIndexType>& scvIndices,
                                          const GeometryHelper& geometryHelper)
    : ParentType(),
      geomType_(isGeometry.type()),
      area_(isGeometry.volume()),
      center_(isGeometry.center()),
      unitOuterNormal_(is.centerUnitOuterNormal()),
      scvfIndex_(scvfIndex),
      scvIndices_(scvIndices),
      boundary_(is.boundary()),

      axisData_(geometryHelper.axisData()),
      pairData_(std::move(geometryHelper.pairData())),
      localFaceIdx_(geometryHelper.localFaceIndex()),
      dirIdx_(geometryHelper.directionIndex()),
      outerNormalSign_(sign(unitOuterNormal_[directionIndex()])),
      isGhostFace_(false)
      {
          using HasResize = decltype(isValid(Detail::hasResize())(corners_));
          maybeResizeCornerStorage_(HasResize{}, isGeometry.corners());
          for (int i = 0; i < isGeometry.corners(); ++i)
              corners_[i] = isGeometry.corner(i);
      }

    //! The center of the sub control volume face
    const GlobalPosition& center() const
    {
        return center_;
    }

    //! The position of the dof living on the face
    const GlobalPosition& dofPosition() const
    {
        return center_;
    }

    //! The integration point for flux evaluations in global coordinates
    const GlobalPosition& ipGlobal() const
    {
        // Return center for now
        return center_;
    }

    //! The area of the sub control volume face
    Scalar area() const
    {
        return area_;
    }

    //! Returns bolean if the sub control volume face is on the boundary
    bool boundary() const
    {
        return boundary_;
    }

    //! The unit outer normal vector
    const GlobalPosition& unitOuterNormal() const
    {
        return unitOuterNormal_;
    }

    //! Index of the inside sub control volume for spatial param evaluation
    GridIndexType insideScvIdx() const
    {
        return scvIndices_[0];
    }

    //! index of the outside sub control volume for spatial param evaluation
    GridIndexType outsideScvIdx() const
    {
        return scvIndices_[1];
    }

    //! The global index of this sub control volume face
    GridIndexType index() const
    {
        return scvfIndex_;
    }

    //! The positions of the corners
    const GlobalPosition& corner(unsigned int localIdx) const
    {
        assert(localIdx < corners_.size() && "provided index exceeds the number of corners");
        return corners_[localIdx];
    }

    //! The geometry of the sub control volume face
    const Geometry geometry() const
    {
        return Geometry(geomType_, corners_);
    }

    //! The local index of this sub control volume face
    LocalIndexType localFaceIdx() const
    {
        return localFaceIdx_;
    }

    //! Returns the dirction index of the facet (0 = x, 1 = y, 2 = z)
    unsigned int directionIndex() const
    {
        return dirIdx_;
    }

    //! Returns whether the unitNormal of the face points in positive coordinate direction
    bool normalInPosCoordDir() const
    {
        return directionSign() > 0;
    }

    //! Returns the sign of the unit outer normal's vector
    int directionSign() const
    {
        return outerNormalSign_;
    }

    //! Returns the data for one sub face
    const PairData& pairData(const int idx) const
    {
        return pairData_[idx];
    }

    //! Return an array of all pair data
    const std::array<PairData, numPairs>& pairData() const
    {
        return pairData_;
    }

    //! Return an array of all pair data
    const AxisData& axisData() const
    {
        return axisData_;
    }

    //! Returns @c true if the face is a ghost face
    bool isGhostFace() const
    {
        return isGhostFace_;
    }

    //! Returns the length of the face in a certain direction (adaptation of area() for 3d)
    Scalar faceLength(const int localSubFaceIdx) const
    {
        if (dim == 3)
        {
            if (localSubFaceIdx < 2)
                return (corner(1) - corner(0)).two_norm();
            else
                return (corner(2) - corner(0)).two_norm();
        }
        else
            return (corner(1) - corner(0)).two_norm();
    }

   /*!
    * \brief Check if the face has a parallel neighbor
    *
    * \param localSubFaceIdx The local index of the subface
    * \param parallelDegreeIdx The index describing how many faces away from the self face
    */
    bool hasParallelNeighbor(const int localSubFaceIdx, const int parallelDegreeIdx) const
    {
        return pairData(localSubFaceIdx).hasParallelNeighbor[parallelDegreeIdx];
    }

   /*!
    * \brief Check if the face has a half parallel neighbor
    *
    * \param localSubFaceIdx The local index of the subface
    *
    * ------------
    * |          |
    * |          |
    * |          |
    * -----------------------
    * | yyyyyyyy s          |
    * | yyyyyyyy s          |
    * | yyyyyyyy s          |
    * -----------------------
    * In this corner geometry, subcontrolvolumeface s belonging to the element filled by 'y's has a
    * 'halfParallelNeighbor'. This means it has a parallel neighbor itself, but the subcontrolvolumeface that has
    * the same dofIndex does not.
    */
    bool hasHalfParallelNeighbor(const int localSubFaceIdx) const
    {
        return pairData(localSubFaceIdx).hasHalfParallelNeighbor;
    }

    /*!
    * \brief Check if the face has a corner parallel neighbor
    *
    * \param localSubFaceIdx The local index of the subface
    *
    * ------------
    * | yyyyyyyy s
    * | yyyyyyyy s
    * | yyyyyyyy s
    * -----------------------
    * |          |          |
    * |          |          |
    * |          |          |
    * -----------------------
    * In this corner geometry, subcontrolvolumeface s belonging to the element filled by 'y's has a
    * 'cornerParallelNeighbor'.
    */
    bool hasCornerParallelNeighbor(const int localSubFaceIdx) const
    {
        return pairData(localSubFaceIdx).hasCornerParallelNeighbor;
    }

   /*!
    * \brief Check if the face has an outer normal neighbor
    *
    * \param localSubFaceIdx The local index of the subface
    */
    bool hasOuterLateral(const int localSubFaceIdx) const
    {
        return pairData(localSubFaceIdx).hasOuterLateral;
    }

   /*!
    * \brief Check if the face has a backward neighbor
    *
    * \param backwardIdx The index describing how many faces backward this dof is from the opposite face
    */
    template<bool enable = useHigherOrder, std::enable_if_t<enable, int> = 0>
    bool hasBackwardNeighbor(const int backwardIdx) const
    {
        return axisData().hasBackwardNeighbor[backwardIdx];
    }

   /*!
    * \brief Check if the face has a forward neighbor
    *
    * \param forwardIdx The index describing how many faces forward this dof is of the self face
    */
    template<bool enable = useHigherOrder, std::enable_if_t<enable, int> = 0>
    bool hasForwardNeighbor(const int forwardIdx) const
    {
        return axisData().hasForwardNeighbor[forwardIdx];
    }

    //! Returns the dof of the face
    GridIndexType dofIndex() const
    {
        return axisData().selfDof;
    }

    //! Returns the dof of the opposing face
    GridIndexType dofIndexOpposingFace() const
    {
        return axisData().oppositeDof;
    }

    //! Returns the dof the first forward face
    GridIndexType dofIndexForwardFace() const
    {
        return axisData().inAxisForwardDofs[0];
    }

    //! Returns the dof of the first backward face
    GridIndexType dofIndexBackwardFace() const
    {
        return axisData().inAxisBackwardDofs[0];
    }

    //! Returns the distance between the face and the opposite one
    Scalar selfToOppositeDistance() const
    {
        return axisData().selfToOppositeDistance;
    }

    /*!
    * \brief Returns the distance between the parallel dofs
    *
    * \param localSubFaceIdx The local index of the subface
    * \param parallelDegreeIdx The index describing how many faces away from the self
    */
    Scalar parallelDofsDistance(const int localSubFaceIdx, const int parallelDegreeIdx) const
    {
        if (parallelDegreeIdx == 0)
            return (faceLength(localSubFaceIdx) + pairData(localSubFaceIdx).parallelCellWidths[0]) * 0.5;
            // pairData(localSubFaceIdx).parallelCellWidths[0]) will return 0.0 if the subface perpendicular the scvf lies on a boundary
        else
        {
            assert((parallelDegreeIdx == 1) && "Only the width of the first two parallel cells (indicies 0 and 1) is stored for each scvf.");
            return (pairData(localSubFaceIdx).parallelCellWidths[0] + pairData(localSubFaceIdx).parallelCellWidths[1]) * 0.5;
        }
    }

    /*!
    * \brief Returns a copy of the own scvf whith a user-specified center position.
    *        This is needed for retrieving boundary conditions when the actual center does not coincide with the position
    *        on which the boundary condition is defined.
    *
    * \param pos The desired position of the boundary scvf's center
    */
    FreeFlowStaggeredSubControlVolumeFace makeBoundaryFace(const GlobalPosition& pos) const
    {
        FreeFlowStaggeredSubControlVolumeFace boundaryFace = *this;
        boundaryFace.center_ = pos;
        boundaryFace.boundary_ = true;
        boundaryFace.isGhostFace_ = true;
        return boundaryFace;
    }

private:
    void maybeResizeCornerStorage_(std::true_type /*hasResize*/, std::size_t size)
    { corners_.resize(size); }

    void maybeResizeCornerStorage_(std::false_type /*hasResize*/, std::size_t size)
    {}

    Dune::GeometryType geomType_;
    CornerStorage corners_;
    Scalar area_;
    GlobalPosition center_;
    GlobalPosition unitOuterNormal_;
    GridIndexType scvfIndex_;
    std::vector<GridIndexType> scvIndices_;
    bool boundary_;

    Scalar selfToOppositeDistance_;
    AxisData axisData_;
    std::array<PairData, numPairs> pairData_;

    int localFaceIdx_;
    unsigned int dirIdx_;
    int outerNormalSign_;
    bool isGhostFace_;
};

} // end namespace Dumux

#endif // DUMUX_DISCRETIZATION_STAGGERED_FREE_FLOW_SUBCONTROLVOLUMEFACE_HH
