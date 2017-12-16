// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*****************************************************************************
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 2 of the License, or       *
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
 * \brief Data handle class for interaction volumes of mpfa methods.
 *        This class is passed to interaction volumes to store the necessary data in it.
 */
#ifndef DUMUX_DISCRETIZATION_CC_MPFA_INTERACTIONVOLUMEDATAHANDLE_HH
#define DUMUX_DISCRETIZATION_CC_MPFA_INTERACTIONVOLUMEDATAHANDLE_HH

#include <vector>

#include <dumux/common/properties.hh>

namespace Dumux
{

//! Empty data handle class
template<class InteractionVolume>
class EmptyDataHandle
{
public:
    void resize(const InteractionVolume& iv) {}
};

//! Data handle for quantities related to advection
template<class TypeTag, class InteractionVolume, bool EnableAdvection>
class AdvectionDataHandle
{
    //! export matrix & vector types from interaction volume
    using Matrix = typename InteractionVolume::Traits::Matrix;
    using Vector = typename InteractionVolume::Traits::Vector;
    using Scalar = typename Vector::value_type;
    using LocalIndexType = typename InteractionVolume::Traits::LocalIndexType;

    using OutsideDataContainer = std::vector< std::vector<Vector> >;

    static constexpr int dim = GET_PROP_TYPE(TypeTag, GridView)::dimension;
    static constexpr int dimWorld = GET_PROP_TYPE(TypeTag, GridView)::dimensionworld;
    static constexpr int numPhases = GET_PROP_VALUE(TypeTag, NumPhases);

public:

    //! prepare data handle for subsequent fill (normal grids)
    template< int d = dim, int dw = dimWorld, std::enable_if_t<(d==dw), int> = 0>
    void resize(const InteractionVolume& iv)
    {
        // resize transmissibility matrix & pressure vectors
        T_.resize(iv.numFaces(), iv.numKnowns());
        for (unsigned int pIdx = 0; pIdx < numPhases; ++pIdx)
            p_[pIdx].resize(iv.numKnowns());

        // maybe resize gravity container
        static const bool enableGravity = getParamFromGroup<bool>(GET_PROP_VALUE(TypeTag, ModelParameterGroup), "Problem.EnableGravity");
        if (enableGravity)
        {
            CA_.resize(iv.numFaces(), iv.numUnknowns());
            for (unsigned int pIdx = 0; pIdx < numPhases; ++pIdx)
                g_[pIdx].resize(iv.numFaces());
        }
    }


    //! prepare data handle for subsequent fill (surface grids)
    template< int d = dim, int dw = dimWorld, std::enable_if_t<(d<dw), int> = 0>
    void resize(const InteractionVolume& iv)
    {
        static const bool enableGravity = getParamFromGroup<bool>(GET_PROP_VALUE(TypeTag, ModelParameterGroup), "Problem.EnableGravity");

        if (!enableGravity)
        {
            T_.resize(iv.numFaces(), iv.numKnowns());
            outsideT_.resize(iv.numFaces());
            for (unsigned int pIdx = 0; pIdx < numPhases; ++pIdx)
                p_[pIdx].resize(iv.numKnowns());

            for (LocalIndexType i = 0; i < iv.numFaces(); ++i)
            {
                const auto numNeighbors = iv.localScvf(i).neighboringLocalScvIndices().size() - 1;
                outsideT_[i].resize(numNeighbors);
                for (LocalIndexType j = 0; j < numNeighbors; ++j)
                    outsideT_[i][j].resize(iv.numKnowns());
            }
        }

        else
        {
            T_.resize(iv.numFaces(), iv.numKnowns());
            CA_.resize(iv.numFaces(), iv.numUnknowns());
            A_.resize(iv.numUnknowns(), iv.numUnknowns());
            outsideT_.resize(iv.numFaces());

            for (unsigned int pIdx = 0; pIdx < numPhases; ++pIdx)
            {
                p_[pIdx].resize(iv.numKnowns());
                g_[pIdx].resize(iv.numFaces());
                outsideG_[pIdx].resize(iv.numFaces());
            }

            for (LocalIndexType i = 0; i < iv.numFaces(); ++i)
            {
                const auto numNeighbors = iv.localScvf(i).neighboringLocalScvIndices().size() - 1;
                outsideT_[i].resize(numNeighbors);

                for (unsigned int pIdx = 0; pIdx < numPhases; ++pIdx)
                    outsideG_[pIdx][i].resize(numNeighbors);

                for (LocalIndexType j = 0; j < numNeighbors; ++j)
                    outsideT_[i][j].resize(iv.numKnowns());
            }

        }
    }

    //! Access to the iv-wide pressure of one phase
    const Vector& pressures(unsigned int pIdx) const { return p_[pIdx]; }
    Vector& pressures(unsigned int pIdx) { return p_[pIdx]; }

    //! Access to the gravitational flux contributions for one phase
    const Vector& gravity(unsigned int pIdx) const { return g_[pIdx]; }
    Vector& gravity(unsigned int pIdx) { return g_[pIdx]; }

    //! Access to the gravitational flux contributions for all phases
    const std::array< Vector, numPhases >& gravity() const { return g_; }
    std::array< Vector, numPhases >& gravity() { return g_; }

    //! Projection matrix for gravitational acceleration
    const Matrix& advectionCA() const { return CA_; }
    Matrix& advectionCA() { return CA_; }

    //! Additional projection matrix needed on surface grids
    const Matrix& advectionA() const { return A_; }
    Matrix& advectionA() { return A_; }

    //! The transmissibilities associated with advective fluxes
    const Matrix& advectionT() const { return T_; }
    Matrix& advectionT() { return T_; }

    //! The transmissibilities for "outside" faces (used on surface grids)
    const std::vector< std::vector<Vector> >& advectionTout() const { return outsideT_; }
    std::vector< std::vector<Vector> >& advectionTout() { return outsideT_; }

    //! The gravitational acceleration for "outside" faces (used on surface grids)
    const std::array< std::vector<Vector>, numPhases >& gravityOutside() const { return outsideG_; }
    std::array< std::vector<Vector>, numPhases >& gravityOutside() { return outsideG_; }

    //! The gravitational acceleration for one phase on "outside" faces (used on surface grids)
    const std::vector<Vector>& gravityOutside(unsigned int pIdx) const { return outsideG_[pIdx]; }
    std::vector<Vector>& gravityOutside(unsigned int pIdx) { return outsideG_[pIdx]; }

private:
    //! advection-related variables
    Matrix T_;                                               //!< The transmissibilities such that f_i = T_ij*p_j
    Matrix CA_;                                              //!< Matrix to project gravitational acceleration to all scvfs
    Matrix A_;                                               //!< Matrix additionally needed for the projection on surface grids
    std::array< Vector, numPhases > p_;                      //!< The interaction volume-wide phase pressures
    std::array< Vector, numPhases > g_;                      //!< The gravitational acceleration at each scvf (only for enabled gravity)
    std::vector< std::vector<Vector> > outsideT_;            //!< The transmissibilities for "outside" faces (only on surface grids)
    std::array< std::vector<Vector>, numPhases > outsideG_; //!< The gravitational acceleration on "outside" faces (only on surface grids)
};

//! Data handle for quantities related to diffusion
template<class TypeTag, class InteractionVolume, bool EnableDiffusion>
class DiffusionDataHandle
{
    //! export matrix & vector types from interaction volume
    using Matrix = typename InteractionVolume::Traits::Matrix;
    using Vector = typename InteractionVolume::Traits::Vector;
    using OutsideTContainer = std::vector< std::vector<Vector> >;

    static constexpr int dim = GET_PROP_TYPE(TypeTag, GridView)::dimension;
    static constexpr int dimWorld = GET_PROP_TYPE(TypeTag, GridView)::dimensionworld;

    static constexpr int numPhases = GET_PROP_VALUE(TypeTag, NumPhases);
    static constexpr int numComponents = GET_PROP_VALUE(TypeTag, NumComponents);

public:
    //! diffusion caches need to set phase and component index
    void setPhaseIndex(unsigned int phaseIdx) { phaseIdx_ = phaseIdx; }
    void setComponentIndex(unsigned int compIdx) { compIdx_ = compIdx; }

    //! prepare data handle for subsequent fill
    void resize(const InteractionVolume& iv)
    {
        for (unsigned int pIdx = 0; pIdx < numPhases; ++pIdx)
        {
            for (unsigned int cIdx = 0; cIdx < numComponents; ++cIdx)
            {
                if (pIdx == cIdx)
                    continue;

                //! resize transmissibility matrix & mole fraction vector
                T_[pIdx][cIdx].resize(iv.numFaces(), iv.numKnowns());
                xj_[pIdx][cIdx].resize(iv.numKnowns());

                //! resize outsideTij on surface grids
                if (dim < dimWorld)
                {
                    outsideT_[pIdx][cIdx].resize(iv.numFaces());

                    using LocalIndexType = typename InteractionVolume::Traits::LocalIndexType;
                    for (LocalIndexType i = 0; i < iv.numFaces(); ++i)
                      outsideT_[pIdx][cIdx][i].resize(iv.localScvf(i).neighboringLocalScvIndices().size()-1);
                }
            }
        }
    }

    //! Access to the iv-wide mole fractions of a component in one phase
    const Vector& moleFractions(unsigned int pIdx, unsigned int compIdx) const { return xj_[pIdx][compIdx]; }
    Vector& moleFractions(unsigned int pIdx, unsigned int compIdx) { return xj_[pIdx][compIdx]; }

    //! The transmissibilities associated with diffusive fluxes
    const Matrix& diffusionT() const { return T_[phaseIdx_][compIdx_]; }
    Matrix& diffusionT() { return T_[phaseIdx_][compIdx_]; }

    //! The transmissibilities for "outside" faces (used on surface grids)
    const OutsideTContainer& diffusionTout() const { return outsideT_[phaseIdx_][compIdx_]; }
    OutsideTContainer& diffusionTout() { return outsideT_[phaseIdx_][compIdx_]; }

private:
    //! diffusion-related variables
    unsigned int phaseIdx_;                         //!< The phase index set for the context
    unsigned int compIdx_;                          //!< The component index set for the context
    std::array< std::array<Matrix, numComponents>, numPhases > T_;  //!< The transmissibilities such that f_i = T_ij*x_j
    std::array< std::array<Vector, numComponents>, numPhases > xj_; //!< The interaction volume-wide mole fractions
    std::array< std::array<OutsideTContainer, numComponents>, numPhases> outsideT_;
};

//! Data handle for quantities related to heat conduction
template<class TypeTag, class InteractionVolume, bool EnableHeatConduction>
class HeatConductionDataHandle
{
    //! export matrix & vector types from interaction volume
    using Matrix = typename InteractionVolume::Traits::Matrix;
    using Vector = typename InteractionVolume::Traits::Vector;

    static constexpr int dim = GET_PROP_TYPE(TypeTag, GridView)::dimension;
    static constexpr int dimWorld = GET_PROP_TYPE(TypeTag, GridView)::dimensionworld;

public:
    //! prepare data handle for subsequent fill
    void resize(const InteractionVolume& iv)
    {
        //! resize transmissibility matrix & temperature vector
        T_.resize(iv.numFaces(), iv.numKnowns());
        Tj_.resize(iv.numKnowns());

        //! resize outsideTij on surface grids
        if (dim < dimWorld)
        {
            outsideT_.resize(iv.numFaces());

            using LocalIndexType = typename InteractionVolume::Traits::LocalIndexType;
            for (LocalIndexType i = 0; i < iv.numFaces(); ++i)
              outsideT_[i].resize(iv.localScvf(i).neighboringLocalScvIndices().size()-1);
        }
    }

    //! Access to the iv-wide temperatures
    const Vector& temperatures() const { return Tj_; }
    Vector& temperatures() { return Tj_; }

    //! The transmissibilities associated with conductive fluxes
    const Matrix& heatConductionT() const { return T_; }
    Matrix& heatConductionT() { return T_; }

    //! The transmissibilities for "outside" faces (used on surface grids)
    const std::vector< std::vector<Vector> >& heatConductionTout() const { return outsideT_; }
    std::vector< std::vector<Vector> >& heatConductionTout() { return outsideT_; }

private:
    // heat conduction-related variables
    Matrix T_;                                    //!< The transmissibilities such that f_i = T_ij*T_j
    Vector Tj_;                                   //!< The interaction volume-wide temperatures
    std::vector< std::vector<Vector> > outsideT_; //!< The transmissibilities for "outside" faces (only necessary on surface grids)
};

//! Process-dependent data handles when related process is disabled
template<class TypeTag, class InteractionVolume>
class AdvectionDataHandle<TypeTag, InteractionVolume, false> : public EmptyDataHandle<InteractionVolume> {};
template<class TypeTag, class InteractionVolume>
class DiffusionDataHandle<TypeTag, InteractionVolume, false> : public EmptyDataHandle<InteractionVolume> {};
template<class TypeTag, class InteractionVolume>
class HeatConductionDataHandle<TypeTag, InteractionVolume, false> : public EmptyDataHandle<InteractionVolume> {};

//! Interaction volume data handle class
template<class TypeTag, class InteractionVolume>
class InteractionVolumeDataHandle : public AdvectionDataHandle<TypeTag, InteractionVolume, GET_PROP_VALUE(TypeTag, EnableAdvection)>,
                                    public DiffusionDataHandle<TypeTag, InteractionVolume, GET_PROP_VALUE(TypeTag, EnableMolecularDiffusion)>,
                                    public HeatConductionDataHandle<TypeTag, InteractionVolume, GET_PROP_VALUE(TypeTag, EnableEnergyBalance)>
{
    using AdvectionHandle = AdvectionDataHandle<TypeTag, InteractionVolume, GET_PROP_VALUE(TypeTag, EnableAdvection)>;
    using DiffusionHandle = DiffusionDataHandle<TypeTag, InteractionVolume, GET_PROP_VALUE(TypeTag, EnableMolecularDiffusion)>;
    using HeatConductionHandle = HeatConductionDataHandle<TypeTag, InteractionVolume, GET_PROP_VALUE(TypeTag, EnableEnergyBalance)>;

public:
    //! prepare data handles for subsequent fills
    void resize(const InteractionVolume& iv)
    {
        AdvectionHandle::resize(iv);
        DiffusionHandle::resize(iv);
        HeatConductionHandle::resize(iv);
    }
};

} // end namespace Dumux

#endif
