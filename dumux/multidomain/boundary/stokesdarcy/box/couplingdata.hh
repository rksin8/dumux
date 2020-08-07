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
 * \ingroup StokesDarcyCoupling
 * \copydoc Dumux::StokesDarcyCouplingData
 */

#ifndef DUMUX_STOKES_DARCY_BOX_COUPLINGDATA_HH
#define DUMUX_STOKES_DARCY_BOX_COUPLINGDATA_HH

#include <dune/geometry/quadraturerules.hh>

#include <dumux/multidomain/boundary/stokesdarcy/couplingdata.hh>
#include <dumux/multidomain/couplingmanager.hh> //TODO by Lars: Why is this the right coupling manager, not the one in this folder?

//Needed for nMomentumCouplingCondition
#include <dumux/freeflow/navierstokes/staggered/velocitygradients.hh>
#include <dumux/discretization/staggered/freeflow/boundarytypes.hh>
#include <optional>

namespace Dumux {
/*!
 * \ingroup StokesDarcyCoupling
 * \brief A base class which provides some common methods used for Stokes-Darcy coupling.
 */
template<class MDTraits, class CouplingManager, bool enableEnergyBalance>
class StokesDarcyCouplingDataBoxBase : public StokesDarcyCouplingDataImplementationBase<MDTraits, CouplingManager>
{
    using ParentType = StokesDarcyCouplingDataImplementationBase<MDTraits, CouplingManager>;

    using Scalar = typename MDTraits::Scalar;

    template<std::size_t id> using SubDomainTypeTag = typename MDTraits::template SubDomain<id>::TypeTag;
    template<std::size_t id> using GridGeometry = GetPropType<SubDomainTypeTag<id>, Properties::GridGeometry>;
    template<std::size_t id> using Element = typename GridGeometry<id>::GridView::template Codim<0>::Entity;
    template<std::size_t id> using FVElementGeometry = typename GridGeometry<id>::LocalView;
    template<std::size_t id> using SubControlVolumeFace = typename GridGeometry<id>::LocalView::SubControlVolumeFace;
    template<std::size_t id> using SubControlVolume = typename GridGeometry<id>::LocalView::SubControlVolume;
    template<std::size_t id> using Indices = typename GetPropType<SubDomainTypeTag<id>, Properties::ModelTraits>::Indices;
    template<std::size_t id> using ElementVolumeVariables = typename GetPropType<SubDomainTypeTag<id>, Properties::GridVolumeVariables>::LocalView;
    template<std::size_t id> using VolumeVariables = typename GetPropType<SubDomainTypeTag<id>, Properties::GridVolumeVariables>::VolumeVariables;
    template<std::size_t id> using Problem = GetPropType<SubDomainTypeTag<id>, Properties::Problem>;
    template<std::size_t id> using FluidSystem = GetPropType<SubDomainTypeTag<id>, Properties::FluidSystem>;
    template<std::size_t id> using ModelTraits = GetPropType<SubDomainTypeTag<id>, Properties::ModelTraits>;

    static constexpr auto stokesIdx = CouplingManager::stokesIdx;
    static constexpr auto darcyIdx = CouplingManager::darcyIdx;

    // Needed for velocityPorousMedium
    using VelocityVector = typename Element<stokesIdx>::Geometry::GlobalCoordinate;
    // Needed for nMomentum
    template<std::size_t id> using BoundaryTypes = GetPropType<SubDomainTypeTag<id>, Properties::BoundaryTypes>;
    using StokesVelocityGradients = StaggeredVelocityGradients<Scalar, GridGeometry<stokesIdx>, BoundaryTypes<stokesIdx>, Indices<stokesIdx>>;

    using AdvectionType = GetPropType<SubDomainTypeTag<darcyIdx>, Properties::AdvectionType>;
    using DarcysLaw = DarcysLawImplementation<SubDomainTypeTag<darcyIdx>, GridGeometry<darcyIdx>::discMethod>;
    using ForchheimersLaw = ForchheimersLawImplementation<SubDomainTypeTag<darcyIdx>, GridGeometry<darcyIdx>::discMethod>;

    using DiffusionCoefficientAveragingType = typename StokesDarcyCouplingOptions::DiffusionCoefficientAveragingType;

public:
    StokesDarcyCouplingDataBoxBase(const CouplingManager& couplingmanager): ParentType(couplingmanager) {}

    using ParentType::couplingPhaseIdx;

    /*!
     * \brief Returns the momentum flux across the coupling boundary.
     *
     * Calls (old/new)MomentumCouplingCondition depending on the value of the parameter "Problem.NewIc"
     * Defaults to oldMomentumCouplingCondition
     *
     */
    template<class ElementFaceVariables>
    Scalar momentumCouplingCondition(const Element<stokesIdx>& element,
                                     const FVElementGeometry<stokesIdx>& fvGeometry,
                                     const ElementVolumeVariables<stokesIdx>& stokesElemVolVars,
                                     const ElementFaceVariables& stokesElemFaceVars,
                                     const SubControlVolumeFace<stokesIdx>& scvf) const
    {
        static const bool newIc_ = getParamFromGroup<bool>("Problem", "NewIc", false);
        if (newIc_){
            return newMomentumCouplingCondition(element, fvGeometry, stokesElemVolVars, stokesElemFaceVars, scvf);
        }
        else{
            return oldMomentumCouplingCondition(element, fvGeometry, stokesElemVolVars, stokesElemFaceVars, scvf);
        }
    }

    /*!
     * \brief Returns the momentum flux across the coupling boundary.
     *
     * For the normal momentum coupling, the porous medium side of the coupling condition
     * is evaluated, i.e. -[p n]^pm.
     *
     */
    template<class ElementFaceVariables>
    Scalar oldMomentumCouplingCondition(const Element<stokesIdx>& element,
                                     const FVElementGeometry<stokesIdx>& fvGeometry,
                                     const ElementVolumeVariables<stokesIdx>& stokesElemVolVars,
                                     const ElementFaceVariables& stokesElemFaceVars,
                                     const SubControlVolumeFace<stokesIdx>& scvf) const
    {
        Scalar momentumFlux(0.0);
        const auto& stokesContext = this->couplingManager().stokesCouplingContextVector(element, scvf);

        // integrate darcy pressure over each coupling segment and average
        for (const auto& data : stokesContext)
        {
            if (scvf.index() == data.stokesScvfIdx)
            {
                const auto darcyPhaseIdx = couplingPhaseIdx(darcyIdx);
                const auto& elemVolVars = *(data.elementVolVars);
                const auto& darcyFvGeometry = data.fvGeometry;
                const auto& localBasis = darcyFvGeometry.feLocalBasis();

                // do second order integration as box provides linear functions
                static constexpr int darcyDim = GridGeometry<darcyIdx>::GridView::dimension;
                const auto& rule = Dune::QuadratureRules<Scalar, darcyDim-1>::rule(data.segmentGeometry.type(), 2);
                for (const auto& qp : rule)
                {
                    const auto& ipLocal = qp.position();
                    const auto& ipGlobal = data.segmentGeometry.global(ipLocal);
                    const auto& ipElementLocal = data.element.geometry().local(ipGlobal);

                    std::vector<Dune::FieldVector<Scalar, 1>> shapeValues;
                    localBasis.evaluateFunction(ipElementLocal, shapeValues);

                    Scalar pressure = 0.0;
                    for (const auto& scv : scvs(data.fvGeometry))
                        pressure += elemVolVars[scv].pressure(darcyPhaseIdx)*shapeValues[scv.indexInElement()][0];

                    momentumFlux += pressure*data.segmentGeometry.integrationElement(qp.position())*qp.weight();
                }
            }
        }

        momentumFlux /= scvf.area();

        // normalize pressure
        if(getPropValue<SubDomainTypeTag<stokesIdx>, Properties::NormalizePressure>())
            momentumFlux -= this->couplingManager().problem(stokesIdx).initial(scvf)[Indices<stokesIdx>::pressureIdx];

        momentumFlux *= scvf.directionSign();

        return momentumFlux;
    }

    /*!
    * \brief Returns the momentum flux across the coupling boundary which is calculated according to the new interface condition
    *
    * For the new normal momentum coupling, the porous medium side and also the stokes side is evaluated.
    * [p]^pm + N_s^{bl} \tau T n
    *
    */
    template<class ElementFaceVariables>
    Scalar newMomentumCouplingCondition(const Element<stokesIdx>& element,
                                      const FVElementGeometry<stokesIdx>& fvGeometry,
                                      const ElementVolumeVariables<stokesIdx>& stokesElemVolVars,
                                      const ElementFaceVariables& stokesElemFaceVars,
                                      const SubControlVolumeFace<stokesIdx>& scvf) const
    {
      Scalar momentumFlux(0.0);
      //######## darcy contribution #################
        momentumFlux = oldMomentumCouplingCondition(element,fvGeometry,stokesElemVolVars,stokesElemFaceVars,scvf);
        momentumFlux*= scvf.directionSign(); //Revert sign change, should be applied to total flux

      //######## New stokes contribution #################
        static const bool unsymmetrizedGradientForBeaversJoseph = getParamFromGroup<bool>(this->couplingManager().problem(stokesIdx).paramGroup(),
                                                           "FreeFlow.EnableUnsymmetrizedVelocityGradientForBeaversJoseph", false);
        // TODO: how to deprecate unsymmBeaverJoseph?
        // Replace unsymmetrizedGradientForBeaversJoseph below by false, when deprecation period expired
        static const bool unsymmetrizedGradientForIC = getParamFromGroup<bool>(this->couplingManager().problem(stokesIdx).paramGroup(),
                                                           "FreeFlow.EnableUnsymmetrizedVelocityGradientForIC", unsymmetrizedGradientForBeaversJoseph);
      const std::size_t numSubFaces = scvf.pairData().size();

      // Account for all sub faces
      for (int localSubFaceIdx = 0; localSubFaceIdx < numSubFaces; ++localSubFaceIdx)
      {
        const auto eIdx = scvf.insideScvIdx();
        const auto& lateralScvf = fvGeometry.scvf(eIdx, scvf.pairData(localSubFaceIdx).localLateralFaceIdx);

        // Create a boundaryTypes object (will be empty if not at a boundary)
        std::optional<BoundaryTypes<stokesIdx>> currentScvfBoundaryTypes;
        if (scvf.boundary())
            {
            currentScvfBoundaryTypes.emplace(this->couplingManager().problem(stokesIdx).boundaryTypes(element, scvf));
            }

        std::optional<BoundaryTypes<stokesIdx>> lateralFaceBoundaryTypes;
        if (lateralScvf.boundary())
        {
          lateralFaceBoundaryTypes.emplace(this->couplingManager().problem(stokesIdx).boundaryTypes(element, lateralScvf));
        }


        // Get velocity gradients
        const Scalar velocityGrad_ji = StokesVelocityGradients::velocityGradJI(
          this->couplingManager().problem(stokesIdx), element, fvGeometry, scvf , stokesElemFaceVars[scvf],
          currentScvfBoundaryTypes, lateralFaceBoundaryTypes, localSubFaceIdx);
        Scalar velocityGrad_ij = StokesVelocityGradients::velocityGradIJ(
            this->couplingManager().problem(stokesIdx), element, fvGeometry, scvf , stokesElemFaceVars[scvf],
            currentScvfBoundaryTypes, lateralFaceBoundaryTypes, localSubFaceIdx);

        //TODO: Remove calculation above in this case
            if (unsymmetrizedGradientForIC)
            {
            velocityGrad_ij = 0.0;
        }

        // Calculate stokes contribution to momentum flux: N_s^{bl} \tau T n
        const Scalar Nsbl = this->couplingManager().problem(darcyIdx).spatialParams().factorNMomentumAtPos(scvf.center());
        const Scalar viscosity = stokesElemVolVars[scvf.insideScvIdx()].viscosity();
        // Averaging the gradients over the subfaces to get evaluation at the center
        momentumFlux -= 1.0/numSubFaces * viscosity * Nsbl * (velocityGrad_ji + velocityGrad_ij);
      }
      momentumFlux *= scvf.directionSign();
      return momentumFlux;
    }

    /*!
     * \brief Returns the averaged velocity vector at the interface of the porous medium according to darcys law
     *
     * Calls standardPorousMediumVelocity/newPorousMediumInterfaceVelocity depending on the value of the parameter "Problem.NewIc"
     * Defaults to standardPorousMediumVelocity
     *
     */
    VelocityVector porousMediumVelocity(const Element<stokesIdx>& element, const SubControlVolumeFace<stokesIdx>& scvf) const
    {
        static const bool newIc_ = getParamFromGroup<bool>("Problem", "NewIc", false);
        if (newIc_){
            return newPorousMediumInterfaceVelocity(element, scvf);
        }
        else{
            return standardPorousMediumVelocity(element, scvf);
        }
    }

    /*!
    * \brief Returns the averaged velocity vector at the interface of the porous medium according to darcys law
    *
    * The tangential porous medium velocity needs to be evaluated for the tangential coupling at the
    * stokes-darcy interface. We use darcys law and perform an integral average over all coupling segments.
    *
    */
    VelocityVector standardPorousMediumVelocity(const Element<stokesIdx>& element, const SubControlVolumeFace<stokesIdx>& scvf) const
    {
      static constexpr int darcyDim = GridGeometry<darcyIdx>::GridView::dimension;
      using JacobianType = Dune::FieldMatrix<Scalar, 1, darcyDim>;
      std::vector<JacobianType> shapeDerivatives;
      std::vector<Dune::FieldVector<Scalar, 1>> shapeValues;

      VelocityVector velocity(0.0); //  velocity darcy
      VelocityVector gradP(0.0);    // pressure gradient darcy
      Scalar rho(0.0);              // density darcy
      Scalar intersectionLength = 0.0; //(total)intersection length, could differ from scfv length

      //Getting needed information from the darcy domain
      const auto& stokesContext = this->couplingManager().stokesCouplingContextVector(element, scvf);
      static const bool enableGravity = getParamFromGroup<bool>(this->couplingManager().problem(darcyIdx).paramGroup(), "Problem.EnableGravity");

      // Iteration over the different coupling segments
      for (const auto& data : stokesContext)
      {
        //We are on (one of) the correct scvf(s)
        if (scvf.index() == data.stokesScvfIdx)
        {
          const auto darcyPhaseIdx = couplingPhaseIdx(darcyIdx);
          const auto& elemVolVars = *(data.elementVolVars);
          const auto& darcyFvGeometry = data.fvGeometry;
          const auto& localBasis = darcyFvGeometry.feLocalBasis();


          // Darcy Permeability
          const auto& K = data.volVars.permeability();

          // INTEGRATION, second order as box provides linear functions
          const auto& rule = Dune::QuadratureRules<Scalar, darcyDim-1>::rule(data.segmentGeometry.type(), 2);
          //Loop over all quadrature points in the rule
          for (const auto& qp : rule)
          {
            const auto& ipLocal = qp.position();
            const auto& ipGlobal = data.segmentGeometry.global(ipLocal);
            const auto& ipElementLocal = data.element.geometry().local(ipGlobal);

            //reset pressure gradient and rho at this qp
            gradP=0.0;
            rho=0.0;
            //TODO: Is this needed?
            shapeValues.clear();
            shapeDerivatives.clear();

            //calculate the shape and derivative values at the qp
            localBasis.evaluateFunction(ipElementLocal, shapeValues);
            localBasis.evaluateJacobian(ipElementLocal, shapeDerivatives);

            //calc pressure gradient and rho at qp, every scv belongs to one node
            for (const auto& scv : scvs(data.fvGeometry)){
              //gradP += p_i* (J^-T * L'_i)
              data.element.geometry().jacobianInverseTransposed(ipElementLocal).usmv(elemVolVars[scv].pressure(darcyPhaseIdx), shapeDerivatives[scv.indexInElement()][0], gradP);
              if (enableGravity){
                rho += elemVolVars[scv].density(darcyPhaseIdx)*shapeValues[scv.indexInElement()][0];
              }
            }
            //account for gravity
            if (enableGravity){
              gradP.axpy(-rho, this->couplingManager().problem(darcyIdx).spatialParams().gravity(ipGlobal));
            }
            //Add the integrated segment velocity to the sum: v+= -weight_k * sqrt(det(A^T*A))*K/mu*gradP
            K.usmv(-qp.weight()*data.segmentGeometry.integrationElement(ipLocal)/data.volVars.viscosity(darcyPhaseIdx), gradP, velocity);
          }
          intersectionLength += data.segmentGeometry.volume();
        }
      }
      velocity /= intersectionLength; //averaging
      return velocity;
    }


    /*!
    * \brief Returns the averaged velocity vector at the interface of the porous medium according to darcys law with a different permeability tensor
    *
    * For the new tangential interface condition by Elissa Eggenweiler, a porous medium velocity with altered permeability tensor needs to be evaluated.
    * We use darcys law and perform an integral average over all coupling segments.
    *
    */
    VelocityVector newPorousMediumInterfaceVelocity(const Element<stokesIdx>& element, const SubControlVolumeFace<stokesIdx>& scvf) const
    {
      static constexpr int darcyDim = GridGeometry<darcyIdx>::GridView::dimension;
      using JacobianType = Dune::FieldMatrix<Scalar, 1, darcyDim>;
      std::vector<JacobianType> shapeDerivatives;
      std::vector<Dune::FieldVector<Scalar, 1>> shapeValues;

      VelocityVector velocity(0.0); //  velocity darcy
      VelocityVector gradP(0.0);    // pressure gradient darcy
      Scalar rho(0.0);              // density darcy
      Scalar intersectionLength = 0.0; //(total)intersection length could differ from scfv length

      //Getting needed information from the darcy domain
      const auto& stokesContext = this->couplingManager().stokesCouplingContextVector(element, scvf);
      static const bool enableGravity = getParamFromGroup<bool>(this->couplingManager().problem(darcyIdx).paramGroup(), "Problem.EnableGravity");

      // Iteration over the different coupling segments
      for (const auto& data : stokesContext)
      {
        //We are on (one of) the correct scvf(s)
        if (scvf.index() == data.stokesScvfIdx)
        {
          const auto darcyPhaseIdx = couplingPhaseIdx(darcyIdx);
          const auto& elemVolVars = *(data.elementVolVars);
          const auto& darcyFvGeometry = data.fvGeometry;
          const auto& localBasis = darcyFvGeometry.feLocalBasis();


          // Darcy Permeability
          const auto& K = data.volVars.permeability();

          // INTEGRATION, second order as box provides linear functions
          const auto& rule = Dune::QuadratureRules<Scalar, darcyDim-1>::rule(data.segmentGeometry.type(), 2);
          //Loop over all quadrature points in the rule
          for (const auto& qp : rule)
          {
            const auto& ipLocal = qp.position();
            const auto& ipGlobal = data.segmentGeometry.global(ipLocal);
            const auto& ipElementLocal = data.element.geometry().local(ipGlobal);

            //reset pressure gradient and rho at this qp
            gradP=0.0;
            rho=0.0;
            //TODO: Is this needed?
            shapeValues.clear();
            shapeDerivatives.clear();

            // darcy spatial dependent parameters
            const auto& epsInterface = this->couplingManager().problem(darcyIdx).spatialParams().epsInterfaceAtPos(ipGlobal);
            const auto& M = this->couplingManager().problem(darcyIdx).spatialParams().matrixNTangentialAtPos(ipGlobal);

            //calculate the shape and derivative values at the qp
            localBasis.evaluateFunction(ipElementLocal, shapeValues);
            localBasis.evaluateJacobian(ipElementLocal, shapeDerivatives);

            //calc pressure gradient and rho at qp, every scv belongs to one node
            for (const auto& scv : scvs(data.fvGeometry)){
              //gradP += p_i* (J^-T * L'_i)
              data.element.geometry().jacobianInverseTransposed(ipElementLocal).usmv(elemVolVars[scv].pressure(darcyPhaseIdx), shapeDerivatives[scv.indexInElement()][0], gradP);
              if (enableGravity){
                rho += elemVolVars[scv].density(darcyPhaseIdx)*shapeValues[scv.indexInElement()][0];
              }
            }
            //account for gravity
            if (enableGravity){
              gradP.axpy(-rho, this->couplingManager().problem(darcyIdx).spatialParams().gravity(ipGlobal));
            }
            //Add the integrated segment velocity to the sum: v+= -w_k * sqrt(det(A^T*A))*eps**2*M/mu*gradP
            M.usmv(qp.weight()*data.segmentGeometry.integrationElement(ipLocal)/data.volVars.viscosity(darcyPhaseIdx)*epsInterface*epsInterface, gradP, velocity);          }
          intersectionLength += data.segmentGeometry.volume();
        }
      }
      velocity /= intersectionLength; //averaging
      return velocity;
    }
};

/*!
 * \ingroup StokesDarcyCoupling
 * \brief Coupling data specialization for non-compositional models.
 */
template<class MDTraits, class CouplingManager, bool enableEnergyBalance>
class StokesDarcyCouplingDataImplementation<MDTraits, CouplingManager, enableEnergyBalance, false, DiscretizationMethod::box>
: public StokesDarcyCouplingDataBoxBase<MDTraits, CouplingManager, enableEnergyBalance>
{
    using ParentType = StokesDarcyCouplingDataBoxBase<MDTraits, CouplingManager, enableEnergyBalance>;
    using Scalar = typename MDTraits::Scalar;
    static constexpr auto stokesIdx = typename MDTraits::template SubDomain<0>::Index();
    static constexpr auto darcyIdx = typename MDTraits::template SubDomain<2>::Index();
    static constexpr auto stokesCellCenterIdx = stokesIdx;
    static constexpr auto stokesFaceIdx = typename MDTraits::template SubDomain<1>::Index();

    // the sub domain type tags
    template<std::size_t id>
    using SubDomainTypeTag = typename MDTraits::template SubDomain<id>::TypeTag;

    template<std::size_t id> using GridGeometry = GetPropType<SubDomainTypeTag<id>, Properties::GridGeometry>;
    template<std::size_t id> using Element = typename GridGeometry<id>::GridView::template Codim<0>::Entity;
    template<std::size_t id> using FVElementGeometry = typename GridGeometry<id>::LocalView;
    template<std::size_t id> using SubControlVolumeFace = typename GridGeometry<id>::LocalView::SubControlVolumeFace;
    template<std::size_t id> using SubControlVolume = typename GridGeometry<id>::LocalView::SubControlVolume;
    template<std::size_t id> using Indices = typename GetPropType<SubDomainTypeTag<id>, Properties::ModelTraits>::Indices;
    template<std::size_t id> using ElementFluxVariablesCache = typename GetPropType<SubDomainTypeTag<id>, Properties::GridFluxVariablesCache>::LocalView;
    template<std::size_t id> using ElementVolumeVariables = typename GetPropType<SubDomainTypeTag<id>, Properties::GridVolumeVariables>::LocalView;
    template<std::size_t id> using ElementFaceVariables = typename GetPropType<SubDomainTypeTag<id>, Properties::GridFaceVariables>::LocalView;
    template<std::size_t id> using VolumeVariables  = typename GetPropType<SubDomainTypeTag<id>, Properties::GridVolumeVariables>::VolumeVariables;

    static_assert(GetPropType<SubDomainTypeTag<darcyIdx>, Properties::ModelTraits>::numFluidComponents() == GetPropType<SubDomainTypeTag<darcyIdx>, Properties::ModelTraits>::numFluidPhases(),
                  "Darcy Model must not be compositional");

    using DiffusionCoefficientAveragingType = typename StokesDarcyCouplingOptions::DiffusionCoefficientAveragingType;

public:
    using ParentType::ParentType;
    using ParentType::couplingPhaseIdx;

    /*!
     * \brief Returns the mass flux across the coupling boundary as seen from the Darcy domain.
     */
    Scalar massCouplingCondition(const Element<darcyIdx>& element,
                                 const FVElementGeometry<darcyIdx>& fvGeometry,
                                 const ElementVolumeVariables<darcyIdx>& darcyElemVolVars,
                                 const ElementFluxVariablesCache<darcyIdx>& elementFluxVarsCache,
                                 const SubControlVolumeFace<darcyIdx>& scvf) const
    {
        const auto darcyPhaseIdx = couplingPhaseIdx(darcyIdx);
        const auto& darcyContext = this->couplingManager().darcyCouplingContextVector(element, scvf);

        Scalar flux = 0.0;
        for(const auto& data : darcyContext)
        {
            if(scvf.index() == data.darcyScvfIdx)
            {
                const Scalar velocity = data.velocity * scvf.unitOuterNormal();
                const auto& fluxVarCache = elementFluxVarsCache[scvf];
                const auto& shapeValues = fluxVarCache.shapeValues();

                Scalar darcyDensity = 0.0;
                for (auto&& scv : scvs(fvGeometry))
                    darcyDensity += darcyElemVolVars[scv].density(darcyPhaseIdx)*shapeValues[scv.indexInElement()][0];

                const Scalar stokesDensity = data.volVars.density();
                const bool insideIsUpstream = velocity > 0.0;
                flux += massFlux_(velocity, darcyDensity, stokesDensity, insideIsUpstream)*data.segmentGeometry.volume()/scvf.area();
            }
        }

        return flux;
    }

    /*!
     * \brief Returns the mass flux across the coupling boundary as seen from the free-flow domain.
     */
    Scalar massCouplingCondition(const Element<stokesIdx>& element,
                                 const FVElementGeometry<stokesIdx>& fvGeometry,
                                 const ElementVolumeVariables<stokesIdx>& stokesElemVolVars,
                                 const ElementFaceVariables<stokesIdx>& stokesElemFaceVars,
                                 const SubControlVolumeFace<stokesIdx>& scvf) const
    {
        const auto& stokesContext = this->couplingManager().stokesCouplingContextVector(element, scvf);

        Scalar flux = 0.0;
        for (const auto& data : stokesContext)
        {
            if (scvf.index() == data.stokesScvfIdx)
            {
                const Scalar velocity = stokesElemFaceVars[scvf].velocitySelf() * scvf.directionSign();
                const Scalar stokesDensity = stokesElemVolVars[scvf.insideScvIdx()].density();

                const auto darcyPhaseIdx = couplingPhaseIdx(darcyIdx);
                const auto& elemVolVars = *(data.elementVolVars);
                const auto& elemFluxVarsCache = *(data.elementFluxVarsCache);
                const auto& darcyScvf = data.fvGeometry.scvf(data.darcyScvfIdx);
                const auto& fluxVarCache = elemFluxVarsCache[darcyScvf];
                const auto& shapeValues = fluxVarCache.shapeValues();

                Scalar darcyDensity = 0.0;
                for (auto&& scv : scvs(data.fvGeometry))
                    darcyDensity += elemVolVars[scv].density(darcyPhaseIdx)*shapeValues[scv.indexInElement()][0];

                const bool insideIsUpstream = velocity > 0.0;
                flux += massFlux_(velocity, stokesDensity, darcyDensity, insideIsUpstream)*data.segmentGeometry.volume()/scvf.area();
            }
        }

        return flux;
    }

    /*!
     * \brief Returns the energy flux across the coupling boundary as seen from the Darcy domain.
     */
    template<bool isNI = enableEnergyBalance, typename std::enable_if_t<isNI, int> = 0>
    Scalar energyCouplingCondition(const Element<darcyIdx>& element,
                                   const FVElementGeometry<darcyIdx>& fvGeometry,
                                   const ElementVolumeVariables<darcyIdx>& darcyElemVolVars,
                                   const ElementFluxVariablesCache<darcyIdx>& elementFluxVarsCache,
                                   const SubControlVolumeFace<darcyIdx>& scvf,
                                   const DiffusionCoefficientAveragingType diffCoeffAvgType = DiffusionCoefficientAveragingType::ffOnly) const
    {
        const auto& darcyContext = this->couplingManager().darcyCouplingContextVector(element, scvf);

        Scalar flux = 0.0;
        for(const auto& data : darcyContext)
        {
            if(scvf.index() == data.darcyScvfIdx)
            {
                const auto& stokesVolVars = data.volVars;
                const auto& stokesScvf = data.fvGeometry.scvf(data.stokesScvfIdx);
                // always calculate the flux from stokes to darcy
                const Scalar velocity = -stokesScvf.area()/scvf.area() *(data.velocity * scvf.unitOuterNormal());
                const bool insideIsUpstream = velocity > 0.0;

                // the darcy flux is then multiplied by -1 (darcy flux = -stokes flux)
                flux += -1*energyFlux_(data.fvGeometry,
                                       fvGeometry,
                                       stokesVolVars,
                                       scvf,
                                       darcyElemVolVars,
                                       elementFluxVarsCache,
                                       velocity,
                                       insideIsUpstream,
                                       diffCoeffAvgType);
            }
        }
        return flux;
    }

    /*!
     * \brief Returns the energy flux across the coupling boundary as seen from the free-flow domain.
     */
    template<bool isNI = enableEnergyBalance, typename std::enable_if_t<isNI, int> = 0>
    Scalar energyCouplingCondition(const Element<stokesIdx>& element,
                                   const FVElementGeometry<stokesIdx>& fvGeometry,
                                   const ElementVolumeVariables<stokesIdx>& stokesElemVolVars,
                                   const ElementFaceVariables<stokesIdx>& stokesElemFaceVars,
                                   const SubControlVolumeFace<stokesIdx>& scvf,
                                   const DiffusionCoefficientAveragingType diffCoeffAvgType = DiffusionCoefficientAveragingType::ffOnly) const
    {
        const auto& stokesContext = this->couplingManager().stokesCouplingContext(element, scvf);
        const auto& stokesVolVars = stokesElemVolVars[scvf.insideScvIdx()];

        const auto& elemVolVars = *(stokesContext.elementVolVars);
        const auto& elemFluxVarsCache = *(stokesContext.elementFluxVarsCache);

        const auto& darcyScvf = stokesContext.fvGeometry.scvf(stokesContext.darcyScvfIdx);

        const Scalar velocity = stokesElemFaceVars[scvf].velocitySelf() * scvf.directionSign();
        const bool insideIsUpstream = velocity > 0.0;

        return energyFlux_(fvGeometry,
                           stokesContext.fvGeometry,
                           stokesVolVars,
                           darcyScvf,
                           elemVolVars,
                           elemFluxVarsCache,
                           velocity,
                           insideIsUpstream,
                           diffCoeffAvgType);
    }

private:

    /*!
     * \brief Evaluate the mole/mass flux across the interface.
     */
    Scalar massFlux_(const Scalar velocity,
                     const Scalar insideDensity,
                     const Scalar outSideDensity,
                     bool insideIsUpstream) const
    {
        return this->advectiveFlux(insideDensity, outSideDensity, velocity, insideIsUpstream);
    }

    /*!
     * \brief Evaluate the energy flux across the interface.
     */
    template<bool isNI = enableEnergyBalance, typename std::enable_if_t<isNI, int> = 0>
    Scalar energyFlux_(const FVElementGeometry<stokesIdx>& stokesFvGeometry,
                       const FVElementGeometry<darcyIdx>& darcyFvGeometry,
                       const VolumeVariables<stokesIdx>& stokesVolVars,
                       const SubControlVolumeFace<darcyIdx>& scvf,
                       const ElementVolumeVariables<darcyIdx>& darcyElemVolVars,
                       const ElementFluxVariablesCache<darcyIdx>& elementFluxVarsCache,
                       const Scalar velocity,
                       const bool insideIsUpstream,
                       const DiffusionCoefficientAveragingType diffCoeffAvgType) const
    {
        Scalar flux(0.0);

        const auto& stokesScv = (*scvs(stokesFvGeometry).begin());

        const auto& fluxVarCache = elementFluxVarsCache[scvf];
        const auto& shapeValues = fluxVarCache.shapeValues();

        Scalar temperature = 0.0;
        for (auto&& scv : scvs(darcyFvGeometry))
        {
            const auto& volVars = darcyElemVolVars[scv];
            temperature += volVars.temperature()*shapeValues[scv.indexInElement()][0];
        }

        const auto& darcyVolVars = darcyElemVolVars[scvf.insideScvIdx()];

        const Scalar stokesTerm = stokesVolVars.density(couplingPhaseIdx(stokesIdx)) * stokesVolVars.enthalpy(couplingPhaseIdx(stokesIdx));
        // ToDO interpolate using box basis functions
        const Scalar darcyTerm = darcyVolVars.density(couplingPhaseIdx(darcyIdx)) * darcyVolVars.enthalpy(couplingPhaseIdx(darcyIdx));

        flux += this->advectiveFlux(stokesTerm, darcyTerm, velocity, insideIsUpstream);

        const Scalar deltaT = temperature - stokesVolVars.temperature();
        const Scalar dist = (scvf.ipGlobal() - stokesScv.center()).two_norm();
        if(diffCoeffAvgType == DiffusionCoefficientAveragingType::ffOnly)
        {
            flux += -1*this->thermalConductivity_(stokesVolVars, stokesFvGeometry, stokesScv) * deltaT / dist ;
        }
        else
            DUNE_THROW(Dune::NotImplemented, "Multidomain staggered box coupling only works for DiffusionCoefficientAveragingType = ffOnly");

        return flux;
    }
};

/*!
 * \ingroup StokesDarcyCoupling
 * \brief Coupling data specialization for compositional models.
 */
template<class MDTraits, class CouplingManager, bool enableEnergyBalance>
class StokesDarcyCouplingDataImplementation<MDTraits, CouplingManager, enableEnergyBalance, true, DiscretizationMethod::box>
: public StokesDarcyCouplingDataBoxBase<MDTraits, CouplingManager, enableEnergyBalance>
{
    using ParentType = StokesDarcyCouplingDataBoxBase<MDTraits, CouplingManager, enableEnergyBalance>;
    using Scalar = typename MDTraits::Scalar;
    static constexpr auto stokesIdx = typename MDTraits::template SubDomain<0>::Index();
    static constexpr auto darcyIdx = typename MDTraits::template SubDomain<2>::Index();
    static constexpr auto stokesCellCenterIdx = stokesIdx;
    static constexpr auto stokesFaceIdx = typename MDTraits::template SubDomain<1>::Index();

    // the sub domain type tags
    template<std::size_t id>
    using SubDomainTypeTag = typename MDTraits::template SubDomain<id>::TypeTag;

    template<std::size_t id> using GridGeometry = GetPropType<SubDomainTypeTag<id>, Properties::GridGeometry>;
    template<std::size_t id> using Element = typename GridGeometry<id>::GridView::template Codim<0>::Entity;
    template<std::size_t id> using FVElementGeometry = typename GridGeometry<id>::LocalView;
    template<std::size_t id> using SubControlVolumeFace = typename FVElementGeometry<id>::SubControlVolumeFace;
    template<std::size_t id> using SubControlVolume = typename GridGeometry<id>::LocalView::SubControlVolume;
    template<std::size_t id> using Indices = typename GetPropType<SubDomainTypeTag<id>, Properties::ModelTraits>::Indices;
    template<std::size_t id> using ElementFluxVariablesCache = typename GetPropType<SubDomainTypeTag<id>, Properties::GridFluxVariablesCache>::LocalView;
    template<std::size_t id> using ElementVolumeVariables = typename GetPropType<SubDomainTypeTag<id>, Properties::GridVolumeVariables>::LocalView;
    template<std::size_t id> using ElementFaceVariables = typename GetPropType<SubDomainTypeTag<id>, Properties::GridFaceVariables>::LocalView;
    template<std::size_t id> using VolumeVariables  = typename GetPropType<SubDomainTypeTag<id>, Properties::GridVolumeVariables>::VolumeVariables;
    template<std::size_t id> using FluidSystem  = GetPropType<SubDomainTypeTag<id>, Properties::FluidSystem>;

    static constexpr auto numComponents = GetPropType<SubDomainTypeTag<stokesIdx>, Properties::ModelTraits>::numFluidComponents();
    static constexpr auto replaceCompEqIdx = GetPropType<SubDomainTypeTag<stokesIdx>, Properties::ModelTraits>::replaceCompEqIdx();
    static constexpr bool useMoles = GetPropType<SubDomainTypeTag<stokesIdx>, Properties::ModelTraits>::useMoles();
    static constexpr auto referenceSystemFormulation = GetPropType<SubDomainTypeTag<stokesIdx>, Properties::MolecularDiffusionType>::referenceSystemFormulation();

    static_assert(GetPropType<SubDomainTypeTag<darcyIdx>, Properties::ModelTraits>::numFluidComponents() == numComponents, "Both submodels must use the same number of components");
    static_assert(getPropValue<SubDomainTypeTag<darcyIdx>, Properties::UseMoles>() == useMoles, "Both submodels must either use moles or not");
    static_assert(getPropValue<SubDomainTypeTag<darcyIdx>, Properties::ReplaceCompEqIdx>() == replaceCompEqIdx, "Both submodels must use the same replaceCompEqIdx");
    static_assert(GetPropType<SubDomainTypeTag<darcyIdx>, Properties::MolecularDiffusionType>::referenceSystemFormulation() == referenceSystemFormulation,
                  "Both submodels must use the same reference system formulation for diffusion");

    using NumEqVector = Dune::FieldVector<Scalar, numComponents>;

    using DiffusionCoefficientAveragingType = typename StokesDarcyCouplingOptions::DiffusionCoefficientAveragingType;

    static constexpr bool isFicksLaw = IsFicksLaw<GetPropType<SubDomainTypeTag<stokesIdx>, Properties::MolecularDiffusionType>>();
    static_assert(isFicksLaw == IsFicksLaw<GetPropType<SubDomainTypeTag<darcyIdx>, Properties::MolecularDiffusionType>>(),
                  "Both submodels must use the same diffusion law.");

    static_assert(isFicksLaw, "Box-Staggered Coupling only implemented for Fick's law!");

    using ReducedComponentVector = Dune::FieldVector<Scalar, numComponents-1>;
    using ReducedComponentMatrix = Dune::FieldMatrix<Scalar, numComponents-1, numComponents-1>;

    using MolecularDiffusionType = GetPropType<SubDomainTypeTag<stokesIdx>, Properties::MolecularDiffusionType>;

public:
    using ParentType::ParentType;
    using ParentType::couplingPhaseIdx;
    using ParentType::couplingCompIdx;

    /*!
     * \brief Returns the mass flux across the coupling boundary as seen from the Darcy domain.
     */
    NumEqVector massCouplingCondition(const Element<darcyIdx>& element,
                                      const FVElementGeometry<darcyIdx>& fvGeometry,
                                      const ElementVolumeVariables<darcyIdx>& darcyElemVolVars,
                                      const ElementFluxVariablesCache<darcyIdx>& elementFluxVarsCache,
                                      const SubControlVolumeFace<darcyIdx>& scvf,
                                      const DiffusionCoefficientAveragingType diffCoeffAvgType = DiffusionCoefficientAveragingType::ffOnly) const
    {
        const auto& darcyContext = this->couplingManager().darcyCouplingContext(element, scvf);
        const auto& stokesVolVars = darcyContext.volVars;

        const Scalar velocity = -1*(darcyContext.velocity * scvf.unitOuterNormal());
        const bool insideIsUpstream = velocity > 0.0;

        return -1*massFlux_(darcyIdx,
                            stokesIdx,
                            darcyContext.fvGeometry,
                            fvGeometry,
                            stokesVolVars,
                            scvf,
                            darcyElemVolVars,
                            elementFluxVarsCache,
                            velocity,
                            insideIsUpstream,
                            diffCoeffAvgType);
    }

    /*!
     * \brief Returns the mass flux across the coupling boundary as seen from the free-flow domain.
     */
    NumEqVector massCouplingCondition(const Element<stokesIdx>& element,
                                      const FVElementGeometry<stokesIdx>& fvGeometry,
                                      const ElementVolumeVariables<stokesIdx>& stokesElemVolVars,
                                      const ElementFaceVariables<stokesIdx>& stokesElemFaceVars,
                                      const SubControlVolumeFace<stokesIdx>& scvf,
                                      const DiffusionCoefficientAveragingType diffCoeffAvgType = DiffusionCoefficientAveragingType::ffOnly) const
    {
        const auto& stokesContext = this->couplingManager().stokesCouplingContext(element, scvf);
        const auto& stokesVolVars = stokesElemVolVars[scvf.insideScvIdx()];

        const auto& elemVolVars = *(stokesContext.elementVolVars);
        const auto& elemFluxVarsCache = *(stokesContext.elementFluxVarsCache);

        const auto& darcyScvf = stokesContext.fvGeometry.scvf(stokesContext.darcyScvfIdx);

        const Scalar velocity = stokesElemFaceVars[scvf].velocitySelf() * scvf.directionSign();
        const bool insideIsUpstream = velocity > 0.0;

        return massFlux_(stokesIdx,
                         darcyIdx,
                         fvGeometry,
                         stokesContext.fvGeometry,
                         stokesVolVars,
                         darcyScvf,
                         elemVolVars,
                         elemFluxVarsCache,
                         velocity,
                         insideIsUpstream,
                         diffCoeffAvgType);
    }

    /*!
     * \brief Returns the energy flux across the coupling boundary as seen from the Darcy domain.
     */
    template<bool isNI = enableEnergyBalance, typename std::enable_if_t<isNI, int> = 0>
    Scalar energyCouplingCondition(const Element<darcyIdx>& element,
                                   const FVElementGeometry<darcyIdx>& fvGeometry,
                                   const ElementVolumeVariables<darcyIdx>& darcyElemVolVars,
                                   const ElementFluxVariablesCache<darcyIdx>& elementFluxVarsCache,
                                   const SubControlVolumeFace<darcyIdx>& scvf,
                                   const DiffusionCoefficientAveragingType diffCoeffAvgType = DiffusionCoefficientAveragingType::ffOnly) const
    {
        const auto& darcyContext = this->couplingManager().darcyCouplingContext(element, scvf);
        const auto& stokesVolVars = darcyContext.volVars;

        const Scalar velocity = -1*(darcyContext.velocity * scvf.unitOuterNormal());
        const bool insideIsUpstream = velocity > 0.0;

        return -1*energyFlux_(darcyIdx,
                              stokesIdx,
                              darcyContext.fvGeometry,
                              fvGeometry,
                              stokesVolVars,
                              scvf,
                              darcyElemVolVars,
                              elementFluxVarsCache,
                              velocity,
                              insideIsUpstream,
                              diffCoeffAvgType);
    }

    /*!
     * \brief Returns the energy flux across the coupling boundary as seen from the free-flow domain.
     */
    template<bool isNI = enableEnergyBalance, typename std::enable_if_t<isNI, int> = 0>
    Scalar energyCouplingCondition(const Element<stokesIdx>& element,
                                   const FVElementGeometry<stokesIdx>& fvGeometry,
                                   const ElementVolumeVariables<stokesIdx>& stokesElemVolVars,
                                   const ElementFaceVariables<stokesIdx>& stokesElemFaceVars,
                                   const SubControlVolumeFace<stokesIdx>& scvf,
                                   const DiffusionCoefficientAveragingType diffCoeffAvgType = DiffusionCoefficientAveragingType::ffOnly) const
    {
        const auto& stokesContext = this->couplingManager().stokesCouplingContext(element, scvf);
        const auto& stokesVolVars = stokesElemVolVars[scvf.insideScvIdx()];

        const auto& elemVolVars = *(stokesContext.elementVolVars);
        const auto& elemFluxVarsCache = *(stokesContext.elementFluxVarsCache);

        const auto& darcyScvf = stokesContext.fvGeometry.scvf(stokesContext.darcyScvfIdx);

        const Scalar velocity = stokesElemFaceVars[scvf].velocitySelf() * scvf.directionSign();
        const bool insideIsUpstream = velocity > 0.0;

        return energyFlux_(stokesIdx,
                           darcyIdx,
                           fvGeometry,
                           stokesContext.fvGeometry,
                           stokesVolVars,
                           darcyScvf,
                           elemVolVars,
                           elemFluxVarsCache,
                           velocity,
                           insideIsUpstream,
                           diffCoeffAvgType);
    }

protected:

    /*!
     * \brief Evaluate the compositional mole/mass flux across the interface.
     */
    template<std::size_t i, std::size_t j>
    NumEqVector massFlux_(Dune::index_constant<i> domainI,
                          Dune::index_constant<j> domainJ,
                          const FVElementGeometry<stokesIdx>& stokesFvGeometry,
                          const FVElementGeometry<darcyIdx>& darcyFvGeometry,
                          const VolumeVariables<stokesIdx>& stokesVolVars,
                          const SubControlVolumeFace<darcyIdx>& scvf,
                          const ElementVolumeVariables<darcyIdx>& darcyElemVolVars,
                          const ElementFluxVariablesCache<darcyIdx>& elementFluxVarsCache,
                          const Scalar velocity,
                          const bool insideIsUpstream,
                          const DiffusionCoefficientAveragingType diffCoeffAvgType) const
    {
        NumEqVector flux(0.0);
        NumEqVector diffusiveFlux(0.0);

        const auto& darcyVolVars = darcyElemVolVars[scvf.insideScvIdx()];

        auto moleOrMassFraction = [](const auto& volVars, int phaseIdx, int compIdx)
        { return useMoles ? volVars.moleFraction(phaseIdx, compIdx) : volVars.massFraction(phaseIdx, compIdx); };

        auto moleOrMassDensity = [](const auto& volVars, int phaseIdx)
        { return useMoles ? volVars.molarDensity(phaseIdx) : volVars.density(phaseIdx); };

        // treat the advective fluxes
        auto insideTerm = [&](int compIdx)
        { return moleOrMassFraction(stokesVolVars, couplingPhaseIdx(stokesIdx), compIdx) * moleOrMassDensity(stokesVolVars, couplingPhaseIdx(stokesIdx)); };

        // ToDO interpolate using box basis functions
        auto outsideTerm = [&](int compIdx)
        { return moleOrMassFraction(darcyVolVars, couplingPhaseIdx(darcyIdx), compIdx) * moleOrMassDensity(darcyVolVars, couplingPhaseIdx(darcyIdx)); };

        for (int compIdx = 0; compIdx < numComponents; ++compIdx)
        {
            const int domainICompIdx = couplingCompIdx(stokesIdx, compIdx);
            const int domainJCompIdx = couplingCompIdx(darcyIdx, compIdx);
            flux[couplingCompIdx(domainI, compIdx)] += this->advectiveFlux(insideTerm(domainICompIdx), outsideTerm(domainJCompIdx), velocity, insideIsUpstream);
        }

        // treat the diffusive fluxes
        diffusiveFlux += diffusiveMolecularFluxFicksLaw_(domainI,
                                                         domainJ,
                                                         stokesFvGeometry,
                                                         darcyFvGeometry,
                                                         stokesVolVars,
                                                         scvf,
                                                         darcyElemVolVars,
                                                         elementFluxVarsCache,
                                                         velocity,
                                                         diffCoeffAvgType);

        //convert to correct units if necessary
        if (referenceSystemFormulation == ReferenceSystemFormulation::massAveraged && useMoles)
        {
            for (int compIdx = 0; compIdx < numComponents; ++compIdx)
            {
                const int domainICompIdx = couplingCompIdx(domainI, compIdx);
                diffusiveFlux[domainICompIdx] *= 1/FluidSystem<domainI>::molarMass(domainICompIdx);
            }
        }
        if (referenceSystemFormulation == ReferenceSystemFormulation::molarAveraged && !useMoles)
        {
            for (int compIdx = 0; compIdx < numComponents; ++compIdx)
            {
                const int domainICompIdx = couplingCompIdx(domainI, compIdx);
                diffusiveFlux[domainICompIdx] *= FluidSystem<domainI>::molarMass(domainICompIdx);
            }
        }

        flux += diffusiveFlux;
        // convert to total mass/mole balance, if set be user
        if (replaceCompEqIdx < numComponents)
            flux[replaceCompEqIdx] = std::accumulate(flux.begin(), flux.end(), 0.0);

        return flux;
    }

    /*!
     * \brief Returns the molecular diffusion coefficient within the free flow domain.
     */
    Scalar diffusionCoefficient_(const VolumeVariables<stokesIdx>& volVars, int phaseIdx, int compIdx) const
    {
         return volVars.effectiveDiffusivity(phaseIdx, compIdx);
    }

    /*!
     * \brief Returns the effective diffusion coefficient within the porous medium.
     */
    Scalar diffusionCoefficient_(const VolumeVariables<darcyIdx>& volVars, int phaseIdx, int compIdx) const
    {
        using EffDiffModel = GetPropType<SubDomainTypeTag<darcyIdx>, Properties::EffectiveDiffusivityModel>;
        return EffDiffModel::effectiveDiffusivity(volVars.porosity(),
                                                  volVars.saturation(phaseIdx),
                                                  volVars.diffusionCoefficient(phaseIdx, compIdx));
    }

    Scalar getComponentEnthalpy(const VolumeVariables<stokesIdx>& volVars, int phaseIdx, int compIdx) const
    {
        return FluidSystem<stokesIdx>::componentEnthalpy(volVars.fluidState(), 0, compIdx);
    }

    Scalar getComponentEnthalpy(const VolumeVariables<darcyIdx>& volVars, int phaseIdx, int compIdx) const
    {
        return FluidSystem<darcyIdx>::componentEnthalpy(volVars.fluidState(), phaseIdx, compIdx);
    }

    template<std::size_t i, std::size_t j>
    NumEqVector diffusiveMolecularFluxFicksLaw_(Dune::index_constant<i> domainI,
                                                Dune::index_constant<j> domainJ,
                                                const FVElementGeometry<stokesIdx>& stokesFvGeometry,
                                                const FVElementGeometry<darcyIdx>& darcyFvGeometry,
                                                const VolumeVariables<stokesIdx>& stokesVolVars,
                                                const SubControlVolumeFace<darcyIdx>& scvf,
                                                const ElementVolumeVariables<darcyIdx>& darcyElemVolVars,
                                                const ElementFluxVariablesCache<darcyIdx>& elementFluxVarsCache,
                                                const Scalar velocity,
                                                const DiffusionCoefficientAveragingType diffCoeffAvgType) const
    {
        NumEqVector diffusiveFlux(0.0);

        const auto& fluxVarCache = elementFluxVarsCache[scvf];
        const auto& shapeValues = fluxVarCache.shapeValues();

        const Scalar rhoStokes = massOrMolarDensity(stokesVolVars, referenceSystemFormulation, couplingPhaseIdx(stokesIdx));
        Scalar rhoDarcy = 0.0;
        for (auto&& scv : scvs(darcyFvGeometry))
        {
            const auto& volVars = darcyElemVolVars[scv];
            rhoDarcy += massOrMolarDensity(volVars, referenceSystemFormulation, couplingPhaseIdx(darcyIdx))
                                           *shapeValues[scv.indexInElement()][0];
        }
        const Scalar avgDensity = 0.5 * (rhoStokes + rhoDarcy);

        for (int compIdx = 1; compIdx < numComponents; ++compIdx)
        {
            const int stokesCompIdx = couplingCompIdx(stokesIdx, compIdx);
            const int darcyCompIdx = couplingCompIdx(darcyIdx, compIdx);

            assert(FluidSystem<stokesIdx>::componentName(stokesCompIdx) == FluidSystem<darcyIdx>::componentName(darcyCompIdx));

            const Scalar massOrMoleFractionStokes = massOrMoleFraction(stokesVolVars, referenceSystemFormulation, couplingPhaseIdx(stokesIdx), stokesCompIdx);

            Scalar massOrMoleFractionInterface = 0.0;
            for (auto&& scv : scvs(darcyFvGeometry))
            {
                const auto& volVars = darcyElemVolVars[scv];
                massOrMoleFractionInterface += massOrMoleFraction(volVars, referenceSystemFormulation, couplingPhaseIdx(darcyIdx), darcyCompIdx)
                                                * shapeValues[scv.indexInElement()][0];
            }

            const Scalar deltaMassOrMoleFrac = massOrMoleFractionInterface - massOrMoleFractionStokes;
            const auto& stokesScv = (*scvs(stokesFvGeometry).begin());
            const Scalar dist = (stokesScv.center() - scvf.ipGlobal()).two_norm();
            if(diffCoeffAvgType == DiffusionCoefficientAveragingType::ffOnly)
                diffusiveFlux[couplingCompIdx(domainI, compIdx)] += -avgDensity * diffusionCoefficient_(stokesVolVars, couplingPhaseIdx(stokesIdx), stokesCompIdx)
                                                                     * deltaMassOrMoleFrac / dist;
            else
                DUNE_THROW(Dune::NotImplemented, "Multidomain staggered box coupling only works for DiffusionCoefficientAveragingType = ffOnly");
        }

        const Scalar cumulativeFlux = std::accumulate(diffusiveFlux.begin(), diffusiveFlux.end(), 0.0);
        diffusiveFlux[couplingCompIdx(domainI, 0)] = -cumulativeFlux;

        return diffusiveFlux;
    }

    /*!
     * \brief Evaluate the energy flux across the interface.
     */
    template<std::size_t i, std::size_t j, bool isNI = enableEnergyBalance, typename std::enable_if_t<isNI, int> = 0>
    Scalar energyFlux_(Dune::index_constant<i> domainI,
                       Dune::index_constant<j> domainJ,
                       const FVElementGeometry<stokesIdx>& stokesFvGeometry,
                       const FVElementGeometry<darcyIdx>& darcyFvGeometry,
                       const VolumeVariables<stokesIdx>& stokesVolVars,
                       const SubControlVolumeFace<darcyIdx>& scvf,
                       const ElementVolumeVariables<darcyIdx>& darcyElemVolVars,
                       const ElementFluxVariablesCache<darcyIdx>& elementFluxVarsCache,
                       const Scalar velocity,
                       const bool insideIsUpstream,
                       const DiffusionCoefficientAveragingType diffCoeffAvgType) const
    {
        Scalar flux(0.0);

        const auto& stokesScv = (*scvs(stokesFvGeometry).begin());

        const auto& fluxVarCache = elementFluxVarsCache[scvf];
        const auto& shapeValues = fluxVarCache.shapeValues();

        Scalar temperature = 0.0;
        for (auto&& scv : scvs(darcyFvGeometry))
        {
            const auto& volVars = darcyElemVolVars[scv];
            temperature += volVars.temperature()*shapeValues[scv.indexInElement()][0];
        }

        const auto& darcyVolVars = darcyElemVolVars[scvf.insideScvIdx()];

        const Scalar stokesTerm = stokesVolVars.density(couplingPhaseIdx(stokesIdx)) * stokesVolVars.enthalpy(couplingPhaseIdx(stokesIdx));
        // ToDO interpolate using box basis functions
        const Scalar darcyTerm = darcyVolVars.density(couplingPhaseIdx(darcyIdx)) * darcyVolVars.enthalpy(couplingPhaseIdx(darcyIdx));

        flux += this->advectiveFlux(stokesTerm, darcyTerm, velocity, insideIsUpstream);

        const Scalar deltaT = temperature - stokesVolVars.temperature();
        const Scalar dist = (scvf.ipGlobal() - stokesScv.center()).two_norm();
        if(diffCoeffAvgType == DiffusionCoefficientAveragingType::ffOnly)
        {
            flux += -1*this->thermalConductivity_(stokesVolVars, stokesFvGeometry, stokesScv) * deltaT / dist ;
        }
        else
            DUNE_THROW(Dune::NotImplemented, "Multidomain staggered box coupling only works for DiffusionCoefficientAveragingType = ffOnly");

        auto diffusiveFlux = diffusiveMolecularFluxFicksLaw_(domainI,
                                                             domainJ,
                                                             stokesFvGeometry,
                                                             darcyFvGeometry,
                                                             stokesVolVars,
                                                             scvf,
                                                             darcyElemVolVars,
                                                             elementFluxVarsCache,
                                                             velocity,
                                                             diffCoeffAvgType);

        for (int compIdx = 0; compIdx < diffusiveFlux.size(); ++compIdx)
        {
            const int stokesCompIdx = couplingCompIdx(stokesIdx, compIdx);
            const int darcyCompIdx = couplingCompIdx(darcyIdx, compIdx);
            const int domainCompIdx = couplingCompIdx(domainI, compIdx);

            const Scalar componentEnthalpy = diffusiveFlux[domainCompIdx] > 0 ?
                                             getComponentEnthalpy(stokesVolVars, couplingPhaseIdx(stokesIdx), stokesCompIdx)
                                           : getComponentEnthalpy(darcyVolVars, couplingPhaseIdx(darcyIdx), darcyCompIdx);

            if (referenceSystemFormulation == ReferenceSystemFormulation::massAveraged)
                flux += diffusiveFlux[domainCompIdx] * componentEnthalpy;
            else
                flux += diffusiveFlux[domainCompIdx] * FluidSystem<domainI>::molarMass(domainCompIdx) * componentEnthalpy;
        }

        return flux;
    }
};

} // end namespace Dumux

#endif // DUMUX_STOKES_DARCY_COUPLINGDATA_HH