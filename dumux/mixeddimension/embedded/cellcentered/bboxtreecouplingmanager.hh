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
 * \ingroup EmbeddedCoupling
 * \ingroup CCModel
 * \brief Coupling manager for low-dimensional domains embedded in the bulk
 *        domain. Point sources on each integration point are computed by an AABB tree.
 *        Both domain are assumed to be discretized using a cc finite volume scheme.
 */

#ifndef DUMUX_CC_BBOXTREE_EMBEDDEDCOUPLINGMANAGER_HH
#define DUMUX_CC_BBOXTREE_EMBEDDEDCOUPLINGMANAGER_HH

#include <iostream>
#include <fstream>
#include <string>
#include <utility>

#include <dune/common/timer.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/geometry/referenceelements.hh>

#include <dumux/common/properties.hh>
#include <dumux/common/geometry/intersectingentities.hh>
#include <dumux/multidomain/couplingmanager.hh>
#include <dumux/mixeddimension/glue/glue.hh>
#include <dumux/mixeddimension/embedded/cellcentered/pointsourcedata.hh>

namespace Dumux {

/*!
 * \ingroup EmbeddedCoupling
 * \ingroup CCModel
 * \brief Manages the coupling between bulk elements and lower dimensional elements
 *        Point sources on each integration point are computed by an AABB tree.
 *        Both domain are assumed to be discretized using a cc finite volume scheme.
 */
template<class MDTraits>
class CCBBoxTreeEmbeddedCouplingManager
: public CouplingManager<MDTraits, CCBBoxTreeEmbeddedCouplingManager<MDTraits>>
{
    using Scalar = typename MDTraits::Scalar;
    static constexpr auto bulkIdx = typename MDTraits::template DomainIdx<0>();
    static constexpr auto lowDimIdx = typename MDTraits::template DomainIdx<1>();
    using SolutionVector = typename MDTraits::SolutionVector;
    using PointSourceData = Dumux::PointSourceDataCircleAverage<MDTraits>;
    using CouplingStencils = std::unordered_map<std::size_t, std::vector<std::size_t> >;
    using CouplingStencil = CouplingStencils::mapped_type;

    // the sub domain type tags
    template<std::size_t id>
    using SubDomainTypeTag = typename MDTraits::template SubDomainTypeTag<id>;

    template<std::size_t id> using GridView = typename GET_PROP_TYPE(SubDomainTypeTag<id>, GridView);
    template<std::size_t id> using Problem = typename GET_PROP_TYPE(SubDomainTypeTag<id>, Problem);
    template<std::size_t id> using PointSource = typename GET_PROP_TYPE(SubDomainTypeTag<id>, PointSource);
    template<std::size_t id> using PrimaryVariables = typename GET_PROP_TYPE(SubDomainTypeTag<id>, PrimaryVariables);
    template<std::size_t id> using NumEqVector = typename GET_PROP_TYPE(SubDomainTypeTag<id>, NumEqVector);
    template<std::size_t id> using ElementSolutionVector = typename GET_PROP_TYPE(SubDomainTypeTag<id>, ElementSolutionVector);
    template<std::size_t id> using VolumeVariables = typename GET_PROP_TYPE(SubDomainTypeTag<id>, VolumeVariables);
    template<std::size_t id> using ElementVolumeVariables = typename GET_PROP_TYPE(SubDomainTypeTag<id>, ElementVolumeVariables);
    template<std::size_t id> using FVGridGeometry = typename GET_PROP_TYPE(SubDomainTypeTag<id>, FVGridGeometry);
    template<std::size_t id> using FVElementGeometry = typename FVGridGeometry<id>::LocalView;
    template<std::size_t id> using ElementBoundaryTypes = typename GET_PROP_TYPE(SubDomainTypeTag<id>, ElementBoundaryTypes);
    template<std::size_t id> using ElementFluxVariablesCache = typename GET_PROP_TYPE(SubDomainTypeTag<id>, ElementFluxVariablesCache);
    template<std::size_t id> using Element = typename GridView<id>::template Codim<0>::Entity;

    enum {
        bulkDim = GridView<bulkIdx>::dimension,
        lowDimDim = GridView<lowDimIdx>::dimension,
        dimWorld = GridView<bulkIdx>::dimensionworld
    };

    using GlobalPosition = Dune::FieldVector<Scalar, dimWorld>;
    using GlueType = CCMixedDimensionGlue<GridView<bulkIdx>, GridView<lowDimIdx>>;

public:
    /*!
     * \brief Constructor
     */
    CCBBoxTreeEmbeddedCouplingManager(std::shared_ptr<const FVGridGeometry<bulkIdx>> bulkFvGridGeometry,
                                      std::shared_ptr<const FVGridGeometry<lowDimIdx>> lowDimFvGridGeometry)
    {

        // Check if we are using the cellcentered method in both domains
        static_assert(lowDimDim == 1, "The bounding box coupling manager only works with one-dimensional low-dim grids");
        glue_ = std::make_shared<GlueType>();
    }

    /*!
     * \brief Methods to be accessed by main
     */
    // \{

    void init(std::shared_ptr<Problem<bulkIdx>> bulkProblem,
              std::shared_ptr<Problem<lowDimIdx>> lowDimProblem,
              const SolutionVector& curSol)
    {
        curSol_ = curSol;
        problemTuple_ = std::make_tuple(bulkProblem, lowDimProblem);

        integrationOrder_ = getParam<int>("MixedDimension.IntegrationOrder", 1);
        computePointSourceData(integrationOrder_);
        computeLowDimVolumeFractions();
    }

    /*!
     * \brief Update the solution vector before assembly
     */
    void updateSolution(const SolutionVector& curSol)
    { curSol_ = curSol; }

    // \}

    /*!
     * \brief Methods to be accessed by the assembly
     */
    // \{

    /*!
     * \brief The coupling stencil of domain I, i.e. which domain J dofs
     *        the given domain I element's residual depends on.
     */
    template<std::size_t i, std::size_t j>
    const CouplingStencil& couplingStencil(const Element<i>& element,
                                           Dune::index_constant<i> domainI,
                                           Dune::index_constant<j> domainJ) const
    {
        static_assert(i != j, "A domain cannot be coupled to itself!");

        const auto eIdx = problem(domainI).fvGridGeometry().elementMapper().index(element);
        if (couplingStencils(domainI).count(eIdx))
            return couplingStencils(domainI).at(eIdx);
        else
            return emptyStencil_;
    }

    //! evaluate coupling residual for the derivative residual i with respect to privars of dof j
    //! we only need to evaluate the part of the residual that will be influenced by the the privars of dof j
    //! i.e. the source term.
    //! the coupling residual is symmetric so we only need to one template function here
    template<std::size_t i, std::size_t j>
    NumEqVector<i> evalCouplingResidual(Dune::index_constant<i> domainI,
                                        const Element<i>& elementI,
                                        const FVElementGeometry<i>& fvGeometry,
                                        const ElementVolumeVariables<i>& curElemVolVars,
                                        const ElementBoundaryTypes<i>& elemBcTypes,
                                        const ElementFluxVariablesCache<i>& elemFluxVarsCache,
                                        Dune::index_constant<j> domainJ,
                                        const Element<j>& elementJ)
    {
        static_assert(i != j, "A domain cannot be coupled to itself!");

        const auto eIdx = problem(domainI).fvGridGeometry().elementMapper().index(elementI);
        auto&& scv = fvGeometry.scv(eIdx);
        auto couplingSource = problem(domainI).scvPointSources(elementI, fvGeometry, curElemVolVars, scv);
        couplingSource *= -scv.volume()*curElemVolVars[scv].extrusionFactor();
        return couplingSource;
    }

    //! evaluate coupling residual for the derivative bulk DOF with respect to low dim DOF
    //! we only need to evaluate the part of the residual that will be influence by the low dim DOF
    template<std::size_t i, std::size_t j, class MatrixBlock>
    void addCouplingDerivatives(MatrixBlock& Aij,
                                Dune::index_constant<i> domainI,
                                const Element<i>& elementI,
                                const FVElementGeometry<i>& fvGeometry,
                                const ElementVolumeVariables<i>& curElemVolVars,
                                Dune::index_constant<j> domainJ,
                                const Element<j>& elementJ)
    {
        static_assert(i != j, "A domain cannot be coupled to itself!");

        const auto eIdx = problem(domainI).fvGridGeometry().elementMapper().index(elementI);

        const auto key = std::make_pair(eIdx, 0);
        if (problem(domainI).pointSourceMap().count(key))
        {
            // call the solDependent function. Herein the user might fill/add values to the point sources
            // we make a copy of the local point sources here
            const auto pointSources = problem(domainI).pointSourceMap().at(key);

            // add the point source values to the local residual (negative is sign convention for source term)
            for (const auto& source : pointSources)
                Aij[0][0] -= pointSourceDerivative(source, domainI, domainJ);
        }
    }

    //! helper function for the point source derivative (di/dj)
    template<class PointSource, std::size_t i, std::size_t j>
    Scalar pointSourceDerivative(const PointSource& source,
                                 Dune::index_constant<i> domainI,
                                 Dune::index_constant<j> domainJ) const
    {
        constexpr Scalar sign = (i == j) ? -1.0 : 1.0;
        // calculate the source derivative
        const Scalar radius = this->radius(source.id());
        const Scalar beta = 2*M_PI/(2*M_PI + std::log(radius));
        return sign*beta*source.quadratureWeight()*source.integrationElement()/source.embeddings();
    }

    /*!
     * \brief Bind the coupling context
     */
    template<class Element, std::size_t i, class Assembler>
    void bindCouplingContext(Dune::index_constant<i> domainI, const Element& element, const Assembler& assembler)
    {}

    /*!
     * \brief Update the coupling context for a derivative i->j
     */
    template<std::size_t i, std::size_t j, class Assembler>
    void updateCouplingContext(Dune::index_constant<i> domainI,
                               Dune::index_constant<j> domainJ,
                               const Element<j>& element,
                               const PrimaryVariables<j>& priVars,
                               const Assembler& assembler)
    {
        const auto eIdx = problem(domainJ).fvGridGeometry().elementMapper().index(element);
        curSol_[domainJ][eIdx] = priVars;
    }

    // \}

    /* \brief Compute integration point point sources and associated data
     *
     * This method uses grid glue to intersect the given grids. Over each intersection
     * we later need to integrate a source term. This method places point sources
     * at each quadrature point and provides the point source with the necessary
     * information to compute integrals (quadrature weight and integration element)
     * \param order The order of the quadrature rule for integration of sources over an intersection
     * \param verbose If the point source computation is verbose
     */
    void computePointSourceData(std::size_t order = 1,
                                bool verbose = false)
    {
        // Initialize the bulk bounding box tree
        const auto& bulkTree = problem(bulkIdx).fvGridGeometry().boundingBoxTree();

        // initilize the maps
        // do some logging and profiling
        Dune::Timer watch;
        std::cout << "Initializing the point sources..." << std::endl;

        // clear all internal members like pointsource vectors and stencils
        // initializes the point source id counter
        clear();

        // iterate over all lowdim elements
        const auto lowDimProblem = problem(lowDimIdx);
        for (const auto& lowDimElement : elements(gridView(lowDimIdx)))
        {
            // get the Gaussian quadrature rule for the low dim element
            const auto lowDimGeometry = lowDimElement.geometry();
            const auto& quad = Dune::QuadratureRules<Scalar, lowDimDim>::rule(lowDimGeometry.type(), order);

            const auto lowDimElementIdx = lowDimProblem.fvGridGeometry().elementMapper().index(lowDimElement);

            // apply the Gaussian quadrature rule and define point sources at each quadrature point
            // note that the approximation is not optimal if
            // (a) the one-dimensional elements are too large,
            // (b) whenever a one-dimensional element is split between two or more elements,
            // (c) when gradients of important quantities in the three-dimensional domain are large.

            // iterate over all quadrature points
            for (auto&& qp : quad)
            {
                // global position of the quadrature point
                const auto globalPos = lowDimGeometry.global(qp.position());

                const auto bulkElementIndices = intersectingEntities(globalPos, bulkTree);

                // do not add a point source if the qp is outside of the 3d grid
                // this is equivalent to having a source of zero for that qp
                if (bulkElementIndices.empty())
                    continue;

                //////////////////////////////////////////////////////////
                // get circle average connectivity and interpolation data
                //////////////////////////////////////////////////////////

                static const auto numIp = getParam<int>("MixedDimension.NumCircleSegments");
                const auto radius = lowDimProblem.spatialParams().radius(lowDimElementIdx);
                const auto normal = lowDimGeometry.corner(1)-lowDimGeometry.corner(0);
                const auto length = 2*M_PI*radius/numIp;

                const auto circlePoints = getCirclePoints_(globalPos, normal, radius, numIp);
                std::vector<Scalar> circleIpWeight;
                std::vector<std::size_t> circleStencil;
                for (const auto& p : circlePoints)
                {
                    const auto circleBulkElementIndices = intersectingEntities(p, bulkTree);
                    if (circleBulkElementIndices.empty())
                        continue;

                    for (auto bulkElementIdx : circleBulkElementIndices)
                    {
                        circleStencil.push_back(bulkElementIdx);
                        circleIpWeight.push_back(length);
                    }
                }

                // export low dim circle stencil
                lowDimCouplingStencils_[lowDimElementIdx].insert(lowDimCouplingStencils_[lowDimElementIdx].end(),
                                                                 circleStencil.begin(),
                                                                 circleStencil.end());

                for (auto bulkElementIdx : bulkElementIndices)
                {
                    const auto id = idCounter_++;
                    const auto ie = lowDimGeometry.integrationElement(qp.position());
                    const auto qpweight = qp.weight();

                    bulkPointSources_.emplace_back(globalPos, id, qpweight, ie, std::vector<std::size_t>({bulkElementIdx}));
                    bulkPointSources_.back().setEmbeddings(bulkElementIndices.size());
                    lowDimPointSources_.emplace_back(globalPos, id, qpweight, ie, std::vector<std::size_t>({lowDimElementIdx}));
                    lowDimPointSources_.back().setEmbeddings(bulkElementIndices.size());

                    // pre compute additional data used for the evaluation of
                    // the actual solution dependent source term
                    PointSourceData psData;
                    psData.addLowDimInterpolation(lowDimElementIdx);
                    psData.addBulkInterpolation(bulkElementIdx);
                    // add data needed to compute integral over the circle
                    psData.addCircleInterpolation(circleIpWeight, circleStencil);

                    // publish point source data in the global vector
                    pointSourceData_.push_back(psData);

                    // compute the coupling stencils
                    bulkCouplingStencils_[bulkElementIdx].push_back(lowDimElementIdx);

                    // export bulk circle stencil
                    bulkCircleStencils_[bulkElementIdx].insert(bulkCircleStencils_[bulkElementIdx].end(),
                                                               circleStencil.begin(),
                                                               circleStencil.end());

                    // export inverse bulk circle stencil
                    for (const auto eIdx : circleStencil)
                        bulkCircleStencilsInverse_[eIdx].push_back(bulkElementIdx);
                }
            }
        }

        // make the circle stencil unique
        for (auto&& stencil : bulkCircleStencils_)
        {
            std::sort(stencil.second.begin(), stencil.second.end());
            stencil.second.erase(std::unique(stencil.second.begin(), stencil.second.end()), stencil.second.end());
            stencil.second.erase(std::remove_if(stencil.second.begin(), stencil.second.end(),
                                               [&](auto i){ return i == stencil.first; }),
                                 stencil.second.end());
        }

        for (auto&& stencil : bulkCircleStencilsInverse_)
        {
            std::sort(stencil.second.begin(), stencil.second.end());
            stencil.second.erase(std::unique(stencil.second.begin(), stencil.second.end()), stencil.second.end());
            stencil.second.erase(std::remove_if(stencil.second.begin(), stencil.second.end(),
                                               [&](auto i){ return i == stencil.first; }),
                                 stencil.second.end());
        }

        // make stencils unique
        using namespace Dune::Hybrid;
        forEach(integralRange(Dune::Hybrid::size(problemTuple_)), [&](const auto domainIdx)
        {
            for (auto&& stencil : this->couplingStencils(domainIdx))
            {
                std::sort(stencil.second.begin(), stencil.second.end());
                stencil.second.erase(std::unique(stencil.second.begin(), stencil.second.end()), stencil.second.end());
            }
        });

        std::cout << "took " << watch.elapsed() << " seconds." << std::endl;
    }

    //! Compute the low dim volume fraction in the bulk domain cells
    void computeLowDimVolumeFractions()
    {
        // resize the storage vector
        lowDimVolumeInBulkElement_.resize(gridView(bulkIdx).size(0));

        // compute the low dim volume fractions
        for (const auto& is : intersections(*glue_))
        {
            // all inside elements are identical...
            const auto& inside = is.inside(0);
            const auto intersectionGeometry = is.geometry();
            const std::size_t lowDimElementIdx = problem(lowDimIdx).fvGridGeometry().elementMapper().index(inside);

            // compute the volume the low-dim domain occupies in the bulk domain if it were full-dimensional
            const auto radius = problem(lowDimIdx).spatialParams().radius(lowDimElementIdx);
            for (std::size_t outsideIdx = 0; outsideIdx < is.neighbor(0); ++outsideIdx)
            {
                const auto& outside = is.outside(outsideIdx);
                const std::size_t bulkElementIdx = problem(bulkIdx).fvGridGeometry().elementMapper().index(outside);
                lowDimVolumeInBulkElement_[bulkElementIdx] += intersectionGeometry.volume()*M_PI*radius*radius;
            }
        }
    }

    /*!
     * \brief Methods to be accessed by the subproblems
     */
    // \{

    //! Return a reference to the bulk problem
    Scalar radius(std::size_t id) const
    {
        const auto& data = pointSourceData_[id];
        return problem(lowDimIdx).spatialParams().radius(data.lowDimElementIdx());
    }

    //! The volume the lower dimensional domain occupies in the bulk domain element
    // For one-dimensional low dim domain we assume radial tubes
    Scalar lowDimVolume(const Element<bulkIdx>& element) const
    {
        const auto eIdx = problem(bulkIdx).fvGridGeometry().elementMapper().index(element);
        return lowDimVolumeInBulkElement_[eIdx];
    }

    //! The volume fraction the lower dimensional domain occupies in the bulk domain element
    // For one-dimensional low dim domain we assume radial tubes
    Scalar lowDimVolumeFraction(const Element<bulkIdx>& element) const
    {
        const auto totalVolume = element.geometry().volume();
        return lowDimVolume(element) / totalVolume;
    }

    //! Return a reference to the pointSource data
    const PointSourceData& pointSourceData(std::size_t id) const
    {
        return pointSourceData_[id];
    }

    //! Return a reference to the bulk problem
    template<std::size_t id>
    const Problem<id>& problem(Dune::index_constant<id> domainIdx) const
    {
        return *std::get<id>(problemTuple_);
    }

    //! Return a reference to the bulk problem
    template<std::size_t id>
    Problem<id>& problem(Dune::index_constant<id> domainIdx)
    {
        return *std::get<id>(problemTuple_);
    }

    //! Return a reference to the bulk problem
    template<std::size_t id>
    const GridView<id>& gridView(Dune::index_constant<id> domainIdx) const
    {
        return problem(domainIdx).fvGridGeometry().gridView();
    }

    //! Return data for a bulk point source with the identifier id
    PrimaryVariables<bulkIdx> bulkPriVars(std::size_t id) const
    {
        auto& data = pointSourceData_[id];
        return data.interpolateBulk(curSol_[bulkIdx]);
    }

    //! Return data for a low dim point source with the identifier id
    PrimaryVariables<lowDimIdx> lowDimPriVars(std::size_t id) const
    {
        auto& data = pointSourceData_[id];
        return data.interpolateLowDim(curSol_[lowDimIdx]);
    }

    //! Return reference to bulk point sources
    const std::vector<PointSource<bulkIdx>>& bulkPointSources() const
    { return bulkPointSources_; }

    //! Return reference to low dim point sources
    const std::vector<PointSource<lowDimIdx>>& lowDimPointSources() const
    { return lowDimPointSources_; }

    //! Return reference to bulk coupling stencil member
    template<std::size_t id>
    const CouplingStencils& couplingStencils(Dune::index_constant<id> dom) const
    { return (id == bulkIdx) ? bulkCouplingStencils_ : lowDimCouplingStencils_; }

    //! Clear all internal data members
    void clear()
    {
        bulkPointSources_.clear();
        lowDimPointSources_.clear();
        pointSourceData_.clear();
        bulkCouplingStencils_.clear();
        lowDimCouplingStencils_.clear();
        idCounter_ = 0;
    }

    const std::vector<std::size_t>& getAdditionalDofDependencies(Dune::index_constant<0> bulkDomain, std::size_t bulkElementIdx) const
    { return bulkCircleStencils_.count(bulkElementIdx) ? bulkCircleStencils_.at(bulkElementIdx) : emptyStencil_; }


    const std::vector<std::size_t>& getAdditionalDofDependencies(Dune::index_constant<1> lowDimDomain, std::size_t lowDimElementIdx) const
    { return emptyStencil_; }

    const std::vector<std::size_t>& getAdditionalDofDependenciesInverse(Dune::index_constant<0> bulkDomain, std::size_t bulkElementIdx) const
    { return bulkCircleStencilsInverse_.count(bulkElementIdx) ? bulkCircleStencilsInverse_.at(bulkElementIdx) : emptyStencil_; }

    const std::vector<std::size_t>& getAdditionalDofDependenciesInverse(Dune::index_constant<1> lowDimDomain, std::size_t lowDimElementIdx) const
    { return emptyStencil_; }

    //! Return reference to point source data vector member
    const std::vector<PointSourceData>& pointSourceData() const
    { return pointSourceData_; }

protected:
    //! Return reference to point source data vector member
    std::vector<PointSourceData>& pointSourceData()
    { return pointSourceData_; }

    //! Return reference to bulk point sources
    std::vector<PointSource<bulkIdx>>& bulkPointSources()
    { return bulkPointSources_; }

    //! Return reference to low dim point sources
    std::vector<PointSource<lowDimIdx>>& lowDimPointSources()
    { return lowDimPointSources_; }

    //! Return reference to bulk coupling stencil member
    template<std::size_t id>
    CouplingStencils& couplingStencils(Dune::index_constant<id> dom)
    { return (id == 0) ? bulkCouplingStencils_ : lowDimCouplingStencils_; }

    //! Return a reference to an empty stencil
    const CouplingStencil& emptyStencil() const
    { return emptyStencil_; }

private:

    std::vector<GlobalPosition> getCirclePoints_(const GlobalPosition& center,
                                                 const GlobalPosition& normal,
                                                 const Scalar radius,
                                                 const std::size_t numIp = 20)
    {
        const Scalar eps = 1.5e-7;
        static_assert(dimWorld == 3, "Only implemented for world dimension 3");

        std::vector<GlobalPosition> points(numIp);

        // make sure n is a unit vector
        auto n = normal;
        n /= n.two_norm();

        // caculate a vector u perpendicular to n
        GlobalPosition u;
        if (std::abs(n[0]) < eps && std::abs(n[1]) < eps)
            if (std::abs(n[2]) < eps)
                DUNE_THROW(Dune::MathError, "The normal vector has to be non-zero!");
            else
                u = {0, 1, 0};
        else
            u = {-n[1], n[0], 0};

        u *= radius/u.two_norm();

        // the circle parameterization is p(t) = r*cos(t)*u + r*sin(t)*(n x u) + c
        auto tangent = crossProduct(u, n);
        tangent *= radius/tangent.two_norm();

        // the parameter with an offset
        Scalar t = 0 + 0.1;
        // insert the vertices
        for (std::size_t i = 0; i < numIp; ++i)
        {
            points[i] = GlobalPosition({u[0]*std::cos(t) + tangent[0]*std::sin(t) + center[0],
                                        u[1]*std::cos(t) + tangent[1]*std::sin(t) + center[1],
                                        u[2]*std::cos(t) + tangent[2]*std::sin(t) + center[2]});

            t += 2*M_PI/numIp;

            // periodic t
            if(t > 2*M_PI)
                t -= 2*M_PI;
        }

        return points;
    }

    std::tuple<std::shared_ptr<Problem<0>>, std::shared_ptr<Problem<1>>> problemTuple_;

    std::vector<PointSource<bulkIdx>> bulkPointSources_;
    std::vector<PointSource<lowDimIdx>> lowDimPointSources_;

    mutable std::vector<PointSourceData> pointSourceData_;

    CouplingStencils bulkCouplingStencils_;
    CouplingStencils lowDimCouplingStencils_;
    CouplingStencil emptyStencil_;

    // circle stencils
    CouplingStencils bulkCircleStencils_;
    CouplingStencils bulkCircleStencilsInverse_;

    //! vector for the volume fraction of the lowdim domain in the bulk domain cells
    std::vector<Scalar> lowDimVolumeInBulkElement_;

    //! id generator for point sources
    std::size_t idCounter_ = 0;

    //! The glue object
    std::shared_ptr<GlueType> glue_;

    ////////////////////////////////////////////////////////////////////////////
    //! The coupling context
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    //! TODO: this is the simplest context -> just the solutionvector
    ////////////////////////////////////////////////////////////////////////////
    SolutionVector curSol_;

    // integration order for coupling source
    int integrationOrder_;
};

} // end namespace Dumux

#endif