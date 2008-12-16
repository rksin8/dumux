// $Id$

#ifndef DUNE_BOX2P2CJACOBIAN_HH
#define DUNE_BOX2P2CJACOBIAN_HH

#include<map>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<vector>
#include<sstream>

#include<dune/common/exceptions.hh>
#include<dune/grid/common/grid.hh>
#include<dune/grid/common/referenceelements.hh>
#include<dune/common/geometrytype.hh>
#include<dune/grid/common/quadraturerules.hh>
#include<dune/grid/utility/intersectiongetter.hh>
#include<dune/disc/shapefunctions/lagrangeshapefunctions.hh>
#include<dune/disc/operators/boundaryconditions.hh>
#include"dumux/operators/boxjacobian.hh"
#include"dumux/2p2c/2p2cproblem.hh"
#include"dumux/io/vtkmultiwriter.hh"

/**
 * @file
 * @brief  compute local jacobian matrix for box scheme for two-phase two-component flow equation
 * @author Bernd Flemisch, Klaus Mosthaf
 */



namespace Dune
{
  /** @addtogroup DISC_Disc
   *
   * @{
   */
  /**
   * @brief compute local jacobian matrix for the boxfile for two-phase two-component flow equation
   *
   */


  //! Derived class for computing local jacobian matrices
  /*! A class for computing local jacobian matrix for the two-phase two-component flow equation

	    div j = q; j = -K grad u; in Omega

		u = g on Gamma1; j*n = J on Gamma2.

	Uses box scheme with the Lagrange shape functions.
	It should work for all dimensions and element types.
	All the numbering is with respect to the reference element and the
	Lagrange shape functions

	Template parameters are:

	- Grid  a DUNE grid type
	- RT    type used for return values
  */
  template<class G, class RT, class BoxFunction = LeafP1FunctionExtended<G, RT, 2> >
  class Box2P2CJacobian
    : public BoxJacobian<Box2P2CJacobian<G,RT,BoxFunction>,G,RT,2,BoxFunction>
  {
    typedef typename G::ctype DT;
    typedef typename G::Traits::template Codim<0>::Entity Entity;
    typedef typename Entity::Geometry Geometry;
    typedef Box2P2CJacobian<G,RT,BoxFunction> ThisType;
    typedef typename LocalJacobian<ThisType,G,RT,2>::VBlockType VBlockType;
    typedef typename LocalJacobian<ThisType,G,RT,2>::MBlockType MBlockType;

 	enum {pWIdx = 0, switchIdx = 1, numberOfComponents = 2};	// Solution vector index
	enum {wPhase = 0, nPhase = 1};									// Phase index
	enum {gasPhase = 0, waterPhase = 1, bothPhases = 2};		// Phase state
	enum {water = 0, air = 1};										// Component index

  public:
    // define the number of phases (m) and components (c) of your system, this is used outside
    // to allocate the correct size of (dense) blocks with a FieldMatrix
    enum {dim=G::dimension};
    enum {m=2, c=2};
    enum {SIZE=LagrangeShapeFunctionSetContainer<DT,RT,dim>::maxsize};
    struct VariableNodeData;

    typedef FieldMatrix<RT,dim,dim> FMatrix;
    typedef FieldVector<RT,dim> FVector;

    //! Constructor
    Box2P2CJacobian (TwoPTwoCProblem<G,RT>& params, bool levelBoundaryAsDirichlet_, const G& grid,
			      BoxFunction& sol, bool procBoundaryAsDirichlet_=true)
    : BoxJacobian<ThisType,G,RT,2,BoxFunction>(levelBoundaryAsDirichlet_, grid, sol, procBoundaryAsDirichlet_),
      problem(params), sNDat(this->vertexMapper.size()), vNDat(SIZE), oldVNDat(SIZE), switchFlag(false)
    {
      this->analytic = false;
      switchFlag = false;
      switchFlagLocal = false;
      temperature = 283.15;
    }

	/** @brief compute time dependent term (storage), loop over nodes / subcontrol volumes
	 *  @param e entity
	 *  @param sol solution vector
	 *  @param node local node id
	 *  @return storage term
	 */
    virtual VBlockType computeM (const Entity& e, const VBlockType* sol,
    		int node, std::vector<VariableNodeData>& varData)
    {
    	 GeometryType gt = e.geometry().type();
    	 const typename LagrangeShapeFunctionSetContainer<DT,RT,dim>::value_type&
     	 sfs=LagrangeShapeFunctions<DT,RT,dim>::general(gt,1);

   	 int globalIdx = this->vertexMapper.template map<dim>(e, sfs[node].entity());

   	 VBlockType result;
   	 RT satN = varData[node].satN;
   	 RT satW = varData[node].satW;

   	 // storage of component water
   	 result[water] =
   		 sNDat[globalIdx].porosity*(varData[node].density[wPhase]*satW*varData[node].massfrac[water][wPhase]
   		                +varData[node].density[nPhase]*satN*varData[node].massfrac[water][nPhase]);
   	 // storage of component air
   	 result[air] =
   		 sNDat[globalIdx].porosity*(varData[node].density[nPhase]*satN*varData[node].massfrac[air][nPhase]
   	                    +varData[node].density[wPhase]*satW*varData[node].massfrac[air][wPhase]);

   	 //std::cout << result << " " << node << std::endl;
   	 return result;
    };

    virtual VBlockType computeM (const Entity& e, const VBlockType* sol, int node, bool old = false)
    {
    	if (old)
    		return computeM(e, sol, node, oldVNDat);
    	else
    		return computeM(e, sol, node, vNDat);
    }

    /** @brief compute diffusive/advective fluxes, loop over subcontrol volume faces
	 *  @param e entity
	 *  @param sol solution vector
	 *  @param face face id
	 *  @return flux term
     */
    virtual VBlockType computeA (const Entity& e, const VBlockType* sol, int face)
    {
   	 int i = this->fvGeom.subContVolFace[face].i;
 	 int j = this->fvGeom.subContVolFace[face].j;

 	 // normal vector, value of the area of the scvf
	 const FieldVector<RT,dim> normal(this->fvGeom.subContVolFace[face].normal);
	 GeometryType gt = e.geometry().type();
	 const typename LagrangeShapeFunctionSetContainer<DT,RT,dim>::value_type&
 	 sfs=LagrangeShapeFunctions<DT,RT,dim>::general(gt,1);

	 // global index of the subcontrolvolume face neighbor nodes in element e
	 int globalIdx_i = this->vertexMapper.template map<dim>(e, sfs[i].entity());
  	 int globalIdx_j = this->vertexMapper.template map<dim>(e, sfs[j].entity());

  	 // get global coordinates of nodes i,j
	 const FieldVector<DT,dim> global_i = this->fvGeom.subContVol[i].global;
	 const FieldVector<DT,dim> global_j = this->fvGeom.subContVol[j].global;
	 const FieldVector<DT,dim> local_i = this->fvGeom.subContVol[i].local;
	 const FieldVector<DT,dim> local_j = this->fvGeom.subContVol[j].local;

	 FieldMatrix<RT,m,dim> pGrad(0.), xGrad(0.);
	 FieldVector<RT,dim> temp(0.);
     VBlockType flux(0.);
	 FMatrix Ki(0), Kj(0);

 	 // calculate harmonic mean of permeabilities of nodes i and j
 	 Ki = this->problem.soil().K(global_i,e,local_i);
 	 Kj = this->problem.soil().K(global_j,e,local_j);
	 const FMatrix K = harmonicMeanK(Ki, Kj);

	 // calculate FE gradient (grad p for each phase)
	 for (int k = 0; k < this->fvGeom.numVertices; k++) // loop over adjacent nodes
	 {
		 // FEGradient at node k
		 const FieldVector<DT,dim> feGrad(this->fvGeom.subContVolFace[face].grad[k]);
		 FieldVector<RT,m> pressure(0.0), massfrac(0.0);

		 pressure[wPhase] = vNDat[k].pW;
		 pressure[nPhase] = vNDat[k].pN;

	  	 // compute sum of pressure gradients for each phase
	  	 for (int phase = 0; phase < m; phase++)
	  	 {
	  		 temp = feGrad;
	  		 temp *= pressure[phase];
	  		 pGrad[phase] += temp;
	  	 }
	  	 // for diffusion of air in wetting phase
	  	 temp = feGrad;
     	 temp *= vNDat[k].massfrac[air][wPhase];
     	 xGrad[wPhase] += temp;

	  	 // for diffusion of water in nonwetting phase
     	 temp = feGrad;
     	 temp *= vNDat[k].massfrac[water][nPhase];
     	 xGrad[nPhase] += temp;
	 }

	 // deduce gravity*density of each phase
	 FieldMatrix<RT,m,dim> contribComp(0);
	 for (int phase=0; phase<m; phase++)
	 {
		 contribComp[phase] = problem.gravity();
		 contribComp[phase] *= vNDat[i].density[phase];
		 pGrad[phase] -= contribComp[phase]; // grad p - rho*g
	 }

	 VBlockType outward(0);  // Darcy velocity of each phase

	 // calculate the advective flux using upwind: K*n(grad p -rho*g)
	 for (int phase=0; phase<m; phase++)
	 {
		 FieldVector<RT,dim> v_tilde(0);
		 K.mv(pGrad[phase], v_tilde);  // v_tilde=K*gradP
     	 outward[phase] = v_tilde*normal;
	 }

	 // evaluate upwind nodes
	 int up_w, dn_w, up_n, dn_n;

	 if (outward[wPhase] <= 0) {up_w = i; dn_w = j;}
	 else {up_w = j; dn_w = i;};
	 if (outward[nPhase] <= 0) {up_n = i; dn_n = j;}
	 else {up_n = j; dn_n = i;};

	 RT alpha = 1.0;  // Upwind parameter

	 // water conservation
	 flux[water] =   (alpha* vNDat[up_w].density[wPhase]*vNDat[up_w].mobility[wPhase]
                          * vNDat[up_w].massfrac[water][wPhase]
                          + (1-alpha)* vNDat[dn_w].density[wPhase]*vNDat[dn_w].mobility[wPhase]
                          * vNDat[dn_w].massfrac[water][wPhase])
             * outward[wPhase];
	 flux[water] +=  (alpha* vNDat[up_n].density[nPhase]*vNDat[up_n].mobility[nPhase]
                          * vNDat[up_n].massfrac[water][nPhase]
                          + (1-alpha)* vNDat[dn_n].density[nPhase]*vNDat[dn_n].mobility[nPhase]
                          * vNDat[dn_n].massfrac[water][nPhase])
             * outward[nPhase];
	 // air conservation
	 flux[air]   =   (alpha* vNDat[up_n].density[nPhase]*vNDat[up_n].mobility[nPhase]
                          * vNDat[up_n].massfrac[air][nPhase]
                          + (1-alpha)* vNDat[dn_n].density[nPhase]*vNDat[dn_n].mobility[nPhase]
                          * vNDat[dn_n].massfrac[air][nPhase])
             * outward[nPhase];
	 flux[air]  +=   (alpha* vNDat[up_w].density[wPhase]*vNDat[up_w].mobility[wPhase]
                          * vNDat[up_w].massfrac[air][wPhase]
                          + (1-alpha)* vNDat[dn_w].density[wPhase]*vNDat[dn_w].mobility[wPhase]
                          * vNDat[dn_w].massfrac[air][wPhase])
             * outward[wPhase];

         return flux;

	 // DIFFUSION
	 VBlockType normDiffGrad;

	 // get local to global id map
	 int state_i = vNDat[i].phasestate;
	 int state_j = vNDat[j].phasestate;

	 RT diffusionWW(0.0), diffusionWN(0.0); // diffusion of water
	 RT diffusionAW(0.0), diffusionAN(0.0); // diffusion of air
	 VBlockType avgDensity, avgDpm;

         // calculate tortuosity at the nodes i and j needed for porous media diffusion coefficient
	 RT tauW_i, tauW_j, tauN_i, tauN_j; // tortuosity of wetting and nonwetting phase
	 tauW_i = pow(sNDat[globalIdx_i].porosity * vNDat[i].satW,(7/3))/
             (sNDat[globalIdx_i].porosity*sNDat[globalIdx_i].porosity);
	 tauW_j = pow(sNDat[globalIdx_j].porosity * vNDat[j].satW,(7/3))/
             (sNDat[globalIdx_j].porosity*sNDat[globalIdx_j].porosity);
	 tauN_i = pow(sNDat[globalIdx_i].porosity * vNDat[i].satN,(7/3))/
             (sNDat[globalIdx_i].porosity*sNDat[globalIdx_i].porosity);
	 tauN_j = pow(sNDat[globalIdx_j].porosity * vNDat[j].satN,(7/3))/
             (sNDat[globalIdx_j].porosity*sNDat[globalIdx_j].porosity);

	 // arithmetic mean of porous media diffusion coefficient
	 RT Dwn, Daw;
	 Dwn = (sNDat[globalIdx_i].porosity * vNDat[i].satN * tauN_i * vNDat[i].diff[nPhase] +
                sNDat[globalIdx_j].porosity * vNDat[j].satN * tauN_j * vNDat[j].diff[nPhase])/2;
	 Daw = (sNDat[globalIdx_i].porosity * vNDat[i].satW * tauW_i * vNDat[i].diff[wPhase] +
                sNDat[globalIdx_j].porosity * vNDat[j].satW * tauW_j * vNDat[j].diff[wPhase])/2;

//
//	 avgDpm[wPhase]=2e-9; // needs to be changed !!!
//	 avgDpm[nPhase]=2.25e-5; // water in the gasphase

	 // adapt the diffusion coefficent according to the phase state.
	 // TODO: make this continuously dependent on the phase saturations
	 if (state_i == gasPhase || state_j == gasPhase) {
             // one element is only gas -> no diffusion in water phase
             avgDpm[wPhase] = 0;
	 }
	 if (state_i == waterPhase || state_j == waterPhase) {
             // one element is only water -> no diffusion in gas phase
             avgDpm[nPhase] = 0;
	 }

	 normDiffGrad[wPhase] = xGrad[wPhase]*normal;
	 normDiffGrad[nPhase] = xGrad[nPhase]*normal;

	 // calculate the arithmetic mean of densities
	 avgDensity[wPhase] = 0.5*(vNDat[i].density[wPhase] + vNDat[j].density[wPhase]);
	 avgDensity[nPhase] = 0.5*(vNDat[i].density[nPhase] + vNDat[j].density[nPhase]);

	 // diffusion in the wetting phase
	 diffusionAW = Daw * avgDensity[wPhase] * normDiffGrad[wPhase];
	 diffusionWW = - diffusionAW; // air must be replaced by water
	 diffusionWN = Dwn * avgDensity[nPhase] * normDiffGrad[nPhase];
	 diffusionAN = - diffusionWN;

	 // add diffusion of water to flux
//	 flux[water] += (diffusionWW + diffusionWN);
	 //	std::cout << "Water Flux: " << flux[water] << std::endl;

	 // add diffusion of air to flux
//	 flux[air] += (diffusionAN + diffusionAW);
	 // std::cout << "Air Flux: " << flux[air] << std::endl;


	 return flux;
    };

      /** @brief integrate sources / sinks
       *  @param e entity
       *  @param sol solution vector
       *  @param node local node id
       *  @return source/sink term
       */
      virtual VBlockType computeQ (const Entity& e, const VBlockType* sol, const int node)
          {
              // ASSUME problem.q already contains \rho.q
              return problem.q(this->fvGeom.subContVol[node].global, e, this->fvGeom.subContVol[node].local);
          }

      /** @brief perform variable switch
       *  @param global global node id
       *  @param sol solution vector
       *  @param local local node id
       */
      virtual void primaryVarSwitch (const Entity& e, int globalIdx, VBlockType* sol, int localIdx)
          {
        bool switched = false;
        const FVector global = this->fvGeom.subContVol[localIdx].global;
        const FVector local = this->fvGeom.subContVol[localIdx].local;
        int state = sNDat[globalIdx].phaseState;
//        int switch_counter = sNDat[globalIdx].switched;

        // Evaluate saturation and pressures first
        RT pW = sol[localIdx][pWIdx];
        RT satW = 0.0;
        if (state == bothPhases)
            satW = 1.0-sol[localIdx][switchIdx];
        if (state == waterPhase)
            satW = 1.0;
        if (state == gasPhase)
            satW = 0.0;
    	RT pC = problem.materialLaw().pC(satW, global, e, local);
        RT pN = pW + pC;

        switch(state)
        {
            case gasPhase :
        	RT xWNmass, xWNmolar, pwn, pWSat; // auxiliary variables

        	xWNmass = sol[localIdx][switchIdx];
        	xWNmolar = problem.multicomp().convertMassToMoleFraction(xWNmass, gasPhase);
           	pwn = xWNmolar * pN;
			pWSat = problem.multicomp().vaporPressure(temperature);

        	if (pwn > (1 + 1e-5)*pWSat && !switched)// && switch_counter < 3)
			{
        		// appearance of water phase
				std::cout << "Water appears at node " << globalIdx << "  Coordinates: " << global << std::endl;
				sNDat[globalIdx].phaseState = bothPhases;
				sol[localIdx][switchIdx] = 1.0 - 2e-5; // initialize solution vector
				sNDat[globalIdx].switched += 1;
				switched = true;
            }
            break;

            case waterPhase :
        	RT xAWmass, xAWmolar, henryInv, pbub; // auxiliary variables

        	xAWmass = sol[localIdx][switchIdx];
         	xAWmolar = problem.multicomp().convertMassToMoleFraction(xAWmass, waterPhase);

            henryInv = problem.multicomp().henry(temperature);
            pWSat = problem.multicomp().vaporPressure(temperature);
        	pbub = pWSat + xAWmolar/henryInv; // pWSat + pAW

        	if (pN < (1 - 1e-5)*pbub && !switched)// && switch_counter < 3)
			{
				// appearance of gas phase
				std::cout << "Gas appears at node " << globalIdx << ",  Coordinates: " << global << std::endl;
				sNDat[globalIdx].phaseState = bothPhases;
				sol[localIdx][switchIdx] = 2e-5; // initialize solution vector
				sNDat[globalIdx].switched += 1;
				switched = true;
			}
			break;

            case bothPhases:
        	RT satN = sol[localIdx][switchIdx];

        	if (satN < -1e-5  && !switched)// && switch_counter < 3)
      	  	{
				// disappearance of gas phase
				std::cout << "Gas disappears at node " << globalIdx << "  Coordinates: " << global << std::endl;
				sNDat[globalIdx].phaseState = waterPhase;
				sol[localIdx][switchIdx] = problem.multicomp().xAW(pN); // initialize solution vector
				sNDat[globalIdx].switched += 1;
				switched = true;
			}
        	else if (satW < -1e-5 && !switched)// && switch_counter < 3)
      	  	{
				// disappearance of water phase
				std::cout << "Water disappears at node " << globalIdx << "  Coordinates: " << global << std::endl;
				sNDat[globalIdx].phaseState = gasPhase;
				sol[localIdx][switchIdx] = problem.multicomp().xWN(pN); // initialize solution vector
				sNDat[globalIdx].switched += 1;
				switched = true;
      	  	}
      	  	break;

        }
        if (switched){
            updateVariableData(e, sol, localIdx, vNDat, sNDat[globalIdx].phaseState);
            BoxJacobian<ThisType,G,RT,2,BoxFunction>::localToGlobal(e,sol);
            setSwitchedLocal(); // if switch is triggered at any node, switchFlagLocal is set
        }

   	return;
    }

    // harmonic mean of the permeability computed directly
    virtual FMatrix harmonicMeanK (FMatrix& Ki, const FMatrix& Kj) const
    {
    	double eps = 1e-20;

    	for (int kx=0; kx<dim; kx++){
    		for (int ky=0; ky<dim; ky++){
    			if (Ki[kx][ky] != Kj[kx][ky])
    			{
    				Ki[kx][ky] = 2 / (1/(Ki[kx][ky]+eps) + (1/(Kj[kx][ky]+eps)));
    			}
    		}
    	}
   	 return Ki;
    }


    virtual void clearVisited ()
    {
    	for (int i = 0; i < this->vertexMapper.size(); i++){
   		sNDat[i].visited = false;
//   	 	sNDat[i].switched = false;
    	}
   	 return;
   	}

    // updates old phase state after each time step
    virtual void updatePhaseState ()
    {
      	for (int i = 0; i < this->vertexMapper.size(); i++){
       		sNDat[i].oldPhaseState = sNDat[i].phaseState;
      	 }
       return;
    }

    virtual void resetPhaseState ()
    {
      	for (int i = 0; i < this->vertexMapper.size(); i++){
       		sNDat[i].phaseState = sNDat[i].oldPhaseState;
      	 }
       return;
    }

	  //*********************************************************
	  //*														*
	  //*	Calculation of Data at Elements (elData) 			*
	  //*						 								*
	  //*														*
	  //*********************************************************

    virtual void computeElementData (const Entity& e)
    {
//		 // ASSUMING element-wise constant permeability, evaluate K at the element center
// 		 elData.K = problem.K(this->fvGeom.elementGlobal, e, this->fvGeom.elementLocal);
//
//		 // ASSUMING element-wise constant porosity
// 		 elData.porosity = problem.porosity(this->fvGeom.elementGlobal, e, this->fvGeom.elementLocal);
   	 return;
    }


	  //*********************************************************
	  //*														*
	  //*	Calculation of Data at Nodes that has to be			*
	  //*	determined only once	(sNDat)						*
	  //*														*
	  //*********************************************************

    // analog to EvalStaticData in MUFTE
    virtual void updateStaticDataVS (const Entity& e, VBlockType* sol)
    {
   	 // size of the sNDat vector is determined in the constructor

   	 // get access to shape functions for P1 elements
   	 GeometryType gt = e.geometry().type();
   	 const typename LagrangeShapeFunctionSetContainer<DT,RT,dim>::value_type&
   	 sfs=LagrangeShapeFunctions<DT,RT,dim>::general(gt,1);

   	 // get local to global id map
   	 for (int k = 0; k < sfs.size(); k++)
   	 {
  		 const int globalIdx = this->vertexMapper.template map<dim>(e, sfs[k].entity());

  		 // if nodes are not already visited
  		 if (!sNDat[globalIdx].visited)
  		  {
   			  // evaluate primary variable switch
 			  primaryVarSwitch(e, globalIdx, sol, k);

  			  // mark elements that were already visited
  			  sNDat[globalIdx].visited = true;
  		  }
  	  }

	  return;
    }

    // for initialization of the Static Data (sets porosity)
    virtual void updateStaticData (const Entity& e, VBlockType* sol)
    {
   	 // get access to shape functions for P1 elements
   	 GeometryType gt = e.geometry().type();
   	 const typename LagrangeShapeFunctionSetContainer<DT,RT,dim>::value_type&
   	 sfs=LagrangeShapeFunctions<DT,RT,dim>::general(gt,1);

   	 // get local to global id map
   	 for (int k = 0; k < sfs.size(); k++)
   	 {
   		 const int globalIdx = this->vertexMapper.template map<dim>(e, sfs[k].entity());

  		 // if nodes are not already visited
  		 if (!sNDat[globalIdx].visited)
  		 {
  			 // ASSUME porosity defined at nodes
  			 sNDat[globalIdx].porosity = problem.soil().porosity(this->fvGeom.elementGlobal, e, this->fvGeom.elementLocal);

  			 // set counter for variable switch to zero
  			 sNDat[globalIdx].switched = 0;

//  			 if (!checkSwitched())
//  		   	 {
			 primaryVarSwitch(e, globalIdx, sol, k);
//  		   	 }

  			 // mark elements that were already visited
  			 sNDat[globalIdx].visited = true;
  		 }
   	 }

   	 return;
    }


	  //*********************************************************
	  //*														*
	  //*	Calculation of variable Data at Nodes				*
	  //*	(vNDat)												*
	  //*														*
	  //*********************************************************


    struct VariableNodeData
    {
   	 RT satN;
     RT satW;
     RT pW;
     RT pC;
     RT pN;
     RT temperature;
     VBlockType mobility;  //Vector with the number of phases
     VBlockType density;
     FieldMatrix<RT,c,m> massfrac;
     int phasestate;
     VBlockType diff;
    };

    // analog to EvalPrimaryData in MUFTE, uses members of vNDat
	virtual void updateVariableData(const Entity& e, const VBlockType* sol,
			int i, std::vector<VariableNodeData>& varData, int state)
    {
   	   	 const int globalIdx = this->vertexMapper.template map<dim>(e, i);
   	   	 FVector& global = this->fvGeom.subContVol[i].global;
   	   	 FVector& local = this->fvGeom.subContVol[i].local;

   		 varData[i].pW = sol[i][pWIdx];
   		 if (state == bothPhases) varData[i].satN = sol[i][switchIdx];
   		 if (state == waterPhase) varData[i].satN = 0.0;
   		 if (state == gasPhase) varData[i].satN = 1.0;

   		 varData[i].satW = 1.0 - varData[i].satN;

   		 varData[i].pC = problem.materialLaw().pC(varData[i].satW, global, e, local);
   		 varData[i].pN = varData[i].pW + varData[i].pC;
   		 varData[i].temperature = temperature; // in [K], constant

   		 // Solubilities of components in phases
   		 if (state == bothPhases){
   	   		 varData[i].massfrac[air][wPhase] = problem.multicomp().xAW(varData[i].pN, varData[i].temperature);
   	   		 varData[i].massfrac[water][nPhase] = problem.multicomp().xWN(varData[i].pN, varData[i].temperature);
   		 }
   		 if (state == waterPhase){
   	   		 varData[i].massfrac[water][nPhase] = 0.0;
   	   		 varData[i].massfrac[air][wPhase] =  sol[i][switchIdx];
   		 }
   		 if (state == gasPhase){
   	   		 varData[i].massfrac[water][nPhase] = sol[i][switchIdx];
   	   		 varData[i].massfrac[air][wPhase] = 0.0;
   		 }
   	   	 varData[i].massfrac[water][wPhase] = 1.0 - varData[i].massfrac[air][wPhase];
   	   	 varData[i].massfrac[air][nPhase] = 1.0 - varData[i].massfrac[water][nPhase];
   	   	 varData[i].phasestate = state;

   		 // Mobilities & densities
   		 varData[i].mobility[wPhase] = problem.materialLaw().mobW(varData[i].satW, global, e, local, varData[i].temperature, varData[i].pW);
   		 varData[i].mobility[nPhase] = problem.materialLaw().mobN(varData[i].satN, global, e, local, varData[i].temperature, varData[i].pN);
   		 // Density of Water is set constant here!
   		 varData[i].density[wPhase] = 1000;//problem.wettingPhase().density(varData[i].temperature, varData[i].pN);
   		 varData[i].density[nPhase] = problem.nonwettingPhase().density(varData[i].temperature, varData[i].pN,
   				 varData[i].massfrac[water][nPhase]);

         varData[i].diff[wPhase] = problem.wettingPhase().diffCoeff();
         varData[i].diff[nPhase] = problem.nonwettingPhase().diffCoeff();

         // CONSTANT solubility (for comparison with twophase)
//         varData[i].massfrac[air][wPhase] = 0.0; varData[i].massfrac[water][wPhase] = 1.0;
//         varData[i].massfrac[water][nPhase] = 0.0; varData[i].massfrac[air][nPhase] = 1.0;

         //std::cout << "water in gasphase: " << varData[i].massfrac[water][nPhase] << std::endl;
         //std::cout << "air in waterphase: " << varData[i].massfrac[air][wPhase] << std::endl;

   		 // for output
   		 (*outPressureN)[globalIdx] = varData[i].pN;
   		 (*outCapillaryP)[globalIdx] = varData[i].pC;
  	   	 (*outSaturationW)[globalIdx] = varData[i].satW;
   	   	 (*outSaturationN)[globalIdx] = varData[i].satN;
   	   	 (*outMassFracAir)[globalIdx] = varData[i].massfrac[air][wPhase];
   	   	 (*outMassFracWater)[globalIdx] = varData[i].massfrac[water][nPhase];
   	   	 (*outDensityW)[globalIdx] = varData[i].density[wPhase];
   	   	 (*outDensityN)[globalIdx] = varData[i].density[nPhase];
   	   	 (*outMobilityW)[globalIdx] = varData[i].mobility[wPhase];
   	   	 (*outMobilityN)[globalIdx] = varData[i].mobility[nPhase];
   	   	 (*outPhaseState)[globalIdx] = varData[i].phasestate;

   	   	 return;
    }

	virtual void updateVariableData(const Entity& e, const VBlockType* sol, int i, bool old = false)
	{
		int state;
		const int global = this->vertexMapper.template map<dim>(e, i);
		if (old)
		{
	   	   	state = sNDat[global].oldPhaseState;
			updateVariableData(e, sol, i, oldVNDat, state);
		}
		else
		{
		    state = sNDat[global].phaseState;
			updateVariableData(e, sol, i, vNDat, state);
		}
	}

	void updateVariableData(const Entity& e, const VBlockType* sol, bool old = false)
	{
		int size = this->fvGeom.numVertices;

		for (int i = 0; i < size; i++)
				updateVariableData(e, sol, i, old);
	}

	bool checkSwitched()
	{
		return switchFlag;
	}

	bool checkSwitchedLocal()
	{
		return switchFlagLocal;
	}

	void setSwitchedLocalToGlobal()
	{
		switchFlag = switchFlagLocal;
		return;
	}

	void setSwitchedLocal()
	{
		switchFlagLocal = true;
		return;
	}

	void resetSwitched()
	{
		switchFlag = false;
		return;
	}

	void resetSwitchedLocal()
	{
		switchFlagLocal = false;
		return;
	}

	struct StaticNodeData
    {
   	 bool visited;
   	 int switched;
   	 int phaseState;
   	 int oldPhaseState;
   	 RT elementVolume;
   	 RT porosity;
   	 FMatrix K;
    };

    struct ElementData {
//   	 RT elementVolume;
//     	 RT porosity;
//   	 RT gravity;
   	 } elData;


    // parameters given in constructor
    TwoPTwoCProblem<G,RT>& problem;
    CWaterAir multicomp;
    std::vector<StaticNodeData> sNDat;
    std::vector<VariableNodeData> vNDat;
    std::vector<VariableNodeData> oldVNDat;

    // for output files
    BlockVector<FieldVector<RT, 1> > *outPressureN;
    BlockVector<FieldVector<RT, 1> > *outCapillaryP;
    BlockVector<FieldVector<RT, 1> > *outSaturationN;
    BlockVector<FieldVector<RT, 1> > *outSaturationW;
    BlockVector<FieldVector<RT, 1> > *outMassFracAir;
    BlockVector<FieldVector<RT, 1> > *outMassFracWater;
    BlockVector<FieldVector<RT, 1> > *outDensityW;
    BlockVector<FieldVector<RT, 1> > *outDensityN;
    BlockVector<FieldVector<RT, 1> > *outMobilityW;
    BlockVector<FieldVector<RT, 1> > *outMobilityN;
    BlockVector<FieldVector<RT, 1> > *outPhaseState;
//	BlockVector<FieldVector<RT, 1> > *outPermeability;

  protected:
		bool switchFlag;
		bool switchFlagLocal;
		double temperature;
  };

}
#endif
