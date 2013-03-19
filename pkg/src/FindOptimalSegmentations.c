/* -*- compile-command: "R CMD INSTALL .." -*- */ 

//////////////////////////////////
//                              //
//  FindOptimalSegmentations.c  //
//                              //
//////////////////////////////////

#include "FindOptimalSegmentations.h"

#include <stdlib.h>
#include <math.h>
// Assume that the last annotated region is [n-1, n-1] //
// Assume pMax >= 2
void FindOptimalSegmentations(
    const double * x, const unsigned *sR, const unsigned *eR, 
    const unsigned nMax, const unsigned pMax, unsigned * iDPath, double * cost)
{
	//printf("Cumulative sum \n");
    // Cumulative sum of x //
	double * sx = (double *)malloc(nMax * sizeof(double));
	sx[0] = x[0];
	for (unsigned n = 1; n < nMax; ++n)
	{
		sx[n] = sx[n-1] + x[n]; 
	}
	
	//printf("Cost allocation\n");
	// Keeping Cost and Indexes //
	// Initialize Cost object for the different regions //
	double ** C = (double **) malloc(pMax * sizeof(double*));
	for (unsigned p = 0; p < pMax; ++p)
	{
		//printf("Lg Region %d, %d, %d: %d", p, sR[p], eR[p], (eR[p] - sR[p]+1));
		C[p] = (double *) malloc( (eR[p] - sR[p]+1) * sizeof(double));
		for(unsigned i=0; i < (eR[p] - sR[p]+1); ++i)
		{
			C[p][i] = INFINITY;
		}
	}
	// Initialize Last change object for the different regions //

	//printf("Indexes allocation\n");
	unsigned ** M = (unsigned **) malloc(pMax * sizeof(unsigned*));
	for (unsigned p = 0; p < pMax; ++p)
	{
		M[p] = (unsigned *) malloc( (eR[p] - sR[p]+1) * sizeof(unsigned));
		for(unsigned i=0; i < (eR[p] - sR[p]+1); ++i)
		{
			M[p][i] = 0;
		}
	}
	
	
	// Initialization //
	// the first break is in the first region //
	//printf("Initialisation C[0][t]\n");
	unsigned tOut=sR[0];
	unsigned idOut=0;
	double scx;
	unsigned lx;
	while (tOut <= eR[0]) 
	{
		scx = sx[tOut];
		lx = tOut+1;
		C[0][idOut] = -scx*scx / lx;
		//printf("C[0][%d] = %f^2 / %d = %f\n", tOut, scx, lx, C[0][idOut]);
		M[0][idOut] = 0;
		tOut++;
		idOut++;
	}
	
	// Update //
	//printf("Update C[p>=1][t]\n");
	unsigned idIn;
	unsigned tIn;
	double proposedCost;
	for (unsigned p = 1; p < pMax; ++p)
	{
		idIn=0;
		tIn=sR[p-1];
		
		while( tIn <= eR[p-1])
		{	
			// TODO:make sure we are not going backward 
			tOut=sR[p];
			if(tOut < tIn) {
  			//printf("p: %d, tIn :%d, tOut :%d \n", p, tIn, tOut );
			tOut = tIn +1; // ADDED
			}
			//idOut=0; // OLD
			idOut = tOut - sR[p]; // NEW
			while( tOut <= eR[p])
			{
				scx= sx[tOut] - sx[tIn];
				lx= tOut- tIn;
				proposedCost = C[p-1][idIn] - ((scx*scx) / lx);
				if(proposedCost < C[p][idOut])
				{
					C[p][idOut] = proposedCost;
					M[p][idOut] = idIn;
					//printf("From %d: %d, C[%d][%d] = %f - %f^2 / %d = %f\n", tIn, M[p][idOut], p, tOut, C[p-1][idIn],  scx, lx, C[p][idOut]);

				}
				tOut++;
				idOut++;
			}
			
			tIn++;
			idIn++;
		}
	}
	
	
	
	// Finish //
	//printf("Finish %d, %d \n", pMax-1, M[pMax-1][0]);
	iDPath[pMax-1] = M[pMax-1][0];
	cost[0] = C[pMax-1][0];
	//printf("Retour %d, %f\n", M[pMax-1][0], C[pMax-1][0]);
	for (unsigned p = pMax-2; p > 0; --p) 
	{
		//printf("Retour %d, %d, %d\n", p, iDPath[p+1], M[p][iDPath[p+1]]);
		iDPath[p] = M[p][iDPath[p+1]];
	}
	

	// Free allocated object //
	//printf("Free \n");
	free(sx);
	
	for (unsigned p = 0; p < pMax; ++p)
	{
		free(C[p]);
	}
	free(C);
	
	for (unsigned p = 0; p < pMax; ++p)
	{
		free(M[p]);
	}
	free(M);
}

// TDH 19 March 2013 C implementation of SegAnnotBases, to port to
// python for the webserver.
int bases(
    const double * x, const unsigned * base,
    const unsigned *first_base, const unsigned *last_base, 
    unsigned *sR, unsigned *eR, 
    const unsigned nMax, const unsigned n_regions, 
    // need to calculate the path of optimal breaks:
    unsigned * segStart, double * cost) {

    unsigned p, n;
    for(p=0; p<n_regions; p++){
	// Check that regions are increasing.
	if(p>0 && first_base[p] <= first_base[p-1]){
	    return ERROR_REGIONS_NOT_INCREASING;
	}
	if(last_base[p] <= first_base[p]){
	    return ERROR_LAST_BEFORE_FIRST;
	}
    }
    unsigned pMax = n_regions+1;
    //unsigned *sR, *eR;
    //sR = (unsigned *)malloc(pMax * sizeof(unsigned));
    //eR = (unsigned *)malloc(pMax * sizeof(unsigned));

    //printf("Cumulative sum \n");
    // Cumulative sum of x //
    double * sx = (double *)malloc(nMax * sizeof(double));
    sx[0] = x[0];
    unsigned first_p = 0;
    unsigned last_p = 0;
    for (n = 1; n < nMax; ++n)
    {
	//base must be in increasing order!!!
	if(base[n] < base[n-1]){
	    return ERROR_BASES_NOT_INCREASING;
	}
	// Map first_base and last_base to sR and eR.
	// TODO: verify edge cases.n
	if(first_p < n_regions && base[n] > first_base[first_p]){
	    sR[first_p] = n-1;
	    first_p++;
	}
	if(last_p < n_regions && base[n] > last_base[last_p]){
	    eR[last_p] = n;
	    last_p++;
	}
	sx[n] = sx[n-1] + x[n]; 
    }
    // For the case where the last region goes past the last probe:
    if(last_p < n_regions){
	eR[last_p] = n;
	last_p++;
    }
    // last_p == first_p == n_regions
    eR[n_regions] = nMax-1;
    sR[n_regions] = nMax-1;
	
    //printf("Cost allocation\n");
    // Keeping Cost and Indexes //
    // Initialize Cost object for the different regions //
    double ** C = (double **) malloc(pMax * sizeof(double*));
    for (p = 0; p < pMax; ++p)
    {
	//printf("Lg Region %d, %d, %d: %d", p, sR[p], eR[p], (eR[p] - sR[p]+1));
	C[p] = (double *) malloc( (eR[p] - sR[p]+1) * sizeof(double));
	for(unsigned i=0; i < (eR[p] - sR[p]+1); ++i)
	{
	    C[p][i] = INFINITY;
	}
    }
    // Initialize Last change object for the different regions //

    //printf("Indexes allocation\n");
    unsigned ** M = (unsigned **) malloc(pMax * sizeof(unsigned*));
    for (unsigned p = 0; p < pMax; ++p)
    {
	M[p] = (unsigned *) malloc( (eR[p] - sR[p]+1) * sizeof(unsigned));
	for(unsigned i=0; i < (eR[p] - sR[p]+1); ++i)
	{
	    M[p][i] = 0;
	}
    }
	
	
    // Initialization //
    // the first break is in the first region //
    //printf("Initialisation C[0][t]\n");
    unsigned tOut=sR[0];
    unsigned idOut=0;
    double scx;
    unsigned lx;
    while (tOut <= eR[0]) 
    {
	scx = sx[tOut];
	lx = tOut+1;
	C[0][idOut] = -scx*scx / lx;
	//printf("C[0][%d] = %f^2 / %d = %f\n", tOut, scx, lx, C[0][idOut]);
	M[0][idOut] = 0;
	tOut++;
	idOut++;
    }
	
    // Update //
    //printf("Update C[p>=1][t]\n");
    unsigned idIn;
    unsigned tIn;
    double proposedCost;
    for (unsigned p = 1; p < pMax; ++p)
    {
	idIn=0;
	tIn=sR[p-1];
		
	while( tIn <= eR[p-1])
	{	
	    // TODO:make sure we are not going backward 
	    tOut=sR[p];
	    if(tOut < tIn) {
		//printf("p: %d, tIn :%d, tOut :%d \n", p, tIn, tOut );
		tOut = tIn +1; // ADDED
	    }
	    //idOut=0; // OLD
	    idOut = tOut - sR[p]; // NEW
	    while( tOut <= eR[p])
	    {
		scx= sx[tOut] - sx[tIn];
		lx= tOut- tIn;
		proposedCost = C[p-1][idIn] - ((scx*scx) / lx);
		if(proposedCost < C[p][idOut])
		{
		    C[p][idOut] = proposedCost;
		    M[p][idOut] = idIn;
		    //printf("From %d: %d, C[%d][%d] = %f - %f^2 / %d = %f\n", tIn, M[p][idOut], p, tOut, C[p-1][idIn],  scx, lx, C[p][idOut]);

		}
		tOut++;
		idOut++;
	    }
			
	    tIn++;
	    idIn++;
	}
    }
	
	
	
    // Finish //
    //printf("Finish %d, %d \n", pMax-1, M[pMax-1][0]);
    segStart[pMax-1] = M[pMax-1][0];
    cost[0] = C[pMax-1][0];
    //printf("Retour %d, %f\n", M[pMax-1][0], C[pMax-1][0]);
    for (unsigned p = pMax-2; p > 0; --p) 
    {
	//printf("Retour %d, %d, %d\n", p, segStart[p+1], M[p][segStart[p+1]]);
	segStart[p] = M[p][segStart[p+1]];
    }
    for(p=1;p<pMax;p++){
	segStart[p] += sR[p-1] + 1;
    }
	

    // Free allocated object //
    //printf("Free \n");
    free(sx);
	
    for (unsigned p = 0; p < pMax; ++p)
    {
	free(C[p]);
    }
    free(C);
	
    for (unsigned p = 0; p < pMax; ++p)
    {
	free(M[p]);
    }
    free(M);

    //free(sR);
    //free(eR);

    return 0;
}
