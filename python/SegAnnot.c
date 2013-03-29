#include "SegAnnot.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int SegAnnotBases(
    const double * x, const int * base,
    const int *first_base, const int *last_base, 
    const int nMax, const int n_regions, 
    // need to calculate the path of optimal breaks:
    int * segStart, int * segEnd, double * segMean,
    int *break_min, int *break_mid, int *break_max) {

    int p, n, i;
    for(p=0; p<n_regions; p++){
	// Check that regions are increasing.
	if(p>0 && first_base[p] <= first_base[p-1]){
	    return ERROR_REGIONS_NOT_INCREASING;
	}
	if(last_base[p] <= first_base[p]){
	    return ERROR_LAST_BEFORE_FIRST;
	}
    }
    int pMax = n_regions+1;
    int *sR, *eR;
    sR = (int *)malloc(pMax * sizeof(int));
    eR = (int *)malloc(pMax * sizeof(int));

    //printf("Cumulative sum \n");
    // Cumulative sum of x //
    double * sx = (double *)malloc(nMax * sizeof(double));
    sx[0] = x[0];
    int first_p = 0;
    int last_p = 0;
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
	for(i=0; i < (eR[p] - sR[p]+1); ++i)
	{
	    C[p][i] = INFINITY;
	}
    }
    // Initialize Last change object for the different regions //

    //printf("Indexes allocation\n");
    int ** M = (int **) malloc(pMax * sizeof(int*));
    for ( p = 0; p < pMax; ++p)
    {
	M[p] = (int *) malloc( (eR[p] - sR[p]+1) * sizeof(int));
	for(i=0; i < (eR[p] - sR[p]+1); ++i)
	{
	    M[p][i] = 0;
	}
    }
	
	
    // Initialization //
    // the first break is in the first region //
    //printf("Initialisation C[0][t]\n");
    int tOut=sR[0];
    int idOut=0;
    double scx;
    int lx;
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
    int idIn;
    int tIn;
    double proposedCost;
    for ( p = 1; p < pMax; ++p)
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
    //printf("Finish %d, %d \n", n_regions-1, M[n_regions-1][0]);
    int *break_before = (int*)malloc(sizeof(int)*n_regions);
    break_before[n_regions-1] = M[n_regions][0];
    //cost[0] = C[n_regions-1][0];
    //printf("Retour %d, %f\n", M[n_regions-1][0], C[n_regions-1][0]);
    for ( p = n_regions-2; p >= 0; --p) 
    {
	//printf("Retour %d, %d, %d\n", p, break_before[p+1], M[p][break_before[p+1]]);
	break_before[p] = M[p+1][break_before[p+1]];
    }
    for(p=0;p<n_regions;p++){
	break_before[p] += sR[p] + 1;
    }
    // At this point break_before is an array of 
    // length n_regions, containing 0-indexed indices
    // of the first probe on
    // the 2nd, ..., kth segment.
    
    // Translate back to base pairs.
    int *first_probe = (int*)malloc(sizeof(int)*pMax);
    int *last_probe = (int*)malloc(sizeof(int)*pMax);
    first_probe[0] = 0;
    last_probe[pMax-1] = nMax-1;
    for(p=0; p<n_regions; p++){
	i = break_before[p];
	first_probe[p+1] = i;
	last_probe[p] = i-1;
	break_min[p] = base[i-1];
	break_max[p] = base[i];
	break_mid[p] = (break_min[p]+break_max[p])/2;
	//printf("%d\n",bkpts[p]);
    }
    // Calculate segment means.
    double total;
    for(p=0; p<pMax; p++){
	if(p==0){
	    segStart[p] = base[0];
	}else{
	    segStart[p] = break_mid[p-1];
	}
	if(p==pMax-1){
	    segEnd[p] = base[nMax-1];
	}else{
	    segEnd[p] = break_mid[p];
	}
	total = 0;
	for(i=first_probe[p]; i<=last_probe[p]; i++){
	    total += x[i];
	}
	segMean[p] = total/(last_probe[p]-first_probe[p]+1);
	//printf("%6d %6d %10d %10d %10f\n", first_probe[p]+1, last_probe[p]+1,
	//     segStart[p], segEnd[p], segMean[p]);
    }
    // Free allocated objects //
    //printf("Free \n");
    free(first_probe);
    free(last_probe);
    free(sx);
    free(break_before);
    for ( p = 0; p < pMax; ++p)
    {
	free(C[p]);
    }
    free(C);
	
    for (p = 0; p < pMax; ++p)
    {
	free(M[p]);
    }
    free(M);

    free(sR);
    free(eR);
    
    return 0;
}
