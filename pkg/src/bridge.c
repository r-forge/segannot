
#include "FindOptimalSegmentations.h"

void bridge_FindOptimalSegmentations(double * x, int * sR, int * eR, int * nMax, int * pMax, 
	int * idPath, double * cost)
{
    //printf("nMax: %d\n", nMax[0]);
	//printf("pMax: %d\n", pMax[0]);
	//printf("Data: %f ... %f\n", x[0], x[nMax[0]-1]);
	//printf("Region Start: %d - ... - %d\n", sR[0], sR[pMax[0]-1]);
	//printf("Region End: %d - ... - %d\n", eR[0], eR[pMax[0]-1]);

	FindOptimalSegmentations(x, (unsigned *) sR, (unsigned *) eR, (unsigned) *nMax, (unsigned) *pMax, 
				(unsigned *) idPath, cost);
}
