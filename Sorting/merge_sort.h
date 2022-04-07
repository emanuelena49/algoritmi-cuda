#pragma once
#include <memory>
#ifndef MERGE_SORT
#define MERGE_SORT

void merge_sort (int* v, int vLength) {

	int* support;
	support = (int*) malloc(vLength*sizeof(int));

	merge_sort_rec(v, 0, vLength, support);

	free(support);
}

void merge_sort_rec (int* data, int start, int end, int* support) {
	if (start < end) {
		int middle = (start + end - 1) / 2;
		merge_sort_rec(data, start, middle, support);
		merge_sort_rec(data, middle, end, support);
		merge(data, start, middle, end, support);
	}
}

void merge (int* data, int start, int middle, int end, int* support) {
	
	int i1 = start, i2 = middle;

	for (int i = start; i < end; i++)
	{
		if (i2 == end || (i1 < middle && data[i1] <= data[i2])) {
			support[i] = data[i1];
			i1++;
		}
		else {
			support[i] = data[i2];
			i2++;
		}
	}

	for (int i = start; i < end; i++) {
		data[i] = support[i];
	}
}

#endif // !MERGE_SORT
