/* MIT License

 * Copyright (c) 2018 - Daniel Peter Playne
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef REDUCTION_H
#define REDUCTION_H

// ---------- Find the root of a chain ----------
template <typename TP_out> //unsigned int* g_labels, const unsigned char *g_image
__device__ __inline__ unsigned int find_root(TP_out *labels, TP_out label) {
	// Resolve Label
	TP_out next = labels[label];

	// Follow chain
	while(label != next) {
		// Move to next
		label = next;
		next = labels[label];
	}

	// Return label
	return(label);
}

// ---------- Label Reduction ----------
template <typename TP_out> //unsigned int* g_labels, const unsigned char *g_image
__device__ __inline__ unsigned int reduction(TP_out *g_labels, TP_out label1, TP_out label2) {
	// Get next labels
	TP_out next1 = (label1 != label2) ? g_labels[label1] : 0;
	TP_out next2 = (label1 != label2) ? g_labels[label2] : 0;

	// Find label1
	while((label1 != label2) && (label1 != next1)) {
		// Adopt label
		label1 = next1;

		// Fetch next label
		next1 = g_labels[label1];
	}

	// Find label2
	while((label1 != label2) && (label2 != next2)) {
		// Adopt label
		label2 = next2;

		// Fetch next label
		next2 = g_labels[label2];
	}

	TP_out label3;
	// While Labels are different
	while(label1 != label2) {
		// Label 2 should be smallest
		if(label1 < label2) {
			// Swap Labels
			label1 = label1 ^ label2;
			label2 = label1 ^ label2;
			label1 = label1 ^ label2;
		}

		// AtomicMin label1 to label2
		label3 = atomicMin(&g_labels[label1], label2);
		label1 = (label1 == label3) ? label2 : label3;
	}

	// Return label1
	return(label1);
}

#endif // REDUCTION_H
