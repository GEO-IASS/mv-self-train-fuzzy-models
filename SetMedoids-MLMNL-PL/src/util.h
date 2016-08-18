#ifndef _UTIL_H_
#define _UTIL_H_

#pragma once

#include <stdio.h>
#include <stdbool.h>

#define FPOINT_OFFSET 1e-10

bool load_data(char *fname, double **matrix, size_t nrow, size_t ncol);

void print_mtx_d(double **matrix, size_t nrow, size_t ncol);

void fprint_mtx_d(FILE *file, double **matrix, size_t nrow,
					size_t ncol);

void print_mtx_size_t(size_t **matrix, size_t nrow, size_t ncol);

bool deq(double a, double b);

bool dgt(double a, double b);

bool dlt(double a, double b);

void mtxcpy_d(double **destination, double **source, size_t nrow,
        size_t ncol);

void mtxcpy_size_t(size_t **destination, size_t **source, size_t nrow,
        size_t ncol);

int max(int *vec, size_t size);

int cmpint(const void *a, const void *b);

void print_header(char *str, size_t size);

#endif /* _UTIL_H_ */
