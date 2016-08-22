#include <stdlib.h>

#include "util.h"
#include "matrix.h"
#include "constraint.h"

void int_vec_init(int_vec *v, size_t size) {
	v->get = malloc(sizeof(int) * size);
	v->size = 0;
}

void int_vec_free(int_vec *v) {
	free(v->get);
}

void int_vec_push(int_vec *v, int val) {
	v->get[v->size++] = val;
}

//void int_arr_push(int *base, int *pos, int val, size_t num) {
//    memmove(pos + 1, pos, num - (pos - base));
//    *pos = val;
//}

//int cmp_int(const void *a, const void *b) {
//    return *((int *) a) - *((int *) b);
//}

void constraint_init(constraint *c, size_t mlsize, size_t mnlsize) {
    c->ml = malloc(sizeof(int_vec));
	int_vec_init(c->ml, mlsize);
    c->mnl = malloc(sizeof(int_vec));
	int_vec_init(c->mnl, mnlsize);
}

void constraint_free(constraint *c) {
	int_vec_free(c->ml);
    free(c->ml);
	int_vec_free(c->mnl);
    free(c->mnl);
	free(c);
}

void update_constraint(constraint **c, st_matrix *memb, double in,
        double out) {
    size_t h;
    size_t i;
    size_t k;
    double memb1;
    double memb2;
    for(k = 0; k < memb->ncol; ++k) {
        for(i = 0; i < memb->nrow; ++i) {
            for(h = 0; h < memb->nrow; ++h) {
                if(h == i) {
                    continue;
                }
                memb1 = get(memb, i, k);
                if(memb1 < in) {
                    continue;
                }
                memb2 = get(memb, h, k);
                if(memb2 >= in) {
                    if(!c[i]) {
                        c[i] = malloc(sizeof(constraint));
                        // TODO: use less memory?
                        constraint_init(c[i], memb->nrow,
                                memb->nrow);
                    } else if(bsearch(&h, c[i]->ml->get,
                                c[i]->ml->size, sizeof(int),
                                cmpint) || bsearch(&h, c[i]->mnl->get,
                                c[i]->mnl->size, sizeof(int),
                                cmpint)) {
                        continue;
                    }
                    // TODO: optimize int_vec_push in order to
                    // avoid using so many qsort
                    int_vec_push(c[i]->ml, h);
                    qsort(c[i]->ml->get, c[i]->ml->size,
                            sizeof(int), cmpint);
                    if(!c[h]) {
                        c[h] = malloc(sizeof(constraint));
                        // TODO: use less memory?
                        constraint_init(c[h], memb->nrow,
                                memb->nrow);
                    }
                    int_vec_push(c[h]->ml, i);
                    qsort(c[h]->ml->get, c[h]->ml->size,
                            sizeof(int), cmpint);
                } else if(memb2 <= out) {
                    if(!c[i]) {
                        c[i] = malloc(sizeof(constraint));
                        // TODO: use less memory?
                        constraint_init(c[i], memb->nrow,
                                memb->nrow);
                    } else if(bsearch(&h, c[i]->mnl->get,
                                c[i]->mnl->size, sizeof(int),
                                cmpint) || bsearch(&h, c[i]->ml->get,
                                c[i]->ml->size, sizeof(int),
                                cmpint)) {
                            continue;
                    }
                    int_vec_push(c[i]->mnl, h);
                    qsort(c[i]->mnl->get, c[i]->mnl->size,
                            sizeof(int), cmpint);
                    if(!c[h]) {
                        c[h] = malloc(sizeof(constraint));
                        // TODO: use less memory?
                        constraint_init(c[h], memb->nrow,
                                memb->nrow);
                    }
                    int_vec_push(c[h]->mnl, i);
                    qsort(c[h]->mnl->get, c[h]->mnl->size,
                            sizeof(int), cmpint);
                }
            }
        }
    }
}
