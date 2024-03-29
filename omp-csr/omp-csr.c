/* -*- mode: C; mode: folding; fill-column: 70; -*- */
/* Copyright 2010-2011,  Georgia Institute of Technology, USA. */
/* Copyright 2013,  Regents of the University of California, USA. */
/* See COPYING for license. */
#include "../compat.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <assert.h>

#include <alloca.h>

static int64_t int64_fetch_add (int64_t* p, int64_t incr);
static int64_t int64_casval(int64_t* p, int64_t oldval, int64_t newval);
static int int64_cas(int64_t* p, int64_t oldval, int64_t newval);

#include "../graph500.h"
#include "../xalloc.h"
#include "../generator/graph_generator.h"
#include "../timer.h"

#include "bitmap.h"

#define MINVECT_SIZE 2
#define THREAD_BUF_LEN 16384
#define ALPHA 14
#define BETA  24

static int64_t maxvtx, nv, sz;
static int64_t * restrict xoff; /* Length 2*nv+2 */
static int64_t * restrict xadjstore; /* Length MINVECT_SIZE + (xoff[nv] == nedge) */
static int64_t * restrict xadj;

static void
find_nv (const struct packed_edge * restrict IJ, const int64_t nedge)
{
  maxvtx = -1;
  OMP("omp parallel") {
    int64_t k, gmaxvtx, tmaxvtx = -1;

    OMP("omp for")
      for (k = 0; k < nedge; ++k) {
	if (get_v0_from_edge(&IJ[k]) > tmaxvtx)
	  tmaxvtx = get_v0_from_edge(&IJ[k]);
	if (get_v1_from_edge(&IJ[k]) > tmaxvtx)
	  tmaxvtx = get_v1_from_edge(&IJ[k]);
      }
    gmaxvtx = maxvtx;
    while (tmaxvtx > gmaxvtx)
      gmaxvtx = int64_casval (&maxvtx, gmaxvtx, tmaxvtx);
  }
  nv = 1+maxvtx;
}

static int
alloc_graph (int64_t nedge)
{
  sz = (2*nv+2) * sizeof (*xoff);
  xoff = xmalloc_large_ext (sz);
  if (!xoff) return -1;
  return 0;
}

static void
free_graph (void)
{
  xfree_large (xadjstore);
  xfree_large (xoff);
}

#define XOFF(k) (xoff[2*(k)])
#define XENDOFF(k) (xoff[1+2*(k)])

static int64_t
prefix_sum (int64_t *buf)
{
  int nt, tid;
  int64_t slice_begin, slice_end, t1, t2, k;

  nt = omp_get_num_threads ();
  tid = omp_get_thread_num ();

  t1 = nv / nt;
  t2 = nv % nt;
  slice_begin = t1 * tid + (tid < t2? tid : t2);
  slice_end = t1 * (tid+1) + ((tid+1) < t2? (tid+1) : t2);

  buf[tid] = 0;
  for (k = slice_begin; k < slice_end; ++k)
    buf[tid] += XOFF(k);
  OMP("omp barrier");
  OMP("omp single")
    for (k = 1; k < nt; ++k)
      buf[k] += buf[k-1];
  if (tid)
    t1 = buf[tid-1];
  else
    t1 = 0;
  for (k = slice_begin; k < slice_end; ++k) {
    int64_t tmp = XOFF(k);
    XOFF(k) = t1;
    t1 += tmp;
  }
  OMP("omp flush (xoff)");
  OMP("omp barrier");
  return buf[nt-1];
}

static int
setup_deg_off (const struct packed_edge * restrict IJ, int64_t nedge)
{
  int err = 0;
  int64_t *buf = NULL;
  xadj = NULL;
  OMP("omp parallel") {
    int64_t k, accum;
    OMP("omp for")
      for (k = 0; k < 2*nv+2; ++k)
	xoff[k] = 0;
    OMP("omp for")
      for (k = 0; k < nedge; ++k) {
	int64_t i = get_v0_from_edge(&IJ[k]);
	int64_t j = get_v1_from_edge(&IJ[k]);
	if (i != j) { /* Skip self-edges. */
	  if (i >= 0)
	    OMP("omp atomic")
	      ++XOFF(i);
	  if (j >= 0)
	    OMP("omp atomic")
	      ++XOFF(j);
	}
      }
    OMP("omp single") {
      buf = alloca (omp_get_num_threads () * sizeof (*buf));
      if (!buf) {
	perror ("alloca for prefix-sum hosed");
	abort ();
      }
    }
    OMP("omp for")
      for (k = 0; k < nv; ++k)
	if (XOFF(k) < MINVECT_SIZE) XOFF(k) = MINVECT_SIZE;

    accum = prefix_sum (buf);

    OMP("omp for")
      for (k = 0; k < nv; ++k)
	XENDOFF(k) = XOFF(k);
    OMP("omp single") {
      XOFF(nv) = accum;
      if (!(xadjstore = xmalloc_large_ext ((XOFF(nv) + MINVECT_SIZE) * sizeof (*xadjstore))))
	err = -1;
      if (!err) {
	xadj = &xadjstore[MINVECT_SIZE]; /* Cheat and permit xadj[-1] to work. */
	for (k = 0; k < XOFF(nv) + MINVECT_SIZE; ++k)
	  xadjstore[k] = -1;
      }
    }
  }
  return !xadj;
}

static void
scatter_edge (const int64_t i, const int64_t j)
{
  int64_t where;
  where = int64_fetch_add (&XENDOFF(i), 1);
  xadj[where] = j;
}

static int
i64cmp (const void *a, const void *b)
{
  const int64_t ia = *(const int64_t*)a;
  const int64_t ib = *(const int64_t*)b;
  if (ia < ib) return -1;
  if (ia > ib) return 1;
  return 0;
}

static void
pack_vtx_edges (const int64_t i)
{
  int64_t kcur, k;
  if (XOFF(i)+1 >= XENDOFF(i)) return;
  qsort (&xadj[XOFF(i)], XENDOFF(i)-XOFF(i), sizeof(*xadj), i64cmp);
  kcur = XOFF(i);
  for (k = XOFF(i)+1; k < XENDOFF(i); ++k)
    if (xadj[k] != xadj[kcur])
      xadj[++kcur] = xadj[k];
  ++kcur;
  for (k = kcur; k < XENDOFF(i); ++k)
    xadj[k] = -1;
  XENDOFF(i) = kcur;
}

static void
pack_edges (void)
{
  int64_t v;

  OMP("omp for")
    for (v = 0; v < nv; ++v)
      pack_vtx_edges (v);
}

static void
gather_edges (const struct packed_edge * restrict IJ, int64_t nedge)
{
  OMP("omp parallel") {
    int64_t k;

    OMP("omp for")
      for (k = 0; k < nedge; ++k) {
	int64_t i = get_v0_from_edge(&IJ[k]);
	int64_t j = get_v1_from_edge(&IJ[k]);
	if (i >= 0 && j >= 0 && i != j) {
	  scatter_edge (i, j);
	  scatter_edge (j, i);
	}
      }

    pack_edges ();
  }
}

int
create_graph_from_edgelist (struct packed_edge *IJ, int64_t nedge)
{
  find_nv (IJ, nedge);
  if (alloc_graph (nedge)) return -1;
  if (setup_deg_off (IJ, nedge)) {
    xfree_large (xoff);
    return -1;
  }
  gather_edges (IJ, nedge);
  return 0;
}

static void
fill_bitmap_from_queue(bitmap_t *bm, int64_t *vlist, int64_t out, int64_t in)
{
  OMP("omp for")
    for (long q_index=out; q_index<in; q_index++)
      bm_set_bit_atomic(bm, vlist[q_index]);
}

static void
fill_queue_from_bitmap(bitmap_t *bm, int64_t *vlist, int64_t *out, int64_t *in,
		       int64_t *local)
{
  OMP("omp single") {
    *out = 0;
    *in = 0;
  }
  OMP("omp barrier");
  int64_t nodes_per_thread = (nv + omp_get_num_threads() - 1) /
			     omp_get_num_threads();
  int64_t i = nodes_per_thread * omp_get_thread_num();
  int local_index = 0;
  if (i < nv) {
    int64_t i_end = i + nodes_per_thread;
    if (i_end >= nv)
      i_end = nv;
    if (bm_get_bit(bm, i))
      local[local_index++] = i;
    i = bm_get_next_bit(bm,i);
    while ((i != -1) && (i < i_end)) {
      local[local_index++] = i;
      i = bm_get_next_bit(bm,i);
      if (local_index == THREAD_BUF_LEN) {
	int my_in = int64_fetch_add(in, THREAD_BUF_LEN);
	for (local_index=0; local_index<THREAD_BUF_LEN; local_index++) {
	  vlist[my_in + local_index] = local[local_index];
	}
	local_index = 0;
      }
    }
  }
  int my_in = int64_fetch_add(in, local_index);
  for (int i=0; i<local_index; i++) {
    vlist[my_in + i] = local[i];
  }
}

static int64_t
bfs_bottom_up_step(int64_t *bfs_tree, bitmap_t *past, bitmap_t *next)
{
  OMP("omp single") {
    bm_swap(past, next);
  }
  OMP("omp barrier");
  bm_reset(next);
  static int64_t awake_count;
  OMP("omp single")
    awake_count = 0;
  OMP("omp barrier");
#if 0
  int64_t NV_blk_sz = nv / I64_ELEMS_PER_CACHE_LINE;
  OMP("omp for reduction(+ : awake_count)")
  for (int64_t i=0; i<NV_blk_sz; i++) {
    int64_t NV_beg = i * I64_ELEMS_PER_CACHE_LINE;
    int64_t NV_end = MIN(nv, ((i + 1) * I64_ELEMS_PER_CACHE_LINE));
    
    zero(next, NV_beg, NV_end);

    for(int64_t k = 0; k < I64_ELEMS_PER_CACHE_LINE; k++) {  
      if (bfs_tree[NV_beg+k] == -1) {
        for (int64_t vo = XOFF(NV_beg+k); vo < XENDOFF(NV_beg+k); vo++) {
	    const int64_t j = xadj[vo];
	  if (bm_get_bit(past, j)) {
	    // printf("%lu\n",i);
	    bfs_tree[NV_beg + k] = j;
	    //bm_set_bit_atomic(next, NV_beg + k);
	    bm_set_bit(next, NV_beg + k);
	    awake_count++;
	    break;
	  }
	}
      }
    }
  }
#endif
#if defined(ZFILL_CACHE_LINES) && defined(__ARM_ARCH) && __ARM_ARCH >= 8
   int const tid = omp_get_thread_num();
   int const nthreads = omp_get_num_threads();
   size_t const chunk = nv / nthreads;
   int64_t NV_left = nv % I64_ELEMS_PER_CACHE_LINE;

   uint64_t * const zfill_limit = next->start + WORD_OFFSET((tid+1)*chunk) - U64_ZFILL_OFFSET;
	  
   OMP("omp for reduction(+ : awake_count)")
   for (int64_t i=0; i<nv; i+=I64_ELEMS_PER_CACHE_LINE) {
    uint64_t * const ustart = next->start + WORD_OFFSET(i);
    if (ustart + U64_ZFILL_OFFSET < zfill_limit)
       zfill_u64(ustart + U64_ZFILL_OFFSET);

    for(int64_t k = 0; k < I64_ELEMS_PER_CACHE_LINE; k++) {  
      if (bfs_tree[i+k] == -1) {
        for (int64_t vo = XOFF(i+k); vo < XENDOFF(i+k); vo++) {
	    const int64_t j = xadj[vo];
	  if (bm_get_bit(past, j)) {
	    // printf("%lu\n",i);
	    bfs_tree[i + k] = j;
	    //bm_set_bit_atomic(next, NV_beg + k);
	    bm_set_bit(next, i + k);
	    awake_count++;
	    break;
	  }
	}
      }
    }
  }
   /*
  if (NV_left > 0 && tid == 0)
  {
    for (int64_t i=nv-NV_left; i<nv; i++) {
      if (bfs_tree[i] == -1) {
	  for (int64_t vo = XOFF(i); vo < XENDOFF(i); vo++) {
	    const int64_t j = xadj[vo];
	  if (bm_get_bit(past, j)) {
	    // printf("%lu\n",i);
	    bfs_tree[i] = j;
	    //bm_set_bit_atomic(next, i);
	    bm_set_bit(next, i);
	    awake_count++;
	    break;
	  }
	}
      }
    }
  }
  */
#else
  OMP("omp for reduction(+ : awake_count)")
    for (int64_t i=0; i<nv; i++) {
      if (bfs_tree[i] == -1) {
	  for (int64_t vo = XOFF(i); vo < XENDOFF(i); vo++) {
	    const int64_t j = xadj[vo];
	  if (bm_get_bit(past, j)) {
	    // printf("%lu\n",i);
	    bfs_tree[i] = j;
	    //bm_set_bit_atomic(next, i);
	    bm_set_bit(next, i);
	    awake_count++;
	    break;
	  }
	}
      }
    }
#endif
  OMP("omp barrier");
  return awake_count;
}

static void
bfs_top_down_step(int64_t *bfs_tree, int64_t *vlist, int64_t *local, int64_t *k1_p, int64_t *k2_p)
{
  const int64_t oldk2 = *k2_p;
  int64_t kbuf = 0;
  OMP("omp barrier");
    OMP("omp for")
	for (int64_t k = *k1_p; k < oldk2; ++k) {
	  const int64_t v = vlist[k];
	  const int64_t veo = XENDOFF(v);
	  int64_t vo;
	  for (vo = XOFF(v); vo < veo; ++vo) {
	    const int64_t j = xadj[vo];
	    if (bfs_tree[j] == -1) {
	      if (int64_cas (&bfs_tree[j], -1, v)) {
		if (kbuf < THREAD_BUF_LEN) {
		  local[kbuf++] = j;
		} else {
		  int64_t voff = int64_fetch_add (k2_p, THREAD_BUF_LEN), vk;
		  assert (voff + THREAD_BUF_LEN <= nv);
		  for (vk = 0; vk < THREAD_BUF_LEN; ++vk)
		    vlist[voff + vk] = local[vk];
		  local[0] = j;
		  kbuf = 1;
		}
	      }
	    }
	  }
	}
  if (kbuf) {
	int64_t voff = int64_fetch_add (k2_p, kbuf), vk;
	assert (voff + kbuf <= nv);
	for (vk = 0; vk < kbuf; ++vk)
	  vlist[voff + vk] = local[vk];
  }
  OMP("omp single")
    *k1_p = oldk2;
  OMP("omp barrier");
  return;
}

int
make_bfs_tree (int64_t *bfs_tree_out, int64_t *max_vtx_out,
	       int64_t srcvtx)
{
  int64_t * restrict bfs_tree = bfs_tree_out;
  int err = 0;

  int64_t * restrict vlist = NULL;
  int64_t k1, k2;

  *max_vtx_out = maxvtx;

  vlist = xmalloc_large (nv * sizeof (*vlist));
  if (!vlist) return -1;

  vlist[0] = srcvtx;
  k1 = 0; k2 = 1;
  bfs_tree[srcvtx] = srcvtx;

  bitmap_t past, next;
  bm_init(&past, nv);
  bm_init(&next, nv);

  int64_t down_cutoff = nv / BETA;
  int64_t scout_count = XENDOFF(srcvtx) - XOFF(srcvtx);

  OMP("omp parallel shared(k1, k2, scout_count)") {
    int64_t k;
    int64_t nbuf[THREAD_BUF_LEN];
    int64_t awake_count = 1;
    int64_t edges_to_check = XOFF(nv);

    OMP("omp for")
      for (k = 0; k < srcvtx; ++k)
	bfs_tree[k] = -1;
    OMP("omp for")
      for (k = srcvtx+1; k < nv; ++k)
	bfs_tree[k] = -1;

    while (awake_count != 0) {
      // Top-down
      if (scout_count < ((edges_to_check - scout_count)/ALPHA)) {
	bfs_top_down_step(bfs_tree, vlist, nbuf, &k1, &k2);
	edges_to_check -= scout_count;
	awake_count = k2-k1;
      // Bottom-up
      } else {
	fill_bitmap_from_queue(&next, vlist, k1, k2);
	do {
	  awake_count = bfs_bottom_up_step(bfs_tree, &past, &next);
	} while ((awake_count > down_cutoff));
	fill_queue_from_bitmap(&next, vlist, &k1, &k2, nbuf);
	OMP("omp barrier");
      }
      // Count the number of edges in the frontier
      OMP("omp single")
	scout_count = 0;
      OMP("omp for reduction(+ : scout_count)")
      for (int64_t i=k1; i<k2; i++) {
	int64_t v = vlist[i];
	scout_count += XENDOFF(v) - XOFF(v);
      }
    }
  }

  bm_free(&past);
  bm_free(&next);
  xfree_large (vlist);

  return err;
}

void
destroy_graph (void)
{
  free_graph ();
}

#if defined(_OPENMP)
#if defined(__GNUC__)||defined(__INTEL_COMPILER)
int64_t
int64_fetch_add (int64_t* p, int64_t incr)
{
  return __sync_fetch_and_add (p, incr);
}
int64_t
int64_casval(int64_t* p, int64_t oldval, int64_t newval)
{
  return __sync_val_compare_and_swap (p, oldval, newval);
}
int
int64_cas(int64_t* p, int64_t oldval, int64_t newval)
{
  return __sync_bool_compare_and_swap (p, oldval, newval);
}
#else
/* XXX: These are not correct, but suffice for the above uses. */
int64_t
int64_fetch_add (int64_t* p, int64_t incr)
{
  int64_t t;
  OMP("omp critical") {
    t = *p;
    *p += incr;
  }
  OMP("omp flush (p)");
  return t;
}
int64_t
int64_casval(int64_t* p, int64_t oldval, int64_t newval)
{
  int64_t v;
  OMP("omp critical (CAS)") {
    v = *p;
    if (v == oldval)
      *p = newval;
  }
  OMP("omp flush (p)");
  return v;
}
int
int64_cas(int64_t* p, int64_t oldval, int64_t newval)
{
  int out = 0;
  OMP("omp critical (CAS)") {
    int64_t v = *p;
    if (v == oldval) {
      *p = newval;
      out = 1;
    }
  }
  OMP("omp flush (p)");
  return out;
}
#endif
#else
int64_t
int64_fetch_add (int64_t* p, int64_t incr)
{
  int64_t t = *p;
  *p += incr;
  return t;
}
int64_t
int64_casval(int64_t* p, int64_t oldval, int64_t newval)
{
  int64_t v = *p;
  if (v == oldval)
    *p = newval;
  return v;
}
int
int64_cas(int64_t* p, int64_t oldval, int64_t newval)
{
  int64_t v = *p;
  int out = 0;
  if (v == oldval) {
    *p = newval;
    out = 1;
  }
  return out;
}
#endif
