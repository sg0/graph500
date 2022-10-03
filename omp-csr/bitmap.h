/* Copyright 2013,  Regents of the University of California, USA. */
/* See COPYING for license. */
#ifndef BITMAP_H
#define BITMAP_H

#include "../compat.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(ZFILL_CACHE_LINES) && defined(__ARM_ARCH) && __ARM_ARCH >= 8
#ifndef CACHE_LINE_SIZE_BYTES
#define CACHE_LINE_SIZE_BYTES   256
#endif
/* The zfill distance must be large enough to be ahead of the L2 prefetcher */
static const int ZFILL_DISTANCE = 100;

/* x-byte cache lines */
static const int I32_ELEMS_PER_CACHE_LINE = CACHE_LINE_SIZE_BYTES / sizeof(int32_t);
static const int I64_ELEMS_PER_CACHE_LINE = CACHE_LINE_SIZE_BYTES / sizeof(int64_t);
static const int U64_ELEMS_PER_CACHE_LINE = CACHE_LINE_SIZE_BYTES / sizeof(uint64_t);
static const int FLT_ELEMS_PER_CACHE_LINE = CACHE_LINE_SIZE_BYTES / sizeof(float);
static const int DBL_ELEMS_PER_CACHE_LINE = CACHE_LINE_SIZE_BYTES / sizeof(double);

/* Offset from a[j] to zfill */
static const int I32_ZFILL_OFFSET = ZFILL_DISTANCE * I32_ELEMS_PER_CACHE_LINE;
static const int I64_ZFILL_OFFSET = ZFILL_DISTANCE * I64_ELEMS_PER_CACHE_LINE;
static const int U64_ZFILL_OFFSET = ZFILL_DISTANCE * U64_ELEMS_PER_CACHE_LINE;
static const int FLT_ZFILL_OFFSET = ZFILL_DISTANCE * FLT_ELEMS_PER_CACHE_LINE;
static const int DBL_ZFILL_OFFSET = ZFILL_DISTANCE * DBL_ELEMS_PER_CACHE_LINE;

static inline void zfill_i32(int32_t * a) 
{ asm volatile("dc zva, %0": : "r"(a)); }

static inline void zfill_i64(int64_t * a) 
{ asm volatile("dc zva, %0": : "r"(a)); }

static inline void zfill_u64(uint64_t * a) 
{ asm volatile("dc zva, %0": : "r"(a)); }

static inline void zfill_flt(float * a) 
{ asm volatile("dc zva, %0": : "r"(a)); }

static inline void zfill_dbl(double * a) 
{ asm volatile("dc zva, %0": : "r"(a)); }
#endif
/*  Implemented in .h (without a .c) to allow optimizing compiler to inline
    For best performance, compile with -O3 or equivalent                     */

extern int int64_cas(int64_t* p, int64_t oldval, int64_t newval);

#define WORD_OFFSET(n) (n/64)
#define BIT_OFFSET(n) (n & 0x3f)

typedef struct {
  uint64_t *start;
  uint64_t *end;
} bitmap_t;

static inline void
bm_reset(bitmap_t* bm)
{
  OMP("omp for")
    for(uint64_t* it=bm->start; it<bm->end; it++)
      *it = 0;
}

static inline void
bm_init(bitmap_t* bm, int size)
{
  int num_longs = (size + 63) / 64;
  bm->start = (uint64_t*) malloc(sizeof(uint64_t) * num_longs);
  bm->end = bm->start + num_longs;
  bm_reset(bm);
}

static inline uint64_t
bm_get_bit(bitmap_t* bm, long pos)
{
  return bm->start[WORD_OFFSET(pos)] & (1l<<BIT_OFFSET(pos));
}

static inline long
bm_get_next_bit(bitmap_t* bm, long pos)
{
  long next = pos;
  int bit_offset = BIT_OFFSET(pos);
  uint64_t *it = bm->start + WORD_OFFSET(pos);
  uint64_t temp = (*it);
  if (bit_offset != 63) {
    temp = temp >> (bit_offset+1);
  } else {
    temp = 0;
  }
  if (!temp) {
    next = (next & 0xffffffc0);
    while (!temp) {
      it++;
      if (it >= bm->end)
        return -1;
      temp = *it;
      next += 64;
    }
  } else {
    next++;
  }
  while(!(temp&1)) {
    temp = temp >> 1;
    next++;
  }
  return next;
}

static inline void
bm_set_bit(bitmap_t* bm, long pos)
{
  bm->start[WORD_OFFSET(pos)] |= ((uint64_t) 1l<<BIT_OFFSET(pos));
}

static inline void
bm_set_bit_atomic(bitmap_t* bm, long pos)
{
  uint64_t old_val, new_val;
  uint64_t *loc = bm->start + WORD_OFFSET(pos);
  do {
    old_val = *loc;
    new_val = old_val | ((uint64_t) 1l<<BIT_OFFSET(pos));
  } while(!int64_cas((int64_t*) loc, old_val, new_val));
}

#if defined(ZFILL_CACHE_LINES) && defined(__ARM_ARCH) && __ARM_ARCH >= 8
static inline void
zero(bitmap_t* bm, long beg, long end) 
{
  uint64_t * const zfill_limit = bm->start + WORD_OFFSET(end) - U64_ZFILL_OFFSET;
  uint64_t * const ustart = bm->start + WORD_OFFSET(beg);

  if (ustart + U64_ZFILL_OFFSET < zfill_limit)
    zfill_u64(ustart + U64_ZFILL_OFFSET);
}
#endif

static inline void bm_swap(bitmap_t* a, bitmap_t* b)
{
  uint64_t* temp;
  temp = a->start;
  a->start = b->start;
  b->start = temp;
  temp = a->end;
  a->end = b->end;
  b->end = temp;
}

static inline void
bm_free(bitmap_t* bm)
{
  free(bm->start);
}

#endif // BITMAP_H
