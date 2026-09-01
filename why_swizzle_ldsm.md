# K-Contiguous W, 128-Byte-Swizzled Shared Memory, and ldmatrix.x2

This note describes only the weight operand used by the M=9 tensor-core
kernel. The activation operand uses a different ldmatrix form and is outside
the scope of this note.

The relevant instruction is:

    ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16

The two important views of the weight are:

- Storage view: W[N,K], row-major FP8, with K contiguous.
- MMA view: B[K,N] = W transpose, the column-major B operand of
  mma.sync.aligned.m16n8k32.row.col.

No global-memory transpose or weight repacking is performed.

## 1. CTA and warp-level weight tiles

Every M=9 CTA stages one logical weight tile:

    W_cta[BN,BK]

The retained kernels use BK=128 and BN=32 or BN=64. One consumer warp q owns
eight W rows:

    W_warp = W_cta[8q : 8q+8, 0 : 128]

Thus:

- BN=32 has four consumer warps.
- BN=64 has eight consumer warps.
- Each consumer warp produces one M16-by-N8 output band.
- Each warp traverses BK128 as four K32 MMA steps.

A BN32 example is used below, but the shared-memory mapping repeats every
eight rows and is identical for each consumer warp in BN64.

## 2. Global-memory layout

For W[n,k] with logical shape [BN,BK]=[32,128]:

    address(W[n,k]) = base + n*128 + k

Each element occupies one byte because W is FP8 E4M3. Therefore:

- W[n,k+1] is the next byte.
- W[n+1,k] is 128 bytes away.
- Every W row is one contiguous 128-byte span.

Global memory is ordinary row-major storage:

                 K, contiguous
                 0                              127
    n=0          [--------------------------------] 128 bytes
    n=1          [--------------------------------] 128 bytes
    ...
    n=31         [--------------------------------] 128 bytes

This layout is already appropriate for TMA global-memory ingestion. The
swizzle is needed for the shared-memory consumer pattern, not to fix global
coalescing.

## 3. Eight 16-byte sectors per BK128 row

Split each 128-byte W row into eight 16-byte sectors:

    S0 = K[  0: 16]
    S1 = K[ 16: 32]
    S2 = K[ 32: 48]
    S3 = K[ 48: 64]
    S4 = K[ 64: 80]
    S5 = K[ 80: 96]
    S6 = K[ 96:112]
    S7 = K[112:128]

Each sector contains 16 FP8 values. It also spans four consecutive
four-byte shared-memory banks.

Without swizzling, the physical sector order is identical in every row:

    physical slot       0  1  2  3  4  5  6  7
    row 0 contains      S0 S1 S2 S3 S4 S5 S6 S7
    row 1 contains      S0 S1 S2 S3 S4 S5 S6 S7
    ...
    row 7 contains      S0 S1 S2 S3 S4 S5 S6 S7

Because the row stride is exactly 128 bytes, the start of every row has the
same shared-bank phase.

## 4. The 128-byte TMA swizzle

TMA deposits W into shared memory with
CU_TENSOR_MAP_SWIZZLE_128B. The kernel computes consumer addresses with the
same mapping.

For local shared row r and logical 16-byte sector s:

    physical_sector(r,s) = s XOR (r mod 8)

The byte offset inside the selected 16-byte sector is unchanged.

Equivalently, physical slot p in row r contains logical sector:

    logical_sector = p XOR (r mod 8)

The complete physical layout for rows 0 through 7 is:

    physical slot       0  1  2  3  4  5  6  7
    row 0               S0 S1 S2 S3 S4 S5 S6 S7
    row 1               S1 S0 S3 S2 S5 S4 S7 S6
    row 2               S2 S3 S0 S1 S6 S7 S4 S5
    row 3               S3 S2 S1 S0 S7 S6 S5 S4
    row 4               S4 S5 S6 S7 S0 S1 S2 S3
    row 5               S5 S4 S7 S6 S1 S0 S3 S2
    row 6               S6 S7 S4 S5 S2 S3 S0 S1
    row 7               S7 S6 S5 S4 S3 S2 S1 S0

The pattern repeats for rows 8 through 15, 16 through 23, and so on, because
only r mod 8 participates in the XOR. This exactly matches the N8 ownership of
one consumer warp.

For a fixed logical sector s, rows r=0 through 7 map it to:

    s XOR 0, s XOR 1, ..., s XOR 7

Those values enumerate physical sectors 0 through 7 exactly once. Since each
16-byte sector spans four banks, the eight row starts are dispersed across
all 32 banks instead of repeating one four-bank group.

## 5. One ldmatrix.x2 operation

Within one BK128 pipeline stage, a consumer warp iterates:

    kk = 0, 32, 64, 96

At each kk, the code forms the shared address supplied by lane l as:

    h     = (l >> 3) & 1
    r     = l & 7
    W row = 8q + r
    W K   = kk + 16h

and executes one ldmatrix.m8n8.x2.b16.

For x2, only lanes 0 through 15 provide the 16 row-start addresses:

| Address lanes | Matrix | W rows | Logical K start |
|---:|---|---|---:|
| 0 through 7 | B0 | 8q through 8q+7 | kk |
| 8 through 15 | B1 | 8q through 8q+7 | kk+16 |

All 32 lanes execute the instruction and receive registers. Lanes 16 through
31 do not provide additional row-start addresses for the x2 form.

At kk=0, the address-provider mapping is:

    B0, logical S0:
      lane 0 -> W row 8q+0, physical sector 0
      lane 1 -> W row 8q+1, physical sector 1
      lane 2 -> W row 8q+2, physical sector 2
      lane 3 -> W row 8q+3, physical sector 3
      lane 4 -> W row 8q+4, physical sector 4
      lane 5 -> W row 8q+5, physical sector 5
      lane 6 -> W row 8q+6, physical sector 6
      lane 7 -> W row 8q+7, physical sector 7

    B1, logical S1:
      lane  8 -> W row 8q+0, physical sector 1
      lane  9 -> W row 8q+1, physical sector 0
      lane 10 -> W row 8q+2, physical sector 3
      lane 11 -> W row 8q+3, physical sector 2
      lane 12 -> W row 8q+4, physical sector 5
      lane 13 -> W row 8q+5, physical sector 4
      lane 14 -> W row 8q+6, physical sector 7
      lane 15 -> W row 8q+7, physical sector 6

At later kk values the same rule applies to sector pairs S2/S3, S4/S5, and
S6/S7.

The source pointer is the physical swizzled address, but ldmatrix returns the
register fragment corresponding to the intended logical matrix.

## 6. Logical matrices loaded by x2

One m8n8.b16 matrix contains:

    8 rows x 8 packed-b16 elements
    = 8 rows x 16 FP8 values
    = 128 bytes

Therefore x2 loads two matrices:

    B0 = W[8q : 8q+8, kk    : kk+16]
    B1 = W[8q : 8q+8, kk+16 : kk+32]

Together they form exactly one warp-level N8-by-K32 FP8 footprint:

    W[8q : 8q+8, kk : kk+32]

In packed-b16 coordinates:

                         packed columns
                    0  1  2  3  4  5  6  7
                  +-------------------------+
    W row 8q+0   |        B0 or B1          |
    W row 8q+1   |                           |
    ...          |                           |
    W row 8q+7   |                           |
                  +-------------------------+

For B0, packed column c contains:

    W[8q+r, kk + 2c]
    W[8q+r, kk + 2c + 1]

For B1, packed column c contains:

    W[8q+r, kk + 16 + 2c]
    W[8q+r, kk + 16 + 2c + 1]

The instruction is non-transposed. W rows already represent the N columns of
the logical MMA B operand, so no additional ldmatrix transpose is required.

## 7. What every lane receives

Each lane receives two 32-bit registers:

    uint32 b[2]

Register b[0] contains that lane fragment from B0. Register b[1] contains the
corresponding fragment from B1.

Define:

    g = lane >> 2
    p = lane & 3

Then lane l=4g+p receives:

    W row = 8q + g

    b[0] = four FP8 values:
           W[8q+g, kk + 4p : kk + 4p + 4]

    b[1] = four FP8 values:
           W[8q+g, kk + 16 + 4p : kk + 16 + 4p + 4]

The four lanes in each group collectively cover one W row:

| Lanes | W row | b[0] covers | b[1] covers |
|---:|---:|---|---|
| 0 through 3 | 8q+0 | kk+0 through kk+15 | kk+16 through kk+31 |
| 4 through 7 | 8q+1 | kk+0 through kk+15 | kk+16 through kk+31 |
| 8 through 11 | 8q+2 | kk+0 through kk+15 | kk+16 through kk+31 |
| 12 through 15 | 8q+3 | kk+0 through kk+15 | kk+16 through kk+31 |
| 16 through 19 | 8q+4 | kk+0 through kk+15 | kk+16 through kk+31 |
| 20 through 23 | 8q+5 | kk+0 through kk+15 | kk+16 through kk+31 |
| 24 through 27 | 8q+6 | kk+0 through kk+15 | kk+16 through kk+31 |
| 28 through 31 | 8q+7 | kk+0 through kk+15 | kk+16 through kk+31 |

For example:

    lane 0:
      b[0] <- W[8q+0, kk+0  : kk+4]
      b[1] <- W[8q+0, kk+16 : kk+20]

    lane 1:
      b[0] <- W[8q+0, kk+4  : kk+8]
      b[1] <- W[8q+0, kk+20 : kk+24]

    lane 2:
      b[0] <- W[8q+0, kk+8  : kk+12]
      b[1] <- W[8q+0, kk+24 : kk+28]

    lane 3:
      b[0] <- W[8q+0, kk+12 : kk+16]
      b[1] <- W[8q+0, kk+28 : kk+32]

    lanes 4 through 7 repeat the same K ownership for W row 8q+1.
    ...
    lanes 28 through 31 repeat it for W row 8q+7.

This destination ownership uses all 32 lanes even though only lanes 0 through
15 supplied the x2 source addresses.

## 8. Four x2 loads cover one BK128 stage

The consumer warp repeats the operation four times:

| kk | b[0] logical K range | b[1] logical K range | Combined |
|---:|---|---|---|
| 0 | 0 through 15 | 16 through 31 | K 0 through 31 |
| 32 | 32 through 47 | 48 through 63 | K 32 through 63 |
| 64 | 64 through 79 | 80 through 95 | K 64 through 95 |
| 96 | 96 through 111 | 112 through 127 | K 96 through 127 |

Each x2 load supplies the B registers for one
mma.sync.aligned.m16n8k32 operation. Four such operations reduce one BK128
stage. The outer pipeline loop repeats BK128 stages until the full problem K
dimension has been accumulated.

## 9. End-to-end layout summary

    Global memory
      W[BN,BK], FP8
      K contiguous; one BK128 row is 128 bytes
             |
             | TMA with CU_TENSOR_MAP_SWIZZLE_128B
             v
    Shared memory
      same logical W[BN,BK] coordinates
      each logical 16-byte sector s is stored at physical sector
      s XOR (row mod 8)
             |
             | consumer warp q, kk in {0,32,64,96}
             | ldmatrix.m8n8.x2.b16
             v
    Registers in all 32 lanes
      b[0] = N8-by-K16 fragment for K=kk through kk+15
      b[1] = N8-by-K16 fragment for K=kk+16 through kk+31
             |
             | mma.sync.m16n8k32.row.col
             v
    FP32 accumulator tile
      one M16-by-N8 output band owned by consumer warp q

The essential distinction is:

- The 128-byte swizzle changes physical shared-memory placement.
- ldmatrix.x2 reconstructs the intended logical B0 and B1 fragments.
- The MMA sees the correct N8-by-K32 W data independent of the physical
  sector permutation.

