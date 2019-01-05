
#define SEQ_BUFFER_SIZE 10240

#define ALPHA 21
#define ALPHA2 441
#define INDEX1(pos, aa) ALPHA* pos + aa

#define INDEX2(left_pos, right_pos, left_label, right_label)              \
  nsingle +                                                               \
      ALPHA2*(((left_pos * (2 * ncol - left_pos - 1)) >> 1) + right_pos - \
              left_pos - 1) +                                             \
      ALPHA* left_label + right_label
#define INDEX6(pos1, pos2) \
  (pos1 * ncol - pos1 * (pos1 + 1) / 2 + pos2 - pos1 - 1) * ALPHA2

#define INDEX_FM(pos, aa, r) (pos * ALPHA + aa) * rank + r
#define INDEX_S(pos, aa) nvar_fm + pos* ALPHA + aa
