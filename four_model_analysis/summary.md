# Four-model test results
key hash verified: f117e9a7124fa5219e6384bd980da8252c31bd602150b7dcb489342a3143d0d2

## Success by model x position
| model | P1 | P2 | total |
|---|---|---|---|
| deep | 6/8 | 8/8 | 14/16 (88%) |
| R | 4/8 | 5/8 | 9/16 (56%) |
| C | 4/8 | 6/8 | 10/16 (62%) |
| C_trunc | 1/8 | 1/8 | 2/16 (12%) |

## Pairwise (two-sided Fisher, pooled P1+P2; Newcombe 95% CI of difference)
| comparison | rates | diff (pp) | CI (pp) | p | role |
|---|---|---|---|---|---|
| deep vs R | 14/16 vs 9/16 | +31 | [-0, +56] | 0.1134 | PRIMARY |
| deep vs C | 14/16 vs 10/16 | +25 | [-5, +51] | 0.2200 | secondary |
| deep vs C_trunc | 14/16 vs 2/16 | +75 | [+42, +88] | 0.0000 | exploratory |
| R vs C | 9/16 vs 10/16 | -6 | [-36, +25] | 1.0000 | secondary |
| R vs C_trunc | 9/16 vs 2/16 | +44 | [+11, +66] | 0.0233 | exploratory |
| C vs C_trunc | 10/16 vs 2/16 | +50 | [+16, +71] | 0.0091 | exploratory |

## Per position pairwise (exploratory; 8 vs 8)
- P1 deep 6/8 vs R 4/8: p=0.608
- P1 deep 6/8 vs C 4/8: p=0.608
- P1 deep 6/8 vs C_trunc 1/8: p=0.041
- P1 R 4/8 vs C 4/8: p=1.000
- P1 R 4/8 vs C_trunc 1/8: p=0.282
- P1 C 4/8 vs C_trunc 1/8: p=0.282
- P2 deep 8/8 vs R 5/8: p=0.200
- P2 deep 8/8 vs C 6/8: p=0.467
- P2 deep 8/8 vs C_trunc 1/8: p=0.001
- P2 R 5/8 vs C 6/8: p=1.000
- P2 R 5/8 vs C_trunc 1/8: p=0.119
- P2 C 6/8 vs C_trunc 1/8: p=0.041

## Failure types (counts)
| model | cube_pushed_away | dropped_after_grasp | edge_not_grasped |
|---|---|---|---|
| deep | 1 | 0 | 1 |
| R | 0 | 1 | 6 |
| C | 2 | 0 | 4 |
| C_trunc | 2 | 1 | 11 |

## Failure notes by model
- deep: V15(P1,edge_not_grasped,time out); V24(P1,cube_pushed_away,time out)
- R: V07(P1,dropped_after_grasp,grasp not precise dropped  time out); V42(P2,edge_not_grasped,); V47(P1,edge_not_grasped,time out); V50(P2,edge_not_grasped,time out); V54(P1,edge_not_grasped,time out); V59(P2,edge_not_grasped,time out); V63(P1,edge_not_grasped,time out)
- C: V20(P2,cube_pushed_away,grasp not precise cube pushed away time out); V23(P1,edge_not_grasped,time out); V29(P1,edge_not_grasped,time out); V36(P2,cube_pushed_away,grasp not precise cube pushed away time out); V38(P1,edge_not_grasped,time out); V48(P1,edge_not_grasped,time out)
- C_trunc: V04(P2,edge_not_grasped,time our); V05(P1,edge_not_grasped,time out); V11(P2,cube_pushed_away,cube pushed out of workspace time out); V16(P1,edge_not_grasped,time out); V18(P2,edge_not_grasped,time out); V21(P1,edge_not_grasped,time out); V25(P2,edge_not_grasped,time out); V30(P1,edge_not_grasped,time out); V33(P2,edge_not_grasped,time out); V39(P1,edge_not_grasped,time out); V41(P2,edge_not_grasped,time out); V45(P1,edge_not_grasped,time out); V51(P2,dropped_after_grasp,dropped cube out of workspace); V64(P1,cube_pushed_away,cube pushed out of workspace)

## Time order: successes per 8-block half / block
- deep: b1o b2o b3o b4x b5o b6x b7o b8o b9o b10o b11o b12o b13o b14o b15o b16o
- R: b1o b2x b3o b4o b5o b6o b7o b8o b9o b10o b11x b12x b13x b14x b15x b16x
- C: b1o b2o b3o b4o b5x b6x b7o b8x b9x b10x b11o b12x b13o b14o b15o b16o
- C_trunc: b1x b2x b3x b4x b5x b6x b7x b8x b9x b10x b11x b12x b13x b14o b15o b16x

## Date factor
days: ['2026-11-19']

## Early end / outside window
outside_time_window trials: []
