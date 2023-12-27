\ Model scenario_one_upperbound_logit_one
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x1 + Constant
Subject To
 c1: x1 <= 2
 c2: - x1 <= 2
 c3: x2 <= 2
 c4: - x2 <= 2
 c5: - 3 x1 - x2 <= 5
Bounds
 x1 >= -2
 x2 >= -2
 Constant = 1
End
