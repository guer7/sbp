function [H, HI, D1, D2, M, Q, e_1, e_m, S_1, S_m, x, h] = sbp_optimal(order, N, L, narrowing)
% Construct optimal SBP operators
% [H, HI, D1, D2, M, Q, e_1, e_m, S_1, S_m, x, h] = sbp_optimal(order, N, L, narrowing)
%
% Input arguments:
% order    : Order of accuracy in the interior (6 or 8)
% N        : Number of grid points
% L        : Domain length
% narrowing: Width of internal stencil in D2
%            0 - wide
%            1 - two additional stencil points (one on each side) compared
%            to minimally narrow
%            2 - minimally narrow 
%
% Output arguments:
% H, HI, D1, D2, M, Q, e_1, e_m, S_1, S_m: optimal SBP operators
% x                                      : Grid
% h                                      : Interior grid spacing

switch order
    case 6
        [H, HI, D1, D2, M, Q, e_1, e_m, S_1, S_m, x, h] = d2_noneq_6(N,L,narrowing);
    case 8
        [H, HI, D1, D2, M, Q, e_1, e_m, S_1, S_m, x, h] = d2_noneq_8(N,L,narrowing);
    otherwise
        error('optimal SBP operator not implemented');
end

end