### SimbaE1 here!

This is going to be a TCEC competitive engine, just saying.
Currently is in PoC phase, but nearing the end of that.
The plan is basically:
1. NNUE derived policy head finds probable moves.
2. NNUE + MCTS tries to find the best move.
3. NNUE + AB looks for tactical sequences, and 'tells' #1 about them by increases q value.
